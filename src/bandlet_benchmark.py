"""
bandlet_benchmark.py
=====================
Standalone benchmark script -- run this on each machine/GPU, it saves ONE
.npz file with all raw numerical results. Send the .npz files back; plotting
happens separately once every run is in.

Two experiments:
  1. TIMING (most important): full bandlet pipeline (encode+decode) time
     vs. image size, on whatever CPU and GPU (if any) are present on this
     machine. Uses a procedurally-generated test image with genuinely
     constant complexity-per-unit-area at every size (no upsampling, no
     external downloads -- fully reproducible across machines), so the
     timing trend isn't confounded by content getting relatively smoother
     at larger sizes.
  2. SPARSITY (next important): sorted coefficient-magnitude decay curves,
     bandlet vs. several classical wavelet families (Haar, Daubechies-4/6/8,
     a biorthogonal 9/7, Coiflets), via PyWavelets, on real photographs.
     Steeper decay = sparser representation = better compression potential.

Requirements:
    pip install torch PyWavelets scikit-image numpy

Usage:
    python bandlet_benchmark.py --out results_A100.npz
    python bandlet_benchmark.py --out results_T4.npz --sizes 64,128,256,512,1024,2048
    python bandlet_benchmark.py --out results_cpu_only.npz --no-cuda

Run once per machine. The script auto-detects CUDA and benchmarks both CPU
and GPU (if present) for the timing experiment. The sparsity experiment
runs once on the fastest available device (results are numerically the
same regardless of device, up to floating-point noise -- see the notes at
the bottom of this file).
"""

import argparse
import time
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)


# ============================================================================
# CORE BANDLET TRANSFORM
# Identical, unmodified logic to the validated reference notebook
# (gpu_bandlet_transform.ipynb) -- exact invertibility already proven there
# on CPU and 5 GPU types. Copied here as a plain script with self-tests
# stripped out, so this file has no dependency on the notebook.
# ============================================================================

DIRS = [-3, -2, -1, 0, 1, 2, 3]   # candidate integer shear slopes
MIN_BLOCK = 4                      # smallest allowed quadtree leaf
SKIP_K = 999.0                     # sentinel marking a "no shear, no lift" leaf


def lift53_forward(x, dim):
    """Perfect-reconstruction 5/3 lifting split along `dim` (size even)."""
    x = x.transpose(dim, -1)
    s = x[..., 0::2].clone()
    d = x[..., 1::2].clone()
    L = s.shape[-1]
    s_ext = torch.cat([s, s[..., -1:]], dim=-1)
    d = d - 0.5 * (s_ext[..., 0:L] + s_ext[..., 1:L + 1])
    d_ext = torch.cat([d[..., 0:1], d], dim=-1)
    s = s + 0.25 * (d_ext[..., 0:L] + d_ext[..., 1:L + 1])
    return s.transpose(dim, -1), d.transpose(dim, -1)


def lift53_inverse(s, d, dim):
    s = s.transpose(dim, -1)
    d = d.transpose(dim, -1)
    L = s.shape[-1]
    d_ext = torch.cat([d[..., 0:1], d], dim=-1)
    s = s - 0.25 * (d_ext[..., 0:L] + d_ext[..., 1:L + 1])
    s_ext = torch.cat([s, s[..., -1:]], dim=-1)
    d = d + 0.5 * (s_ext[..., 0:L] + s_ext[..., 1:L + 1])
    N = 2 * s.shape[-1]
    x = torch.empty(s.shape[:-1] + (N,), dtype=s.dtype, device=s.device)
    x[..., 0::2] = s
    x[..., 1::2] = d
    return x.transpose(dim, -1)


def dwt2d_level(img):
    s, d = lift53_forward(img, dim=-1)
    LL, LH = lift53_forward(s, dim=-2)
    HL, HH = lift53_forward(d, dim=-2)
    return LL, LH, HL, HH


def idwt2d_level(LL, LH, HL, HH):
    s = lift53_inverse(LL, LH, dim=-2)
    d = lift53_inverse(HL, HH, dim=-2)
    return lift53_inverse(s, d, dim=-1)


def shear_batch(blocks, ks):
    B, H, W = blocks.shape
    device = blocks.device
    rows = torch.arange(H, device=device).view(1, H, 1).float()
    shift = torch.round(ks.view(B, 1, 1) * rows).long().expand(B, H, W)
    col_idx = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
    src_idx = torch.remainder(col_idx - shift, W)
    return torch.gather(blocks, 2, src_idx)


def unshear_batch(blocks, ks):
    B, H, W = blocks.shape
    device = blocks.device
    rows = torch.arange(H, device=device).view(1, H, 1).float()
    shift = torch.round(ks.view(B, 1, 1) * rows).long().expand(B, H, W)
    col_idx = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
    src_idx = torch.remainder(col_idx + shift, W)
    return torch.gather(blocks, 2, src_idx)


def best_leaf_batched(blocks, dirs=DIRS):
    B, H, W = blocks.shape
    device = blocks.device
    nd = len(dirs)
    ks = torch.tensor(dirs, device=device, dtype=torch.float32)

    blocks_rep = blocks.unsqueeze(1).expand(B, nd, H, W).reshape(B * nd, H, W)
    ks_rep = ks.view(1, nd).expand(B, nd).reshape(B * nd)

    sheared = shear_batch(blocks_rep, ks_rep)
    s, d = lift53_forward(sheared, dim=1)
    cost = s.abs().sum(dim=(1, 2)) + d.abs().sum(dim=(1, 2))
    cost = cost.view(B, nd)

    s_skip = blocks[:, 0::2, :]
    d_skip = blocks[:, 1::2, :]
    cost_skip = s_skip.abs().sum(dim=(1, 2)) + d_skip.abs().sum(dim=(1, 2))

    all_cost = torch.cat([cost, cost_skip.view(B, 1)], dim=1)
    best_idx = torch.argmin(all_cost, dim=1)
    best_cost = all_cost.gather(1, best_idx.view(B, 1)).squeeze(1)
    is_skip = (best_idx == nd)
    safe_idx = torch.where(is_skip, torch.zeros_like(best_idx), best_idx)

    Hh = H // 2
    s = s.view(B, nd, Hh, W)
    d = d.view(B, nd, Hh, W)
    idx = safe_idx.view(B, 1, 1, 1).expand(B, 1, Hh, W)
    s_lifted = s.gather(1, idx).squeeze(1)
    d_lifted = d.gather(1, idx).squeeze(1)

    is_skip3 = is_skip.view(B, 1, 1)
    best_s = torch.where(is_skip3, s_skip, s_lifted)
    best_d = torch.where(is_skip3, d_skip, d_lifted)
    best_k = torch.where(is_skip, torch.full((B,), SKIP_K, device=device), ks[safe_idx])
    return best_cost, best_k, best_s, best_d


def leaf_inverse_batched(s, d, ks, H):
    is_skip = (ks == SKIP_K)
    lifted = lift53_inverse(s, d, dim=1)
    B, Hh, W = s.shape
    raw = torch.empty(B, H, W, dtype=s.dtype, device=s.device)
    raw[:, 0::2, :] = s
    raw[:, 1::2, :] = d
    out = torch.where(is_skip.view(-1, 1, 1), raw, lifted)
    ks_for_unshear = torch.where(is_skip, torch.zeros_like(ks), ks)
    return unshear_batch(out, ks_for_unshear)


def blocks_from_grid(sub, bsz):
    N = sub.shape[0]
    G = N // bsz
    x = sub.view(G, bsz, G, bsz).permute(0, 2, 1, 3).reshape(G * G, bsz, bsz)
    return x, G


def segment_subband(sub, lam=1.0, dirs=DIRS, min_block=MIN_BLOCK):
    N = sub.shape[0]
    levels = []

    bsz = min_block
    blocks, G = blocks_from_grid(sub, bsz)
    cost, k, s, d = best_leaf_batched(blocks, dirs)
    total_cost = (cost + lam).view(G, G)
    is_leaf = torch.ones(G, G, dtype=torch.bool, device=sub.device)
    levels.append(dict(bsz=bsz, G=G, total_cost=total_cost, is_leaf=is_leaf,
                        is_leaf_host=is_leaf.cpu().tolist(),
                        k=k.view(G, G), k_host=k.view(G, G).cpu().tolist(), s=s, d=d))

    while bsz < N:
        bsz *= 2
        blocks, G = blocks_from_grid(sub, bsz)
        cost, k, s, d = best_leaf_batched(blocks, dirs)
        leaf_total = (cost + lam).view(G, G)

        prev = levels[-1]
        child_cost = prev['total_cost'].view(G, 2, G, 2).sum(dim=(1, 3))
        leaf_here = leaf_total <= child_cost
        total_cost = torch.where(leaf_here, leaf_total, child_cost)

        levels.append(dict(bsz=bsz, G=G, total_cost=total_cost, is_leaf=leaf_here,
                            is_leaf_host=leaf_here.cpu().tolist(),
                            k=k.view(G, G), k_host=k.view(G, G).cpu().tolist(), s=s, d=d))
    return levels


def collect_leaves(levels):
    top = len(levels) - 1
    out = []

    def recurse(level_idx, gr, gc):
        lvl = levels[level_idx]
        bsz = lvl['bsz']
        if lvl['is_leaf_host'][gr][gc] or level_idx == 0:
            flat = gr * lvl['G'] + gc
            out.append((gr * bsz, gc * bsz, bsz, lvl['k_host'][gr][gc],
                         lvl['s'][flat], lvl['d'][flat]))
        else:
            recurse(level_idx - 1, 2 * gr, 2 * gc)
            recurse(level_idx - 1, 2 * gr, 2 * gc + 1)
            recurse(level_idx - 1, 2 * gr + 1, 2 * gc)
            recurse(level_idx - 1, 2 * gr + 1, 2 * gc + 1)

    recurse(top, 0, 0)
    return out


def bandlet_forward(img, lam=1.0, dirs=DIRS, min_block=MIN_BLOCK):
    LL, LH, HL, HH = dwt2d_level(img)
    trees = {name: segment_subband(sub, lam, dirs, min_block)
             for name, sub in [('LH', LH), ('HL', HL), ('HH', HH)]}
    return LL, trees


def bandlet_inverse(LL, trees, threshold=None):
    """Cross-subband batched inverse (see notebook sections 6/11 for the
    5-round profiling story behind this specific implementation)."""
    N = LL.shape[-1]
    device = LL.device

    from collections import defaultdict
    by_bsz = defaultdict(list)
    for name in ('LH', 'HL', 'HH'):
        levels = trees[name]
        by_level = defaultdict(list)
        for level_idx, ref in ((r[0], r) for r in _collect_leaf_refs(levels)):
            by_level[level_idx].append(ref)
        for level_idx, refs in by_level.items():
            lvl = levels[level_idx]
            bsz = lvl['bsz']
            flat_idxs = torch.tensor([r[1] for r in refs], device=device, dtype=torch.long)
            s = lvl['s'].index_select(0, flat_idxs)
            d = lvl['d'].index_select(0, flat_idxs)
            ks = torch.tensor([r[5] for r in refs], device=device, dtype=torch.float32)
            r0s = [r[2] for r in refs]
            c0s = [r[3] for r in refs]
            by_bsz[bsz].append((name, s, d, ks, r0s, c0s))

    outs = {name: torch.zeros(N, N, device=device) for name in ('LH', 'HL', 'HH')}

    for bsz, pieces in by_bsz.items():
        names = [n for (n, s, d, ks, r0s, c0s) in pieces for _ in r0s]
        s_all = torch.cat([p[1] for p in pieces], dim=0)
        d_all = torch.cat([p[2] for p in pieces], dim=0)
        ks_all = torch.cat([p[3] for p in pieces], dim=0)
        r0s_all = [r0 for p in pieces for r0 in p[4]]
        c0s_all = [c0 for p in pieces for c0 in p[5]]

        if threshold is not None:
            s_all = torch.where(s_all.abs() >= threshold, s_all, torch.zeros_like(s_all))
            d_all = torch.where(d_all.abs() >= threshold, d_all, torch.zeros_like(d_all))
        blocks = leaf_inverse_batched(s_all, d_all, ks_all, bsz)

        B = len(r0s_all)
        r0s_t = torch.tensor(r0s_all, device=device)
        c0s_t = torch.tensor(c0s_all, device=device)
        rng = torch.arange(bsz, device=device)
        row_idx = (r0s_t.view(-1, 1, 1) + rng.view(1, bsz, 1)).expand(-1, bsz, bsz)
        col_idx = (c0s_t.view(-1, 1, 1) + rng.view(1, 1, bsz)).expand(-1, bsz, bsz)
        flat_idx = (row_idx * N + col_idx).reshape(B, -1)
        flat_blocks = blocks.reshape(B, -1)

        for name in ('LH', 'HL', 'HH'):
            sel = [i for i, n in enumerate(names) if n == name]
            if not sel:
                continue
            sel_t = torch.tensor(sel, device=device)
            outs[name].view(-1).scatter_(0, flat_idx[sel_t].reshape(-1), flat_blocks[sel_t].reshape(-1))

    return idwt2d_level(LL, outs['LH'], outs['HL'], outs['HH'])


def _collect_leaf_refs(levels):
    top = len(levels) - 1
    out = []

    def recurse(level_idx, gr, gc):
        lvl = levels[level_idx]
        bsz = lvl['bsz']
        if lvl['is_leaf_host'][gr][gc] or level_idx == 0:
            flat = gr * lvl['G'] + gc
            out.append((level_idx, flat, gr * bsz, gc * bsz, bsz, lvl['k_host'][gr][gc]))
        else:
            recurse(level_idx - 1, 2 * gr, 2 * gc)
            recurse(level_idx - 1, 2 * gr, 2 * gc + 1)
            recurse(level_idx - 1, 2 * gr + 1, 2 * gc)
            recurse(level_idx - 1, 2 * gr + 1, 2 * gc + 1)

    recurse(top, 0, 0)
    return out


# ============================================================================
# TEST CONTENT
# ============================================================================

def periodic_scale_invariant_image(N, period=16, device='cpu', seed=0):
    """Fixed spatial period, independent of N -- edge density per unit area
    is exactly constant by construction at every scale. Used for the timing
    experiment specifically so the size-vs-time trend isn't confounded by
    content becoming relatively smoother at larger N (see the notebook's
    section 14 for why this matters -- an earlier super-resolution-based
    pyramid had exactly this confound)."""
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(N, device=device).float(),
                             torch.arange(N, device=device).float(), indexing='ij')
    img = (torch.sin(xx * (2 * math.pi / period)) +
           torch.sin(yy * (2 * math.pi / period * 1.3))) > 0.3
    img = img.float() + 0.02 * torch.randn(N, N, device=device)
    return img.clamp(0.0, 1.0)


def load_real_images(size=512):
    """Real photographs, bundled with scikit-image (no downloads)."""
    from skimage import data as skdata
    out = {}
    for name in ['camera', 'brick', 'checkerboard']:
        img = getattr(skdata, name)()
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32) / 255.0
        H, W = img.shape
        top = max(0, (H - size) // 2)
        left = max(0, (W - size) // 2)
        img = img[top:top + size, left:left + size]
        if img.shape != (size, size):
            reps = (size // img.shape[0] + 1, size // img.shape[1] + 1)
            img = np.tile(img, reps)[:size, :size]
        out[name] = img
    return out


# ============================================================================
# EXPERIMENT 1 (MOST IMPORTANT): size vs time, CPU and GPU
# ============================================================================

def run_timing_experiment(sizes, devices, lam=0.02, warmup=True):
    """Returns dict: {device_name: {size: {'total_ms': ..., 'encode_ms': ...,
    'decode_ms': ..., 'max_err': ..., 'n_leaves': ...}}}"""
    results = {}
    for device in devices:
        dev_name = device.type
        print(f"\n[timing] device={dev_name}")
        results[dev_name] = {}
        for size in sizes:
            img = periodic_scale_invariant_image(size, device=device, seed=1)

            if device.type == 'cuda':
                if warmup:
                    _ = bandlet_inverse(*bandlet_forward(img, lam=lam))
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            LL, trees = bandlet_forward(img, lam=lam)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            rec = bandlet_inverse(LL, trees)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            err = (img - rec).abs().max().item()
            n_leaves = sum(len(collect_leaves(trees[b])) for b in trees)

            row = dict(encode_ms=1000 * (t1 - t0), decode_ms=1000 * (t2 - t1),
                       total_ms=1000 * (t2 - t0), max_err=err, n_leaves=n_leaves)
            results[dev_name][size] = row
            print(f"  size={size:5d}  encode={row['encode_ms']:8.2f}ms  "
                  f"decode={row['decode_ms']:7.2f}ms  total={row['total_ms']:8.2f}ms  "
                  f"err={err:.2e}  leaves={n_leaves}")
    return results


# ============================================================================
# EXPERIMENT 2 (NEXT IMPORTANT): sparsity / coefficient-magnitude decay,
# bandlet vs multiple classical wavelet families
# ============================================================================

# Naming note: "Mallat" isn't a specific wavelet filter -- Stephane Mallat's
# name is most closely associated with the biorthogonal 9/7 (CDF) wavelet,
# the JPEG2000 default and a centerpiece example in his textbook "A Wavelet
# Tour of Signal Processing". That's what's used here under that label;
# rename/swap the PyWavelets string below if a different wavelet was meant.
WAVELETS = {
    'Haar': 'haar',
    'Daubechies-4': 'db2',    # 4-tap filter
    'Daubechies-6': 'db3',    # 6-tap filter
    'Daubechies-8': 'db4',    # 8-tap filter
    'Biorthogonal 9/7 (Mallat/JPEG2000)': 'bior4.4',
    'Coiflet-2': 'coif2',
}


def classical_wavelet_coeffs(img_np, wavelet_name):
    import pywt
    LL, (LH, HL, HH) = pywt.dwt2(img_np, wavelet_name, mode='periodization')
    return np.concatenate([LH.ravel(), HL.ravel(), HH.ravel()])


def bandlet_coeffs(img, lam=0.02, device='cpu'):
    LL, trees = bandlet_forward(img, lam=lam)
    vals = []
    for name in ('LH', 'HL', 'HH'):
        for (_, _, _, _, s, d) in collect_leaves(trees[name]):
            vals.append(s.reshape(-1))
            vals.append(d.reshape(-1))
    return torch.cat(vals).cpu().numpy()


def sorted_magnitude_curve(coeffs, n_samples=500):
    """Log-rank-spaced samples along the descending-sorted |coefficient|
    curve -- compact enough to save directly (a few KB), sufficient to
    redraw the full log-log decay curve faithfully."""
    mags = np.sort(np.abs(coeffs))[::-1]
    M = len(mags)
    n_samples = min(n_samples, M)
    ranks = np.unique(np.geomspace(1, M, n_samples).astype(int))
    return ranks.astype(np.int64), mags[ranks - 1].astype(np.float32)


def run_sparsity_experiment(size=512, device='cpu', lam=0.02):
    print(f"\n[sparsity] size={size} device={device.type}")
    images = load_real_images(size=size)
    results = {}   # {image_name: {method_name: (ranks, magnitudes)}}
    for img_name, img_np in images.items():
        print(f"  image: {img_name}")
        results[img_name] = {}
        img_t = torch.from_numpy(img_np).to(device)

        c = bandlet_coeffs(img_t, lam=lam, device=device)
        results[img_name]['Bandlet'] = sorted_magnitude_curve(c)
        print(f"    Bandlet: {len(c)} coefficients")

        for label, wname in WAVELETS.items():
            try:
                c = classical_wavelet_coeffs(img_np, wname)
                results[img_name][label] = sorted_magnitude_curve(c)
                print(f"    {label} ({wname}): {len(c)} coefficients")
            except ImportError:
                print("    PyWavelets not installed -- skipping classical wavelets. "
                      "Install with: pip install PyWavelets")
                return results
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='output .npz path, e.g. results_A100.npz')
    p.add_argument('--sizes', default='64,128,256,512,1024,2048')
    p.add_argument('--sparsity-size', type=int, default=512)
    p.add_argument('--lam', type=float, default=0.02)
    p.add_argument('--no-cuda', action='store_true')
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]

    has_cuda = torch.cuda.is_available() and not args.no_cuda
    devices = [torch.device('cpu')] + ([torch.device('cuda')] if has_cuda else [])
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else None

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {has_cuda}" + (f" ({gpu_name})" if gpu_name else ""))
    print(f"Sizes: {sizes}")

    timing = run_timing_experiment(sizes, devices, lam=args.lam)

    sparsity_device = devices[-1]  # GPU if available, else CPU
    sparsity = run_sparsity_experiment(size=args.sparsity_size, device=sparsity_device, lam=args.lam)

    # ---- flatten everything into a single .npz -------------------------------
    save_dict = {
        'sizes': np.array(sizes),
        'devices': np.array([d.type for d in devices]),
        'gpu_name': np.array(gpu_name if gpu_name else 'none'),
        'torch_version': np.array(torch.__version__),
        'lam': np.array(args.lam),
    }
    for dev_name, per_size in timing.items():
        for size, row in per_size.items():
            for key, val in row.items():
                save_dict[f'timing__{dev_name}__{size}__{key}'] = np.array(val)

    for img_name, per_method in sparsity.items():
        for method_name, (ranks, mags) in per_method.items():
            safe_method = method_name.replace(' ', '_').replace('/', '-')
            save_dict[f'sparsity__{img_name}__{safe_method}__ranks'] = ranks
            save_dict[f'sparsity__{img_name}__{safe_method}__mags'] = mags

    np.savez_compressed(args.out, **save_dict)
    print(f"\nSaved all results to {args.out}")
    print(f"File size: {__import__('os').path.getsize(args.out) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
