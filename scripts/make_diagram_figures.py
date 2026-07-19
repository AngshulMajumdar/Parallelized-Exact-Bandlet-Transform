"""
make_diagram_figures.py
========================
Reproduces the paper's two illustrative figures. Both are generated from
real computation via src/bandlet_benchmark.py -- not decorative diagrams --
so this doubles as another correctness check: the shear round-trip error
and the quadtree partition it prints should both come out at floating-point
machine precision.

Usage:
    python scripts/make_diagram_figures.py --out paper/figs

Requires torch (CPU is fine -- both figures are cheap) and, for the
quadtree partition figure, scikit-image (for the bundled 'camera' test
image; no download needed).
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bandlet_benchmark as bb  # noqa: E402

torch = bb.torch

mpl.rcParams.update({'font.size': 11, 'figure.dpi': 150})


def make_shear_figure(out_dir):
    N = 16
    yy, xx = np.mgrid[0:N, 0:N]
    block_np = ((xx - 2.0 * yy) > 4).astype(np.float32)
    block = torch.tensor(block_np)

    k = 2.0
    sheared = bb.shear_batch(block.unsqueeze(0), torch.tensor([k])).squeeze(0)
    unsheared = bb.unshear_batch(sheared.unsqueeze(0), torch.tensor([k])).squeeze(0)

    err = float(np.abs(block.numpy() - unsheared.numpy()).max())
    print(f"[shear figure] round-trip max |err| = {err:.3e} "
          f"({'OK, exact' if err == 0.0 else 'UNEXPECTED -- should be exactly 0'})")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    titles = ['Original block\n(diagonal edge)', 'After shear ($k=2$)\n(edge now vertical)',
              'After un-shear\n(exact recovery)']
    imgs = [block.numpy(), sheared.numpy(), unsheared.numpy()]
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#999999')

    plt.tight_layout()
    out_pdf = os.path.join(out_dir, 'shear_mechanism.pdf')
    out_png = os.path.join(out_dir, 'shear_mechanism.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved {out_pdf} and {out_png}")


def make_quadtree_figure(out_dir, lam=0.02, size=128):
    img_np = bb.load_real_images(size=size)['camera']
    img = torch.from_numpy(img_np)
    LL, LH, HL, HH = bb.dwt2d_level(img)
    levels = bb.segment_subband(LH, lam=lam)
    leaves = bb.collect_leaves(levels)

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 5.4))
    ax.imshow(LH.numpy(), cmap='gray', vmin=-0.2, vmax=0.2)
    for (r0, c0, bsz, k, s, d) in leaves:
        color = '#888888' if k == bb.SKIP_K else '#39c93f'
        rect = plt.Rectangle((c0, r0), bsz, bsz, edgecolor=color, facecolor='none', linewidth=0.8)
        ax.add_patch(rect)
        if k != bb.SKIP_K:
            cy, cx = r0 + bsz / 2, c0 + bsz / 2
            L = bsz * 0.42
            norm = math.hypot(k, 1.0)
            dx, dy = L * k / norm, L * 1.0 / norm
            ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color='#e0392b', linewidth=1.1)
    ax.set_xticks([]); ax.set_yticks([])
    n_skip = sum(1 for l in leaves if l[3] == bb.SKIP_K)
    sizes = sorted(set(l[2] for l in leaves))
    ax.set_title(f'LH subband quadtree partition, camera ${size}\\times{size}$\n'
                 f'{len(leaves)} leaves ({n_skip} skipped), block sizes {sizes}', fontsize=10.5)
    print(f"[quadtree figure] {len(leaves)} leaves, {n_skip} skipped, block sizes {sizes}")

    plt.tight_layout()
    out_pdf = os.path.join(out_dir, 'quadtree_partition.pdf')
    out_png = os.path.join(out_dir, 'quadtree_partition.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved {out_pdf} and {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='paper/figs')
    p.add_argument('--lam', type=float, default=0.02)
    p.add_argument('--size', type=int, default=128)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    make_shear_figure(args.out)
    make_quadtree_figure(args.out, lam=args.lam, size=args.size)


if __name__ == '__main__':
    main()
