"""
run_offline_selftest.py
========================
Runs the core correctness checks with NO GPU (or even real torch) required,
using the NumPy-backed shim of the torch API in tests/torch_shim/. This is
exactly the methodology used throughout development: every fix was
verified against this shim before being sent out for a real GPU run, since
the development environment had no GPU access at all. It is not a
substitute for testing on real hardware (see notebooks/ for that), but it
catches logic errors immediately and needs nothing installed beyond numpy
and scikit-image.

Usage:
    python tests/run_offline_selftest.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch_shim'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np  # noqa: E402
import bandlet_benchmark as bb  # noqa: E402

torch = bb.torch

PASS = []
FAIL = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    (PASS if condition else FAIL).append(name)
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    print("=== 1. Lifting DWT: perfect reconstruction ===")
    torch.manual_seed(0)
    x = torch.randn(8, 64, 64)
    LL, LH, HL, HH = bb.dwt2d_level(x)
    rec = bb.idwt2d_level(LL, LH, HL, HH)
    err = (x - rec).abs().max().item()
    check("DWT round trip", err < 1e-10, f"max err {err:.2e}")

    print("\n=== 2. Shear/unshear: exact permutation ===")
    torch.manual_seed(1)
    blocks = torch.randn(20, 16, 16)
    ks = torch.tensor([float(k) for k in [-3, -2, -1, 0, 1, 2, 3] * 3][:20])
    sheared = bb.shear_batch(blocks, ks)
    rec = bb.unshear_batch(sheared, ks)
    err = (blocks - rec).abs().max().item()
    check("shear round trip", err == 0.0, f"max err {err:.2e}")

    print("\n=== 3. Leaf transform (shear+lift+skip): exact inverse ===")
    torch.manual_seed(2)
    blocks = torch.randn(30, 8, 8)
    cost, k, s, d = bb.best_leaf_batched(blocks)
    rec = bb.leaf_inverse_batched(s, d, k, 8)
    err = (blocks - rec).abs().max().item()
    check("leaf transform round trip", err < 1e-10, f"max err {err:.2e}")

    print("\n=== 4. Quadtree segmentation: exact reconstruction across lam extremes ===")
    for lam, desc in [(0.0, "lam=0 (maximal splitting)"),
                       (0.5, "lam=0.5 (typical)"),
                       (1000.0, "lam=1000 (full merge)")]:
        torch.manual_seed(3)
        N = 64
        sub = torch.randn(N, N)
        levels = bb.segment_subband(sub, lam=lam)
        leaves = bb.collect_leaves(levels)
        rec = torch.zeros(N, N)
        for (r0, c0, bsz, kk, ss, dd) in leaves:
            kt = torch.tensor([kk])
            block = bb.leaf_inverse_batched(ss.unsqueeze(0), dd.unsqueeze(0), kt, bsz).squeeze(0)
            rec[r0:r0 + bsz, c0:c0 + bsz] = block
        err = (sub - rec).abs().max().item()
        check(f"quadtree round trip, {desc}", err < 1e-9, f"max err {err:.2e}, {len(leaves)} leaves")

    print("\n=== 5. Full pipeline on a real image ===")
    img_np = bb.load_real_images(size=64)['camera']
    img = torch.from_numpy(img_np)
    LL, trees = bb.bandlet_forward(img, lam=0.02)
    rec = bb.bandlet_inverse(LL, trees)
    err = (img - rec).abs().max().item()
    check("full pipeline round trip (real image)", err < 1e-5, f"max err {err:.2e}")

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == '__main__':
    main()
