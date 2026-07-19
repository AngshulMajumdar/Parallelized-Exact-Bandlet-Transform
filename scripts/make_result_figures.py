"""
make_result_figures.py
=======================
Reproduces the paper's two data-driven figures (and the tables that go with
them) from the raw .npz files produced by src/bandlet_benchmark.py.

Usage:
    python scripts/make_result_figures.py \
        --results results/results_T4.npz results/results_L4.npz \
                  results/results_A100.npz results/results_Blackwell.npz \
        --out paper/figs

Expects each .npz to have been produced by bandlet_benchmark.py (same run
format for all files -- same --sizes, same --sparsity-size). Labels for the
timing plot are taken from each file's own gpu_name field.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.dpi': 150,
})

# Fixed colors so repeated runs are visually comparable; falls back to the
# matplotlib default cycle for any additional files beyond these four.
DEFAULT_COLORS = ['#4c78a8', '#54a24b', '#b279a2', '#e45756', '#eeca3b', '#72b7b2']

METHOD_LABELS = {
    'Bandlet': 'Bandlet',
    'Haar': 'Haar',
    'Daubechies-4': 'Daubechies-4 (db2)',
    'Daubechies-6': 'Daubechies-6 (db3)',
    'Daubechies-8': 'Daubechies-8 (db4)',
    'Biorthogonal_9-7_(Mallat-JPEG2000)': 'Biorthogonal 9/7 (Mallat/JPEG2000)',
    'Coiflet-2': 'Coiflet-2',
}
METHOD_COLORS = {
    'Bandlet': '#e45756',
    'Haar': '#9d9d9d',
    'Daubechies-4': '#72b7b2',
    'Daubechies-6': '#4c78a8',
    'Daubechies-8': '#54a24b',
    'Biorthogonal_9-7_(Mallat-JPEG2000)': '#b279a2',
    'Coiflet-2': '#eeca3b',
}


def make_timing_figure(npz_paths, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    ax1, ax2 = axes

    print("\n=== Timing table (Table: same-machine GPU speedup) ===")
    for i, path in enumerate(npz_paths):
        d = np.load(path, allow_pickle=True)
        label = str(d['gpu_name'])
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

        sizes = sorted(set(int(k.split('__')[2]) for k in d.keys() if k.startswith('timing__')))
        cpu_times = np.array([float(d[f'timing__cpu__{s}__total_ms']) for s in sizes])
        gpu_times = np.array([float(d[f'timing__cuda__{s}__total_ms']) for s in sizes])

        ax1.plot(sizes, gpu_times, 'o-', color=color, linewidth=2.2, markersize=5.5,
                  label=f'{label} — GPU')
        ax1.plot(sizes, cpu_times, 's--', color=color, linewidth=1.2, markersize=4,
                  alpha=0.5, label=f'{label} — CPU (same machine)')

        speedup = cpu_times / gpu_times
        ax2.plot(sizes, speedup, 'o-', color=color, linewidth=2.2, markersize=5.5, label=label)

        print(f"\n{label}:")
        for s, sp, c, g in zip(sizes, speedup, cpu_times, gpu_times):
            print(f"  N={s:5d}  cpu={c:9.2f}ms  gpu={g:8.2f}ms  speedup={sp:6.2f}x")

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Image size ($N \\times N$)')
    ax1.set_ylabel('Total pipeline time (ms, encode + decode)')
    ax1.set_title('(a) Absolute time vs. size')
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.legend(fontsize=7.3, loc='upper left', ncol=1, framealpha=0.9)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)
    ax2.axhline(1.0, color='gray', linewidth=1, linestyle=':')
    ax2.set_xlabel('Image size ($N \\times N$)')
    ax2.set_ylabel('Speedup (CPU time / GPU time), same machine')
    ax2.set_title('(b) GPU speedup vs. its own CPU, by size')
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)

    fig.suptitle('Bandlet transform: CPU vs. GPU scaling with image size', fontsize=13, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'size_vs_time_cpu_gpu.png')
    out_pdf = os.path.join(out_dir, 'size_vs_time_cpu_gpu.pdf')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"\nSaved {out_png} and {out_pdf}")


def make_sparsity_figure(npz_path, out_dir):
    d = np.load(npz_path, allow_pickle=True)
    images = sorted(set(k.split('__')[1] for k in d.keys() if k.startswith('sparsity__')))
    methods = sorted(set(k.split('__')[2] for k in d.keys() if k.startswith('sparsity__')))
    if not methods:
        print(f"No sparsity data found in {npz_path} -- skipping sparsity figure "
              "(was PyWavelets installed when this run was made?)")
        return

    ranks = d[f'sparsity__{images[0]}__{methods[0]}__ranks']
    mask = (ranks >= 10) & (ranks <= 10000)
    x = np.log(ranks[mask].astype(float))
    A = np.vstack([x, np.ones_like(x)]).T

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.8))
    print("\n=== Sparsity decay exponents (Table: fitted decay rate r) ===")
    rows = []
    for method in methods:
        mags_stack = np.stack([d[f'sparsity__{img}__{method}__mags'] for img in images], axis=0)
        mean_mag = mags_stack.mean(axis=0)

        y = np.log(np.clip(mean_mag[mask], 1e-12, None))
        r, _c = np.linalg.lstsq(A, y, rcond=None)[0]
        r = -r
        rows.append((method, r))

        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        lw = 2.8 if method == 'Bandlet' else 1.5
        z = 10 if method == 'Bandlet' else 5
        ax.plot(ranks, mean_mag, color=color, linewidth=lw, label=f'{label}  (r={r:.3f})', zorder=z)

    for method, r in sorted(rows, key=lambda t: -t[1]):
        print(f"  {METHOD_LABELS.get(method, method):40s} r = {r:.3f}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Coefficient rank (sorted by |magnitude|, descending)')
    ax.set_ylabel(f'|coefficient magnitude| (averaged across {", ".join(images)})')
    ax.set_title('Coefficient-magnitude decay: bandlet vs. classical wavelets\n'
                  '$|c|_{(n)} \\sim n^{-r}$ fit over rank 10-10,000; steeper decay = sparser')
    ax.legend(fontsize=9.5, loc='lower left', framealpha=0.9, title='method (fitted decay rate r)')

    plt.tight_layout()
    out_png = os.path.join(out_dir, 'sparsity_decay.png')
    out_pdf = os.path.join(out_dir, 'sparsity_decay.pdf')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"\nSaved {out_png} and {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', nargs='+', required=True,
                    help='One or more .npz files from bandlet_benchmark.py '
                         '(one per GPU/machine). The first file with sparsity '
                         'data is used for the sparsity figure -- sparsity '
                         'results are deterministic and were confirmed '
                         'bit-identical across GPUs in the paper\'s own runs.')
    p.add_argument('--out', default='paper/figs', help='output directory for figures')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    make_timing_figure(args.results, args.out)

    sparsity_source = None
    for path in args.results:
        d = np.load(path, allow_pickle=True)
        if any(k.startswith('sparsity__') for k in d.keys()):
            sparsity_source = path
            break
    if sparsity_source:
        make_sparsity_figure(sparsity_source, args.out)
    else:
        print("\nNo sparsity data found in any provided file -- skipping sparsity figure.")


if __name__ == '__main__':
    main()
