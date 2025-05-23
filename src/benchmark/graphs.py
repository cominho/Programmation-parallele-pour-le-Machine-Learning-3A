# src/benchmark/plot.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_speedups(csv_path, output=None):
    """
    Plot speed-ups (Numba & Cython vs NumPy) with error bars.
    """
    df = pd.read_csv(csv_path)
    dims = df['dim'].values

    su_nb = df['speedup_numba'].values
    err_nb = df['sd_numba'].values / df['mean_numba'].values * su_nb
    su_cy = df['speedup_cython'].values
    err_cy = df['sd_cython'].values / df['mean_cython'].values * su_cy

    fig, ax = plt.subplots()
    ax.errorbar(dims, su_nb, yerr=err_nb, fmt='-o', label='Numba')
    ax.errorbar(dims, su_cy, yerr=err_cy, fmt='-x', label='Cython')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Speed-up vs NumPy')
    ax.set_title('Speed-up with error bars')
    ax.legend()
    ax.grid(True)

    if output:
        fig.savefig(output, bbox_inches='tight')
    plt.show()


def plot_times(csv_path, output=None):
    """
    Plot mean execution times with error bars, log-log scale.
    """
    df = pd.read_csv(csv_path)
    dims = df['dim'].values
    methods = ['numpy', 'numba', 'cython']

    fig, ax = plt.subplots()
    for m in methods:
        means = df[f'mean_{m}'].values
        errs  = df[f'sd_{m}'].values
        ax.errorbar(dims, means, yerr=errs, fmt='-o', label=m.capitalize())

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Execution time (s)')
    ax.set_title('Mean execution time (log-log)')
    ax.legend()
    ax.grid(True)

    if output:
        fig.savefig(output, bbox_inches='tight')
    plt.show()


def plot_config_heatmap(csv_path, output=None):
    """
    Heatmap of Cython speed-up vs (threads, block_size).
    """
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index='block_size', columns='threads', values='speedup_cython')
    matrix = pivot.values
    threads = pivot.columns.values
    blocks  = pivot.index.values

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect='auto', origin='lower')
    ax.set_xticks(np.arange(len(threads)))
    ax.set_xticklabels(threads)
    ax.set_yticks(np.arange(len(blocks)))
    ax.set_yticklabels(blocks)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Block size')
    ax.set_title('Cython speed-up heatmap')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Speed-up')

    if output:
        fig.savefig(output, bbox_inches='tight')
    plt.show()


def plot_multi_impl_speedups(csv_map, impl='cython', output=None):
    """
    Plot speed-ups for multiple methods on the same axes for one implementation.
    impl: 'cython' or 'numba'
    csv_map: dict mapping method name to CSV file path.
    """
    plt.figure()
    for name, path in csv_map.items():
        df = pd.read_csv(path)
        dims = df['dim'].values
        plt.plot(
            dims,
            df[f'speedup_{impl}'],
            marker='o',
            label=f"{name.capitalize()} {impl.capitalize()}"
        )

    plt.xscale('log', base=2)
    plt.xlabel('Sequence length')
    plt.ylabel(f'{impl.capitalize()} Speed-up vs NumPy')
    plt.title(f'{impl.capitalize()} Speed-ups by method')
    plt.legend()
    plt.grid(True)

    if output:
        plt.savefig(output, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import sys
    # CLI: first arg can be impl ('cython' or 'numba'), rest NAME=PATH
    if len(sys.argv) > 2 and '=' in sys.argv[2]:
        impl = sys.argv[1]
        csv_map = {}
        for arg in sys.argv[2:]:
            if '=' in arg:
                name, path = arg.split('=', 1)
                csv_map[name] = path
        plot_multi_impl_speedups(csv_map, impl=impl)
    else:
        plot_speedups(sys.argv[1])
