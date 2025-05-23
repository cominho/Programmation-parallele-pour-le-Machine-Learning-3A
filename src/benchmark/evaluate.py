import os
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from statistics import mean, stdev
from att_bench_lib.numpy_att import compute_attention_numpy
from att_bench_lib.numba_att import compute_attention_numba
from att_bench_lib.cython_att import compute_attention_cython
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from src.utils.resources import configure_resources
import random

# Timing harness
def measure(fn, args, warmup=3, repeat=5):
    """Warm up, then measure `repeat` runs of fn(*args)."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return times

def evaluate_configuration(Q, K, V, cfg):
    """Run compute_attention_cython with cfg=(threads, block_size, dtype)."""
    nt, bs, dt = cfg
    try:
        Qd, Kd, Vd = Q.astype(dt), K.astype(dt), V.astype(dt)
        ts = measure(
            compute_attention_cython,
            (Qd, Kd, Vd, nt, bs),
            warmup=2, repeat=5
        )
        return mean(ts)
    except Exception as e:
        print(f"[WARNING] config {cfg} failed: {e}")
        return float('inf')

# Bandit (UCB)
class Bandit:
    def __init__(self, arms):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}

    def select_arm(self):
        total = sum(self.counts.values())
        # ensure each arm is tried once
        for arm, cnt in self.counts.items():
            if cnt == 0:
                return arm
        # UCB
        ucb = {
            arm: self.values[arm]
                 + np.sqrt(2 * np.log(total) / self.counts[arm])
            for arm in self.arms
        }
        return max(ucb, key=ucb.get)

    def update(self, arm, reward):
        cnt = self.counts[arm] + 1
        self.values[arm] = ((self.values[arm] * (cnt - 1)) + reward) / cnt
        self.counts[arm] = cnt

# Bayesian Optimization (Gaussian Process + EI)
class BayesianOptimizer:
    def __init__(self, grid):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        import scipy.stats as stats

        self.grid = grid
        # encode dtype as integer
        self.dt_map = {np.float32:0, np.float64:1}
        self.inv_map = {v:k for k,v in self.dt_map.items()}
        self.X, self.y = [], []
        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0),
            alpha=1e-6, normalize_y=True
        )
        self.stats = stats

    def suggest(self):
        import numpy as np
        # initial: sample each once
        if len(self.X) < len(self.grid):
            return self.grid[len(self.X)]
        # fit gp
        X_arr = np.array(self.X)
        y_arr = np.array(self.y)
        self.gp.fit(X_arr, y_arr)
        # acquisition: EI
        def EI(x, xi=0.01):
            mu, sigma = self.gp.predict(x.reshape(1,-1), return_std=True)
            best = np.min(self.y)
            imp = best - mu - xi
            Z = imp/(sigma + 1e-9)
            return imp * self.stats.norm.cdf(Z) + sigma * self.stats.norm.pdf(Z)
        eis = [EI(np.array([nt, bs, self.dt_map[dt]])) for nt, bs, dt in self.grid]
        idx = int(np.argmax(eis))
        return self.grid[idx]

    def update(self, cfg, reward):
        nt, bs, dt = cfg
        self.X.append([nt, bs, self.dt_map[dt]])
        self.y.append(reward)

# Random search
def random_search(Q, K, V, threads, blocks, dtypes, n_trials=30):
    best_cfg, best_t = None, float('inf')
    for _ in range(n_trials):
        cfg = (
            random.choice(threads),
            random.choice(blocks),
            random.choice(dtypes)
        )
        t = evaluate_configuration(Q, K, V, cfg)
        if t < best_t:
            best_t, best_cfg = t, cfg
    return best_cfg


# Successive halving (Hyperband building block)
def successive_halving(Q, K, V, threads, blocks, dtypes, max_iters=30, eta=3):
    """
    Successive Halving: eliminate the worst 1/eta of configs at each rung.
    Guarantees at least one config per rung and never drops all survivors.
    """
    grid = [(nt, bs, dt) for nt in threads for bs in blocks for dt in dtypes]
    n = len(grid)
    # start with the full set
    survivors = list(grid)

    # number of rungs
    s_max = int(np.floor(np.log(n) / np.log(eta)))
    B     = max_iters

    for s in reversed(range(s_max + 1)):
        # at rung s we evaluate n_i configs for r_i repetitions
        n_i = max(1, int(np.ceil(n * (eta ** -s))))
        r_i = max(1, int(B * (eta ** -s)))

        # if somehow survivors is empty, reset to full grid
        if not survivors:
            survivors = list(grid)

        # evaluate only up to n_i of the survivors
        scores = []
        for cfg in survivors[:n_i]:
            t_list = measure(
                compute_attention_cython,
                (Q.astype(cfg[2]), K.astype(cfg[2]), V.astype(cfg[2]), cfg[0], cfg[1]),
                warmup=1,
                repeat=r_i
            )
            scores.append((cfg, mean(t_list)))

        # sort by execution time (ascending)
        scores.sort(key=lambda x: x[1])

        # keep only the top 1/eta of them (but at least 1)
        k = max(1, len(scores) // eta)
        survivors = [cfg for cfg, _ in scores[:k]]

    # after all rungs, survivors[0] is the best
    return survivors[0]

# CMA-ES Search
def cma_es_search(Q, K, V, threads, blocks, dtypes, budget=30):
    from cma import CMAEvolutionStrategy
    # maps for dtype
    dt_map = {dt:i for i,dt in enumerate(dtypes)}
    inv_map = {i:dt for dt,i in dt_map.items()}

    def decode(x):
        nt = int(np.clip(round(x[0]), min(threads), max(threads)))
        bs = int(np.clip(round(x[1]), min(blocks), max(blocks)))
        di = int(np.clip(round(x[2]), 0, len(dtypes)-1))
        return nt, bs, inv_map[di]

    def obj(x):
        cfg = decode(x)
        return evaluate_configuration(Q, K, V, cfg)

    x0 = [threads[0], blocks[0], dt_map[dtypes[0]]]
    sigma0 = 1.0
    es = CMAEvolutionStrategy(x0, sigma0, {'maxfevals':budget})
    while not es.stop():
        X = es.ask()
        es.tell(X, [obj(x) for x in X])
    return decode(es.result.xbest)

# DOE + Response surface methodology
def doe_rsm_search(Q, K, V, threads, blocks, dtypes, n_samples=20):
    """
    DOE + Response Surface Modeling search.

    - Latin Hypercube sampling in [0,1]^3 with n_samples points.
    - Fit a quadratic regression (RSM).
    - Evaluate the surrogate on a fine 21^3 grid.
    - Map the best fractional coordinate back to discrete thread/block/dtype.
    """
    from pyDOE2 import lhs
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Latin Hypercube sampling
    L = lhs(3, samples=n_samples)
    X_data = []
    y_data = []
    for l in L:
        cfg = (
            threads[int(l[0] * (len(threads) - 1))],
            blocks[int(l[1] * (len(blocks)  - 1))],
            dtypes[int(round(l[2] * (len(dtypes) - 1)))]
        )
        X_data.append([l[0], l[1], l[2]])
        y_data.append(evaluate_configuration(Q, K, V, cfg))

    X_arr = np.array(X_data)
    y_arr = np.array(y_data)

    # Fit quadratic model (linear + squared terms)
    poly = np.hstack([X_arr, X_arr**2])
    model = LinearRegression().fit(poly, y_arr)

    # Generate a fine grid of fractional points [0,1]^3
    grid_pts = []
    for i in np.linspace(0, 1, 21):
        for j in np.linspace(0, 1, 21):
            for k in np.linspace(0, 1, 21):
                grid_pts.append([i, j, k])
    grid = np.array(grid_pts)

    # Predict and pick the best
    preds = model.predict(np.hstack([grid, grid**2]))
    best_frac = grid[int(np.argmin(preds))]
    
    # Map fractional back to valid indices
    i0 = int(np.clip(best_frac[0] * (len(threads) - 1), 0, len(threads) - 1))
    i1 = int(np.clip(best_frac[1] * (len(blocks ) - 1), 0, len(blocks)  - 1))
    i2 = int(np.clip(round(best_frac[2] * (len(dtypes) - 1)), 0, len(dtypes) - 1))

    return threads[i0], blocks[i1], dtypes[i2]


# Adaptive search dispatcher

def adaptive_search(Q, K, V, threads, blocks, dtypes, method='bandit', max_iters=30):
    if method == 'bandit':
        algo = Bandit([(nt,bs,dt) for nt in threads for bs in blocks for dt in dtypes])
        best_cfg, best_t = None, float('inf')
        for _ in range(max_iters):
            cfg = algo.select_arm()
            t = evaluate_configuration(Q, K, V, cfg)
            algo.update(cfg, -t)
            if t < best_t:
                best_t, best_cfg = t, cfg
        return best_cfg

    elif method == 'bayes':
        grid = [(nt,bs,dt) for nt in threads for bs in blocks for dt in dtypes]
        algo = BayesianOptimizer(grid)
        best_cfg, best_t = None, float('inf')
        for _ in range(max_iters):
            cfg = algo.suggest()
            t = evaluate_configuration(Q, K, V, cfg)
            algo.update(cfg, t)
            if t < best_t:
                best_t, best_cfg = t, cfg
        return best_cfg

    elif method == 'random':
        return random_search(Q, K, V, threads, blocks, dtypes, n_trials=max_iters)

    elif method == 'hyperband':
        return successive_halving(Q, K, V, threads, blocks, dtypes, max_iters=max_iters)

    elif method == 'cmaes':
        return cma_es_search(Q, K, V, threads, blocks, dtypes, budget=max_iters)

    elif method == 'doe':
        return doe_rsm_search(Q, K, V, threads, blocks, dtypes, n_samples=max_iters)

    else:
        raise ValueError(f"Unknown method '{method}'")

# Main benchmark loop

def run_benchmark(
    dims=None, threads=None, blocks=None, dtypes=None,
    method='bandit', max_iters=30, out_dir='results'
):
    dims    = dims    or [64,128,256,512]
    threads = threads or [1,2,4,8]
    blocks  = blocks  or [8,16,32,64]
    dtypes  = dtypes  or [np.float32, np.float64]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for dim in dims:
        print(f"\n[INFO] Running dim={dim}, method={method}")
        Q = np.random.randn(dim,dim).astype(np.float32)
        K = np.random.randn(dim,dim).astype(np.float32)
        V = np.random.randn(dim,dim).astype(np.float32)

        best_nt, best_bs, best_dt = adaptive_search(
            Q, K, V, threads, blocks, dtypes, method=method, max_iters=max_iters
        )

        Qd, Kd, Vd = Q.astype(best_dt), K.astype(best_dt), V.astype(best_dt)
        t_np = measure(compute_attention_numpy,  (Qd,Kd,Vd), warmup=1, repeat=10)
        t_nb = measure(compute_attention_numba,  (Qd,Kd,Vd), warmup=2, repeat=10)
        t_cy = measure(compute_attention_cython,(Qd,Kd,Vd,best_nt,best_bs), warmup=2, repeat=10)

        rec = {
            'dim': dim,
            'dtype': best_dt.__name__,
            'threads': best_nt,
            'block_size': best_bs,
            'mean_numpy':  mean(t_np),  'sd_numpy':  stdev(t_np),
            'mean_numba':  mean(t_nb),  'sd_numba':  stdev(t_nb),
            'mean_cython': mean(t_cy), 'sd_cython': stdev(t_cy),
            'speedup_numba':  mean(t_np)/mean(t_nb),
            'speedup_cython': mean(t_np)/mean(t_cy)
        }
        records.append(rec)

    df = pd.DataFrame(records)
    path = out_dir / f"{method}.csv"
    df.to_csv(path, index=False)
    print(f"\n[INFO] Saved summary to {path}")


def main():
    configure_resources()
    parser = argparse.ArgumentParser(description="Adaptive self-attention benchmark")
    parser.add_argument('--method',
                        choices=['bandit','bayes','random','hyperband','cmaes','doe'],
                        default='bandit')
    parser.add_argument('--calls',  type=int, default=20, dest='max_iters')
    parser.add_argument('--dims',   nargs='+', type=int, default=[64,128,256])
    parser.add_argument('--out',    type=str, default='output', dest='out_dir')
    args = parser.parse_args()

    run_benchmark(
        dims=args.dims,
        method=args.method,
        max_iters=args.max_iters,
        out_dir=args.out
    )

if __name__ == '__main__':
    main()
