#!/usr/bin/env python3
"""
run_rq3.py
RQ3: Computational cost -- Stacking training time across (sample_size, num_vars).

Single-node:
    python run_rq3.py                          # full grid
    python run_rq3.py --quick                  # reduced grid

Distributed (multi-node SLURM):
    python run_rq3.py --chunk 0 --n-chunks 8   # run configs 0..5
    python run_rq3.py --chunk 1 --n-chunks 8   # run configs 6..11
    python run_rq3.py --merge                  # combine chunk outputs
"""

import time
import argparse
import os
import json
import warnings
import multiprocessing as mp

import pandas as pd
import numpy as np

from stacking import StackingEnsemble

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'training_rescueRobot_25600_64.csv')
TIMEOUT = 200

FULL_CONFIGS = [(s, v)
    for s in [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    for v in [2, 4, 8, 16, 32, 64]]

QUICK_CONFIGS = [(s, v)
    for s in [200, 800, 3200]
    for v in [2, 8, 32]]


def _fit_stacking(sample, vars_int, data, n_folds, seed, max_vars, result_queue):
    try:
        feature_cols = [f'x{i}' for i in range(1, vars_int + 1)]
        X_raw = data.loc[:sample - 1, feature_cols]
        y = data.loc[:sample - 1, 'hazard']
        start = time.time()
        stacking = StackingEnsemble(n_folds=n_folds, random_state=seed, max_vars=max_vars)
        stacking.fit(X_raw, y)
        result_queue.put(('success', time.time() - start))
    except Exception as e:
        result_queue.put(('error', str(e)))


def run_stacking_cost(sample, vars_int, data, seed):
    n_folds = min(10, max(2, sample // 20))
    max_vars = min(vars_int, 15)
    result_queue = mp.Queue()
    process = mp.Process(
        target=_fit_stacking,
        args=(sample, vars_int, data, n_folds, seed, max_vars, result_queue))
    process.start()
    process.join(timeout=TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return TIMEOUT
    try:
        status, value = result_queue.get_nowait()
        return value if status == 'success' else TIMEOUT
    except Exception:
        return TIMEOUT


def main():
    parser = argparse.ArgumentParser(description='RQ3: Computational cost')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk', type=int, default=None)
    parser.add_argument('--n-chunks', type=int, default=8)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'logs')
    cache = os.path.join(base_dir, 'cache')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache, exist_ok=True)

    if args.merge:
        # Merge all chunk outputs
        results = []
        for fname in sorted(os.listdir(cache)):
            if fname.startswith('rq3_chunk') and fname.endswith('.json'):
                with open(os.path.join(cache, fname)) as f:
                    results.extend(json.load(f))
        out = os.path.join(log_dir, 'rq3_stacking.tsv')
        with open(out, 'w') as f:
            f.write("method vars sample time\n")
            for r in results:
                f.write(f"Stacking {r['vars']} {r['sample']} {r['time']:.4f}\n")
        print(f"Merged {len(results)} results -> {out}")
        return

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS

    # Chunk selection
    if args.chunk is not None:
        total = len(configs)
        chunk_size = (total + args.n_chunks - 1) // args.n_chunks
        start = args.chunk * chunk_size
        end = min(start + chunk_size, total)
        configs = configs[start:end]
        print(f"=== RQ3 chunk {args.chunk}/{args.n_chunks}: configs {start}-{end-1} ===")

    print(f"Configs: {len(configs)}, Timeout: {TIMEOUT}s")
    print(f"Loading data...")
    data = pd.read_csv(DATA_FILE)
    print(f"Shape: {data.shape}")

    results = []
    for idx, (sample, vars_int) in enumerate(configs):
        print(f"  [{idx+1}/{len(configs)}] s={sample} v={vars_int}", end=" ", flush=True)
        elapsed = run_stacking_cost(sample, vars_int, data, args.seed)
        results.append({'method': 'Stacking', 'vars': vars_int,
                        'sample': sample, 'time': round(elapsed, 4)})
        tag = " TIMEOUT" if elapsed >= TIMEOUT else ""
        print(f"-> {elapsed:.2f}s{tag}")

    # Save
    if args.chunk is not None:
        out = os.path.join(cache, f'rq3_chunk{args.chunk}.json')
        with open(out, 'w') as f:
            json.dump(results, f)
        print(f"Saved: {out}")
    else:
        out = os.path.join(log_dir, 'rq3_stacking.tsv')
        with open(out, 'w') as f:
            f.write("method vars sample time\n")
            for r in results:
                f.write(f"Stacking {r['vars']} {r['sample']} {r['time']:.4f}\n")
        print(f"Saved: {out}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
