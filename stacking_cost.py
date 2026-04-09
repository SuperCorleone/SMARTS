#!/usr/bin/env python3
"""
stacking_cost.py
RQ3: Computational Cost — Stacking vs BMA (MCMC/BAS).

Mirrors bma-package/bma_cost.py experiment design:
  - Measure Stacking training time across (sample_size, num_vars) configurations
  - Same grid as baseline: samples in {200..25600}, vars in {2..64}
  - Save results in same format as baseline_rq3_results.tsv

Baseline data (MCMC + BAS) is already in logs/baseline_rq3_results.tsv.

Usage:
    python stacking_cost.py                  # full grid (slow)
    python stacking_cost.py --quick          # reduced grid (fast test)
"""

import time
import pandas as pd
import numpy as np
import warnings
import os
import argparse
from datetime import datetime
import multiprocessing as mp
from functools import partial

from stacking import StackingEnsemble

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'training_rescueRobot_25600_64.csv')
TIMEOUT = 200  # seconds, same as baseline

# Full grid (same as baseline cost_input.csv)
FULL_CONFIGS = [
    (s, v)
    for s in [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    for v in [2, 4, 8, 16, 32, 64]
]

# Quick grid for testing
QUICK_CONFIGS = [
    (s, v)
    for s in [200, 800, 3200]
    for v in [2, 8, 32]
]


def _fit_stacking(sample, vars_int, data, n_folds, random_state, max_vars, result_queue):
    """Target function for multiprocessing: fit Stacking and put result into queue."""
    try:
        feature_cols = [f'x{i}' for i in range(1, vars_int + 1)]
        X_raw = data.loc[:sample - 1, feature_cols]
        y = data.loc[:sample - 1, 'hazard']

        start = time.time()
        stacking = StackingEnsemble(n_folds=n_folds, random_state=random_state, max_vars=max_vars)
        stacking.fit(X_raw, y)
        elapsed = time.time() - start
        result_queue.put(('success', elapsed))
    except Exception as e:
        result_queue.put(('error', str(e)))


def run_stacking_cost(sample, vars_int, data):
    """Train Stacking with a hard timeout. Returns elapsed time or TIMEOUT."""
    # Adjust n_folds for small datasets
    n_folds = min(10, max(2, sample // 20))
    max_vars = min(vars_int, 15)

    # Create a queue to receive the result from the child process
    result_queue = mp.Queue()
    process = mp.Process(
        target=_fit_stacking,
        args=(sample, vars_int, data, n_folds, 42, max_vars, result_queue)
    )
    process.start()
    process.join(timeout=TIMEOUT)

    if process.is_alive():
        # Timeout occurred: terminate the child process
        process.terminate()
        process.join()
        print(" (TIMEOUT)", end="", flush=True)
        return TIMEOUT
    else:
        # Process finished normally
        try:
            status, value = result_queue.get_nowait()
            if status == 'success':
                return value
            else:
                print(f" (Error: {value})", end="", flush=True)
                return TIMEOUT
        except Exception:
            return TIMEOUT


def main():
    parser = argparse.ArgumentParser(description='RQ3: Computational Cost')
    parser.add_argument('--quick', action='store_true', help='Run reduced config grid for testing')
    args = parser.parse_args()

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS

    start_time = datetime.now()
    print(f"=== RQ3: Computational Cost (Stacking) ===")
    print(f"    Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Configs: {len(configs)} {'(quick mode)' if args.quick else '(full grid)'}")
    print(f"    Timeout: {TIMEOUT} seconds per config")

    print(f"\nLoading data from {DATA_FILE}...")
    data = pd.read_csv(DATA_FILE)
    print(f"Data shape: {data.shape}")

    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, 'stacking_rq3_results.tsv')

    results = []
    total = len(configs)

    with open(output_file, 'w') as f:
        f.write("method vars sample time\n")

        for idx, (sample, vars_int) in enumerate(configs):
            print(f"  [{idx+1}/{total}] sample={sample}, vars={vars_int}", end=" ", flush=True)

            elapsed = run_stacking_cost(sample, vars_int, data)
            f.write(f"Stacking {vars_int} {sample} {elapsed:.4f}\n")
            f.flush()

            results.append({'method': 'Stacking', 'vars': vars_int, 'sample': sample, 'time': elapsed})
            to_str = " (TIMEOUT)" if elapsed >= TIMEOUT else ""
            print(f"→ {elapsed:.2f}s{to_str}")

    end_time = datetime.now()
    print(f"\n=== Complete ===")
    print(f"  Duration: {end_time - start_time}")
    print(f"  Results saved: {output_file}")

    # Summary
    df = pd.DataFrame(results)
    print(f"\n  Mean time: {df['time'].mean():.2f}s")
    print(f"  Max time: {df['time'].max():.2f}s")
    print(f"  Timeouts (>={TIMEOUT}s): {(df['time'] >= TIMEOUT).sum()}")


if __name__ == '__main__':
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()