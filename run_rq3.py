#!/usr/bin/env python3
"""
run_rq3.py

Single entry point for RQ3:

  --phase fit
      RQ3-A one-shot fit() cost on the synthetic (sample x vars) grid.

  --phase online
      RQ3-B online-update cost benchmark.
      - grid mode: synthetic samples x vars online-cost sweep
      - legacy mode: old 450-row drift stream benchmark
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
import warnings

import numpy as np
import pandas as pd

from run_rq2 import build_data_splits, apply_drift, DRIFT_SCENARIOS

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'training_rescueRobot_25600_64.csv')
TIMEOUT = 200

FULL_CONFIGS = [(s, v)
    for s in [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    for v in [2, 4, 8, 16, 32, 64]]

QUICK_CONFIGS = [(s, v)
    for s in [200, 800, 3200]
    for v in [2, 8, 32]]

BASE_STREAM_LEN = 450


# ---------------------------------------------------------------------------
# RQ3-A fit benchmark
# ---------------------------------------------------------------------------

def _fit_stacking(sample, vars_int, data, n_folds, seed, max_vars, result_queue):
    try:
        from stacking import StackingEnsemble
        feature_cols = [f'x{i}' for i in range(1, vars_int + 1)]
        X_raw = data.loc[:sample - 1, feature_cols]
        y = data.loc[:sample - 1, 'hazard']
        start = time.time()
        stacking = StackingEnsemble(n_folds=n_folds, random_state=seed, max_vars=max_vars)
        stacking.fit(X_raw, y)
        result_queue.put(('success', time.time() - start))
    except Exception as exc:
        result_queue.put(('error', str(exc)))


def _fit_bma(sample, vars_int, data, n_folds, seed, max_vars, result_queue):
    try:
        from bma import BMA
        feature_cols = [f'x{i}' for i in range(1, vars_int + 1)]
        X_raw = data.loc[:sample - 1, feature_cols]
        y = data.loc[:sample - 1, 'hazard']
        start = time.time()
        bma = BMA(y, X_raw, RegType='Logit', Verbose=False, MaxVars=max_vars)
        bma.fit()
        result_queue.put(('success', time.time() - start))
    except Exception as exc:
        result_queue.put(('error', str(exc)))


def _fit_best_logit(sample, vars_int, data, n_folds, seed, max_vars, result_queue):
    try:
        from run_rq1 import find_best_logit_ms
        feature_cols = [f'x{i}' for i in range(1, vars_int + 1)]
        X_raw = data.loc[:sample - 1, feature_cols]
        y = data.loc[:sample - 1, 'hazard']
        start = time.time()
        find_best_logit_ms(X_raw, y, random_state=seed)
        result_queue.put(('success', time.time() - start))
    except Exception as exc:
        result_queue.put(('error', str(exc)))


def _python_fit_cell(method, sample, vars_int, data, seed):
    cfg = FIT_METHODS[method]
    n_folds = min(10, max(2, sample // 20))
    max_vars = min(vars_int, cfg['max_vars_cap'])
    result_queue = mp.Queue()
    process = mp.Process(
        target=cfg['fit_fn'],
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


def _r_fit_cell(r_method_label, sample, vars_int):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    r_script = os.path.join(base_dir, 'bma_cost.R')
    rscript = shutil.which('Rscript')
    if rscript is None:
        return TIMEOUT, 'Rscript not on PATH'
    cmd = [rscript, r_script, '-s', str(sample), '-v', str(vars_int), '-m', r_method_label]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True, timeout=TIMEOUT + 30)
    except subprocess.TimeoutExpired:
        return TIMEOUT, 'subprocess timeout'
    if proc.returncode != 0:
        return TIMEOUT, (proc.stderr or 'nonzero exit').strip()[:200]
    parts = (proc.stdout or '').strip().split()
    if len(parts) < 4:
        return TIMEOUT, f'unparseable: {proc.stdout!r}'
    try:
        elapsed = float(parts[-1])
    except ValueError:
        return TIMEOUT, f'non-numeric time: {parts[-1]!r}'
    if (time.time() - t0) >= TIMEOUT + 5 and elapsed < TIMEOUT:
        elapsed = TIMEOUT
    return min(elapsed, TIMEOUT), None


FIT_METHODS = {
    'stacking': {
        'fit_fn': _fit_stacking,
        'max_vars_cap': 15,
        'label': 'Stacking',
        'cache_prefix': 'rq3_stacking_chunk',
        'tsv_name': 'rq3_stacking.tsv',
        'kind': 'python',
    },
    'bma': {
        'fit_fn': _fit_bma,
        'max_vars_cap': 8,
        'label': 'BMA',
        'cache_prefix': 'rq3_bma_chunk',
        'tsv_name': 'rq3_bma.tsv',
        'kind': 'python',
    },
    'best_logit': {
        'fit_fn': _fit_best_logit,
        'max_vars_cap': 64,
        'label': 'Best-Logit',
        'cache_prefix': 'rq3_best_logit_chunk',
        'tsv_name': 'rq3_best_logit.tsv',
        'kind': 'python',
    },
    'mcmc': {
        'fit_fn': None,
        'max_vars_cap': 64,
        'label': 'MCMC',
        'cache_prefix': 'rq3_mcmc_chunk',
        'tsv_name': 'rq3_mcmc.tsv',
        'kind': 'r',
        'r_label': 'MCMC',
    },
    'bas': {
        'fit_fn': None,
        'max_vars_cap': 64,
        'label': 'BAS',
        'cache_prefix': 'rq3_bas_chunk',
        'tsv_name': 'rq3_bas.tsv',
        'kind': 'r',
        'r_label': 'BAS',
    },
}


def run_fit_cell(method, sample, vars_int, data, seed):
    cfg = FIT_METHODS[method]
    if cfg['kind'] == 'python':
        return _python_fit_cell(method, sample, vars_int, data, seed), None
    if cfg['kind'] == 'r':
        return _r_fit_cell(cfg['r_label'], sample, vars_int)
    raise ValueError(f"unknown method kind: {cfg['kind']!r}")


def run_fit_phase(args):
    cfg = FIT_METHODS[args.method]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'logs')
    cache_dir = os.path.join(base_dir, 'cache')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if args.merge:
        results = []
        for fname in sorted(os.listdir(cache_dir)):
            if fname.startswith(cfg['cache_prefix']) and fname.endswith('.json'):
                with open(os.path.join(cache_dir, fname)) as f:
                    results.extend(json.load(f))
        out = os.path.join(log_dir, cfg['tsv_name'])
        with open(out, 'w') as f:
            f.write('method vars sample run_id time\n')
            for row in results:
                f.write(f"{cfg['label']} {row['vars']} {row['sample']} {row['run_id']} {row['time']:.4f}\n")
        print(f'Merged {len(results)} measurements -> {out}')
        return

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS
    if args.chunk is not None:
        total = len(configs)
        chunk_size = (total + args.n_chunks - 1) // args.n_chunks
        start_i = args.chunk * chunk_size
        end_i = min(start_i + chunk_size, total)
        configs = configs[start_i:end_i]
        print(f"=== RQ3-A [{args.method}] chunk {args.chunk}/{args.n_chunks}: configs {start_i}-{end_i - 1} ===")

    print(f"Method: {args.method} ({cfg['kind']}), Configs: {len(configs)}, Replicates: {args.n_runs}, Timeout: {TIMEOUT}s")

    data = None
    if cfg['kind'] == 'python':
        print('Loading data...')
        data = pd.read_csv(DATA_FILE)
        print(f'Shape: {data.shape}')

    results = []
    for idx, (sample, vars_int) in enumerate(configs):
        for run_id in range(args.n_runs):
            seed_i = args.seed + run_id
            print(f"  [{idx + 1}/{len(configs)}] s={sample} v={vars_int} run={run_id}", end=' ', flush=True)
            elapsed, err = run_fit_cell(args.method, sample, vars_int, data, seed_i)
            results.append({
                'method': cfg['label'],
                'vars': vars_int,
                'sample': sample,
                'run_id': run_id,
                'time': round(elapsed, 4),
            })
            tag = ' TIMEOUT' if elapsed >= TIMEOUT else ''
            err_tag = f' [{err}]' if err else ''
            print(f'-> {elapsed:.2f}s{tag}{err_tag}')

    if args.chunk is not None:
        out = os.path.join(cache_dir, f"{cfg['cache_prefix']}{args.chunk}.json")
        with open(out, 'w') as f:
            json.dump(results, f)
    else:
        out = os.path.join(log_dir, cfg['tsv_name'])
        with open(out, 'w') as f:
            f.write('method vars sample run_id time\n')
            for row in results:
                f.write(f"{cfg['label']} {row['vars']} {row['sample']} {row['run_id']} {row['time']:.4f}\n")
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# RQ3-B online benchmark
# ---------------------------------------------------------------------------

def measure_online_latency_pair(stacking, bma, X_stream, y_drifted):
    rows = []
    for t in range(len(y_drifted)):
        x_raw = X_stream.iloc[t:t + 1]
        y_t = float(y_drifted[t])

        t0 = time.perf_counter()
        stacking.online_update(x_raw, y_t)
        rows.append({'method': 'Stacking-Online', 't': t, 'time_seconds': time.perf_counter() - t0})

        t0 = time.perf_counter()
        bma.online_update(x_raw, y_t)
        rows.append({'method': 'BMA', 't': t, 'time_seconds': time.perf_counter() - t0})
    return rows


def scaled_drift_intervals(stream_len, scenario):
    intervals = DRIFT_SCENARIOS.get(scenario, [])
    if scenario == 'none' or not intervals or stream_len <= 0:
        return []
    scaled = []
    for start, end in intervals:
        s = int(round(start * stream_len / BASE_STREAM_LEN))
        e = int(round(end * stream_len / BASE_STREAM_LEN))
        s = max(0, min(s, max(stream_len - 1, 0)))
        e = max(s + 1, min(max(e, s + 1), stream_len))
        if scaled and s <= scaled[-1][1]:
            prev_s, prev_e = scaled[-1]
            scaled[-1] = (prev_s, max(prev_e, e))
        else:
            scaled.append((s, e))
    return scaled


def apply_scaled_drift(y, scenario):
    y_drifted = np.asarray(y).copy()
    intervals = scaled_drift_intervals(len(y_drifted), scenario)
    for start, end in intervals:
        y_drifted[start:end] = 1 - y_drifted[start:end]
    return y_drifted, intervals


def filter_configs(configs, sample_filter=None, vars_filter=None):
    out = configs
    if sample_filter:
        sample_filter = set(sample_filter)
        out = [cfg for cfg in out if cfg[0] in sample_filter]
    if vars_filter:
        vars_filter = set(vars_filter)
        out = [cfg for cfg in out if cfg[1] in vars_filter]
    return out


def build_grid_split(data, sample, vars_int, warmup_ratio):
    if sample < 4:
        raise ValueError('sample must be >= 4')
    if not 0.1 <= warmup_ratio <= 0.9:
        raise ValueError('warmup_ratio must be in [0.1, 0.9]')

    feature_cols = [f'x{i}' for i in range(1, vars_int + 1)]
    block = data.loc[:sample - 1, feature_cols + ['hazard']].reset_index(drop=True)
    warmup_size = int(round(sample * warmup_ratio))
    warmup_size = min(max(warmup_size, 2), sample - 1)
    X_warmup = block.iloc[:warmup_size][feature_cols].reset_index(drop=True)
    y_warmup = block.iloc[:warmup_size]['hazard'].to_numpy()
    X_stream = block.iloc[warmup_size:][feature_cols].reset_index(drop=True)
    y_stream = block.iloc[warmup_size:]['hazard'].to_numpy()
    return X_warmup, y_warmup, X_stream, y_stream, warmup_size


def build_online_model(method, X_warmup, y_warmup, args):
    if method == 'stacking':
        from stacking import StackingEnsemble
        model = StackingEnsemble(
            n_folds=min(10, max(2, len(y_warmup) // 20)),
            random_state=args.seed,
            online_lr=args.online_lr,
            mini_batch_size=args.mini_batch,
            alpha=args.alpha,
            drift_threshold=args.drift_threshold,
        )
        model.fit(X_warmup, y_warmup)
        return model, 'Stacking-Online'
    if method == 'bma':
        from bma import BMA
        model = BMA(
            y_warmup, X_warmup,
            RegType='Logit',
            Verbose=False,
            MaxVars=X_warmup.shape[1],
            retrain_every=args.retrain_every,
        ).fit()
        return model, 'BMA'
    raise ValueError(f'unknown online method: {method!r}')


def measure_online_method_latency(method, X_warmup, y_warmup, X_stream, y_drifted, args):
    model, label = build_online_model(method, X_warmup, y_warmup, args)
    times = []
    for t in range(len(y_drifted)):
        x_raw = X_stream.iloc[t:t + 1]
        y_t = float(y_drifted[t])
        t0 = time.perf_counter()
        model.online_update(x_raw, y_t)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times, dtype=float)
    return {
        'method': label,
        'time': float(times.mean()) if len(times) else 0.0,
        'median': float(np.median(times)) if len(times) else 0.0,
        'p95': float(np.quantile(times, 0.95)) if len(times) else 0.0,
        'total': float(times.sum()) if len(times) else 0.0,
        'status': 'ok',
    }


def _online_grid_worker(method, sample, vars_int, args, result_queue):
    try:
        data = pd.read_csv(DATA_FILE)
        X_warmup, y_warmup, X_stream, y_stream, warmup_size = build_grid_split(
            data, sample, vars_int, args.warmup_ratio)
        if args.limit_stream is not None:
            X_stream = X_stream.iloc[:args.limit_stream].reset_index(drop=True)
            y_stream = y_stream[:args.limit_stream]
        y_drifted, intervals = apply_scaled_drift(y_stream, args.drift_scenario)
        result = measure_online_method_latency(method, X_warmup, y_warmup, X_stream, y_drifted, args)
        result.update({
            'vars': vars_int,
            'sample': sample,
            'warmup': warmup_size,
            'stream': len(y_drifted),
            'drift': args.drift_scenario,
            'retrain_every': args.retrain_every,
            'status': result['status'],
            'drift_flips': int((y_drifted != y_stream).sum()),
            'drift_intervals': intervals,
        })
        result_queue.put(result)
    except Exception as exc:
        result_queue.put({
            'method': 'Stacking-Online' if method == 'stacking' else 'BMA',
            'vars': vars_int,
            'sample': sample,
            'time': float(TIMEOUT),
            'median': float(TIMEOUT),
            'p95': float(TIMEOUT),
            'total': float(TIMEOUT),
            'warmup': 0,
            'stream': 0,
            'drift': args.drift_scenario,
            'retrain_every': args.retrain_every,
            'status': f'error:{type(exc).__name__}',
        })


def run_online_grid_cell(method, sample, vars_int, args):
    result_queue = mp.Queue()
    process = mp.Process(target=_online_grid_worker, args=(method, sample, vars_int, args, result_queue))
    process.start()
    process.join(timeout=TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            'method': 'Stacking-Online' if method == 'stacking' else 'BMA',
            'vars': vars_int,
            'sample': sample,
            'time': float(TIMEOUT),
            'median': float(TIMEOUT),
            'p95': float(TIMEOUT),
            'total': float(TIMEOUT),
            'warmup': 0,
            'stream': 0,
            'drift': args.drift_scenario,
            'retrain_every': args.retrain_every,
            'status': 'timeout',
        }
    try:
        return result_queue.get_nowait()
    except Exception:
        return {
            'method': 'Stacking-Online' if method == 'stacking' else 'BMA',
            'vars': vars_int,
            'sample': sample,
            'time': float(TIMEOUT),
            'median': float(TIMEOUT),
            'p95': float(TIMEOUT),
            'total': float(TIMEOUT),
            'warmup': 0,
            'stream': 0,
            'drift': args.drift_scenario,
            'retrain_every': args.retrain_every,
            'status': 'no-result',
        }


def online_grid_cache_prefix(args):
    return f"rq3_online_grid_drift{args.drift_scenario}_re{args.retrain_every}_chunk"


def online_grid_out_path(base_dir, args):
    return os.path.join(base_dir, 'logs', f'rq3_online_grid_drift{args.drift_scenario}_re{args.retrain_every}.tsv')


def write_online_grid_tsv(out_path, rows):
    with open(out_path, 'w') as f:
        f.write('method vars sample run_id time median p95 total stream warmup drift retrain_every status\n')
        for row in rows:
            f.write(
                f"{row['method']} {row['vars']} {row['sample']} {row['run_id']} "
                f"{row['time']:.6f} {row['median']:.6f} {row['p95']:.6f} "
                f"{row['total']:.6f} {row['stream']} {row['warmup']} "
                f"{row['drift']} {row['retrain_every']} {row['status']}\n"
            )


def run_online_legacy(args, base_dir):
    X_warmup, y_warmup, X_stream, y_stream, _, _ = build_data_splits(base_dir)
    y_drifted = apply_drift(y_stream, args.drift_scenario)
    if args.limit_stream is not None:
        X_stream = X_stream.iloc[:args.limit_stream].reset_index(drop=True)
        y_drifted = y_drifted[:args.limit_stream]

    print(f"=== RQ3-B legacy drift={args.drift_scenario} (retrain_every={args.retrain_every}, samples={len(y_drifted)}) ===")
    print(f"Warmup: {len(y_warmup)}, drift flips: {int((y_drifted != y_stream[:len(y_drifted)]).sum())}")

    stacking, _ = build_online_model('stacking', X_warmup, y_warmup, args)
    bma, _ = build_online_model('bma', X_warmup, y_warmup, args)

    t_start = time.time()
    rows = measure_online_latency_pair(stacking, bma, X_stream, y_drifted)
    elapsed = time.time() - t_start
    print(f"Done: {elapsed:.1f}s ({2 * len(y_drifted)} measurements, ~{elapsed * 1000 / max(2 * len(y_drifted), 1):.1f}ms each on avg)")

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f'rq3_online_drift{args.drift_scenario}.tsv')
    with open(out_path, 'w') as f:
        f.write('method\tdrift\tt\ttime_seconds\n')
        for row in rows:
            f.write(f"{row['method']}\t{args.drift_scenario}\t{row['t']}\t{row['time_seconds']:.6f}\n")
    print(f'Saved: {out_path}')


def run_online_grid(args, base_dir):
    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS
    configs = filter_configs(configs, args.samples, args.vars)
    methods = ['stacking', 'bma'] if args.method == 'all' else [args.method]
    log_dir = os.path.join(base_dir, 'logs')
    cache_dir = os.path.join(base_dir, 'cache')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if not configs:
        raise ValueError('No grid cells left after applying --samples/--vars filters.')

    if args.merge:
        rows = []
        prefix = online_grid_cache_prefix(args)
        for fname in sorted(os.listdir(cache_dir)):
            if fname.startswith(prefix) and fname.endswith('.json'):
                with open(os.path.join(cache_dir, fname)) as f:
                    rows.extend(json.load(f))
        rows.sort(key=lambda row: (row['sample'], row['vars'], row['run_id'], row['method']))
        out_path = online_grid_out_path(base_dir, args)
        write_online_grid_tsv(out_path, rows)
        print(f'Merged {len(rows)} measurements -> {out_path}')
        return

    if args.chunk is not None:
        total = len(configs)
        chunk_size = (total + args.n_chunks - 1) // args.n_chunks
        start_i = args.chunk * chunk_size
        end_i = min(start_i + chunk_size, total)
        configs = configs[start_i:end_i]
        print(f"=== RQ3-B grid drift={args.drift_scenario} re={args.retrain_every} chunk {args.chunk}/{args.n_chunks}: configs {start_i}-{end_i - 1} ===")

    print(f"=== RQ3-B grid drift={args.drift_scenario} re={args.retrain_every} ===")
    print(f"Mode: amortized per-sample online_update latency, Configs: {len(configs)}, Replicates: {args.n_runs}, Warmup ratio: {args.warmup_ratio:.2f}, Timeout: {TIMEOUT}s")

    rows = []
    total_cells = len(configs) * len(methods) * args.n_runs
    cell_idx = 0
    for sample, vars_int in configs:
        for run_id in range(args.n_runs):
            args.seed = args.seed_base + run_id
            for method in methods:
                cell_idx += 1
                print(f"  [{cell_idx}/{total_cells}] {method} s={sample} v={vars_int} run={run_id}", end=' ', flush=True)
                result = run_online_grid_cell(method, sample, vars_int, args)
                result['run_id'] = run_id
                rows.append(result)
                tag = ' TIMEOUT' if result['time'] >= TIMEOUT else ''
                print(f"-> mean={result['time']:.4f}s status={result['status']}{tag}")

    rows.sort(key=lambda row: (row['sample'], row['vars'], row['run_id'], row['method']))
    if args.chunk is not None:
        out_path = os.path.join(cache_dir, f'{online_grid_cache_prefix(args)}{args.chunk}.json')
        with open(out_path, 'w') as f:
            json.dump(rows, f)
    else:
        out_path = online_grid_out_path(base_dir, args)
        write_online_grid_tsv(out_path, rows)
    print(f'Saved: {out_path}')


def run_online_phase(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.online_mode == 'legacy':
        run_online_legacy(args, base_dir)
    else:
        run_online_grid(args, base_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def validate_args(parser, args):
    if args.phase == 'fit':
        if args.method is None:
            args.method = 'stacking'
        if args.method not in FIT_METHODS:
            parser.error(f"--method must be one of {sorted(FIT_METHODS)} for --phase fit")
        if args.n_runs < 1:
            parser.error('--n-runs must be >= 1')
        if args.n_chunks < 1:
            parser.error('--n-chunks must be >= 1')
        return args

    if args.method is None:
        args.method = 'all'
    if args.method not in {'all', 'stacking', 'bma'}:
        parser.error("--method must be one of {'all','stacking','bma'} for --phase online")
    if args.n_runs < 1:
        parser.error('--n-runs must be >= 1')
    if args.n_chunks < 1:
        parser.error('--n-chunks must be >= 1')
    if args.online_mode == 'legacy' and (args.chunk is not None or args.merge):
        parser.error('--chunk/--merge are only supported for --phase online --mode grid')
    args.drift_scenario = args.drift_scenario or '6x'
    return args


def main():
    parser = argparse.ArgumentParser(description='RQ3: computational cost benchmark')
    parser.add_argument('--phase', choices=['fit', 'online'], default='fit')
    parser.add_argument('--method', default=None,
                        help='fit: stacking|bma|best_logit|mcmc|bas; online: all|stacking|bma')
    parser.add_argument('--n-runs', type=int, default=1,
                        help='Replicates per (sample, vars) cell.')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed; replicate i uses seed = base + i.')
    parser.add_argument('--chunk', type=int, default=None)
    parser.add_argument('--n-chunks', type=int, default=8)
    parser.add_argument('--merge', action='store_true')

    parser.add_argument('--online-mode', choices=['grid', 'legacy'], default='grid')
    parser.add_argument('--drift-scenario',
                        choices=[k for k in DRIFT_SCENARIOS.keys() if k != 'none'],
                        default=None)
    parser.add_argument('--retrain-every', type=int, default=100)
    parser.add_argument('--warmup-ratio', type=float, default=0.5)
    parser.add_argument('--samples', nargs='*', type=int, default=None)
    parser.add_argument('--vars', nargs='*', type=int, default=None)
    parser.add_argument('--limit-stream', type=int, default=None)
    parser.add_argument('--mini-batch', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--online-lr', type=float, default=0.05)
    parser.add_argument('--drift-threshold', type=float, default=0.15)

    args = parser.parse_args()
    args.seed_base = args.seed
    args = validate_args(parser, args)
    np.random.seed(args.seed)

    if args.phase == 'fit':
        run_fit_phase(args)
    else:
        run_online_phase(args)


if __name__ == '__main__':
    mp.freeze_support()
    main()
