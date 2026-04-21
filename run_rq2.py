#!/usr/bin/env python3
"""
run_rq2.py
RQ2: Adaptation decision quality -- Stacking-Online vs Stacking-Offline vs BMA.

Only Setting A is supported. The online path sweeps drift-detection thresholds
in parallel: setup/offline/bma are threshold-independent and shared, while
online snapshots, online_ga chunks, and the merged output are per-threshold.

Single-node:
    python run_rq2.py --seed 42 --ga-workers 4 --drift-threshold 0.15

Distributed (multi-node SLURM):
    python run_rq2.py --phase setup                                               # train & pickle (shared)
    python run_rq2.py --phase online   --drift-threshold 0.15                     # snapshots per threshold
    python run_rq2.py --phase online_ga --drift-threshold 0.15 --chunk 0 --n-chunks 10
    python run_rq2.py --phase offline   --chunk 0 --n-chunks 10                   # shared
    python run_rq2.py --phase bma       --chunk 3 --n-chunks 10                   # shared
    python run_rq2.py --phase merge     --drift-threshold 0.15                    # per threshold
"""

import copy
import json
import pickle
import argparse
import os
import sys
import time
import warnings
import platform
import multiprocessing as mp

import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from geneticalgorithm import geneticalgorithm as ga

from stacking import StackingEnsemble
from bma import BMA

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = ['illuminance', 'smoke', 'size', 'distance', 'firm',
                'power', 'band', 'quality', 'speed']
TARGET_COL = 'hazard'

VARS_DICT = {
    5: ('power',   ['int'],  [13, 78],       0.8),
    6: ('band',    ['real'], [14.7, 46.58],   0.4),
    7: ('quality', ['real'], [0, 147.19],     0.2),
    8: ('speed',   ['int'],  [15, 64],        0.1),
}
INDEX_SET = [5, 6, 7, 8]

GA_PARAMS = {
    'max_num_iteration': 50,
    'population_size': 100,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None,
}


# ---------------------------------------------------------------------------
# BMA wrapper
# ---------------------------------------------------------------------------

class BMAPredictor:
    def __init__(self, bma_model):
        self.bma = bma_model

    def predict_single(self, row_data_with_const):
        pred = self.bma.predict(np.asarray(row_data_with_const))
        return float(np.ravel(pred)[0]) if hasattr(pred, '__len__') else float(pred)


# ---------------------------------------------------------------------------
# GA core
# ---------------------------------------------------------------------------

def make_fitness(model, row_data, initial_values):
    def fitness(X):
        data = row_data.copy()
        for i, k in enumerate(INDEX_SET):
            data.iloc[0, data.columns.get_loc(VARS_DICT[k][0])] = X[i]
        prediction = model.predict_single(data)
        if prediction < 0.51:
            prediction /= 10
        delta_change = sum(
            VARS_DICT[k][3] * abs(X[i] - initial_values[i]) /
            (VARS_DICT[k][2][1] - VARS_DICT[k][2][0])
            for i, k in enumerate(INDEX_SET)
        )
        return delta_change - prediction
    return fitness


def run_single_ga(model, row_data_with_const):
    initial_values = [row_data_with_const[VARS_DICT[k][0]].values[0]
                      for k in INDEX_SET]
    fitness_fn = make_fitness(model, row_data_with_const, initial_values)
    vartype = np.array([VARS_DICT[k][1] for k in INDEX_SET])
    varbound = np.array([VARS_DICT[k][2] for k in INDEX_SET])

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        ga_model = ga(
            function=fitness_fn, dimension=len(INDEX_SET),
            variable_type_mixed=vartype, variable_boundaries=varbound,
            convergence_curve=False, progress_bar=False,
            algorithm_parameters=GA_PARAMS)
        ga_model.run()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    assignment = ga_model.output_dict['variable']
    new_row = row_data_with_const.copy()
    for i, k in enumerate(INDEX_SET):
        new_row.iloc[0, new_row.columns.get_loc(VARS_DICT[k][0])] = assignment[i]
    return float(model.predict_single(new_row)), new_row


# Parallel GA worker
_worker_model = None

def _init_worker(model):
    global _worker_model
    _worker_model = model

def _ga_worker(args):
    global _worker_model
    stream_idx, row_values, row_columns = args
    row_data = pd.DataFrame([row_values], columns=row_columns)
    pred, new_row = run_single_ga(_worker_model, row_data)
    return stream_idx, pred, new_row

def run_batch_ga(model, hazard0_samples, n_workers):
    args_list = [(idx, row.values[0].tolist(), list(row.columns))
                 for idx, row in hazard0_samples]
    if n_workers <= 1:
        global _worker_model
        _worker_model = model
        results = [_ga_worker(a) for a in args_list]
    else:
        ctx = mp.get_context('fork') if platform.system() != 'Windows' else mp.get_context('spawn')
        with ctx.Pool(processes=n_workers, initializer=_init_worker,
                      initargs=(model,)) as pool:
            results = pool.map(_ga_worker, args_list)
    return {idx: (pred, new_row) for idx, pred, new_row in results}


# ---------------------------------------------------------------------------
# Data / helpers
# ---------------------------------------------------------------------------

def load_and_encode(path):
    df = pd.read_csv(path)
    df['firm'] = (df['firm'] == 'Yes') * 1
    return df

def compute_re_success(pred, oracle_val):
    if oracle_val > 1e-7:
        re = abs(pred - oracle_val) / oracle_val
    else:
        re = float('nan')
    return float(re), bool(pred > 0.5 and oracle_val > 0.5)

def cache_dir(base_dir):
    d = os.path.join(base_dir, 'cache')
    os.makedirs(d, exist_ok=True)
    return d

def threshold_tag(th):
    """Canonical filename fragment for a drift threshold, e.g. 0.015 -> 'th0.015'."""
    return f"th{float(th):.3f}"

def build_data_splits(base_dir):
    """Setting A: full training set as warmup, full validation set as stream."""
    df_train = load_and_encode(os.path.join(base_dir, 'data', 'training_rescueRobot_450.csv'))
    df_valid = load_and_encode(os.path.join(base_dir, 'data', 'validation_rescueRobot_450.csv'))
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL].values
    X_valid = df_valid[FEATURE_COLS]
    y_valid = df_valid[TARGET_COL].values

    X_warmup = X_train.iloc[:450].reset_index(drop=True)
    y_warmup = y_train[:450]
    X_stream = X_valid.reset_index(drop=True)
    y_stream = y_valid

    return X_warmup, y_warmup, X_stream, y_stream, X_valid, y_valid


# ---------------------------------------------------------------------------
# Phase: setup  (train models, identify hazard0, pickle everything)
# ---------------------------------------------------------------------------

def phase_setup(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    X_warmup, y_warmup, X_stream, y_stream, X_valid, y_valid = \
        build_data_splits(base_dir)

    print("=== RQ2 Setup A ===")
    print(f"Warmup: {len(y_warmup)}, Stream: {len(y_stream)}")

    print("Training Stacking...")
    stacking = StackingEnsemble(
        n_folds=10, random_state=args.seed,
        online_lr=args.online_lr, mini_batch_size=args.mini_batch)
    stacking.fit(X_warmup, y_warmup)
    stacking_offline = copy.deepcopy(stacking)

    # Identify hazard=0 samples and prepare row data
    hazard0_indices = [t for t in range(len(y_stream)) if y_stream[t] == 0]
    if args.limit is not None:
        hazard0_indices = hazard0_indices[:args.limit]
    print(f"Hazard=0 samples: {len(hazard0_indices)}")

    hazard0_rows = {}
    for t in hazard0_indices:
        # has_constant='add' forces constant column even for single-row DataFrames
        # (default 'skip' detects all cols as constant when there's only 1 row)
        hazard0_rows[t] = add_constant(X_stream.iloc[t:t+1], has_constant='add')

    # Save — BMA/Oracle contain mpmath matrices that can't be pickled,
    # so we save data splits instead and retrain BMA in each phase that needs it
    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'stacking_online': stacking,
            'stacking_offline': stacking_offline,
            'X_warmup': X_warmup, 'y_warmup': y_warmup,
            'X_valid': X_valid, 'y_valid': y_valid,
            'X_stream': X_stream, 'y_stream': y_stream,
            'hazard0_indices': hazard0_indices,
            'hazard0_rows': hazard0_rows,
            'warmup_size': len(y_warmup),
        }, f)
    print(f"Saved: {pkl_path}")


# ---------------------------------------------------------------------------
# Phase: online  (sequential prequential loop - the irreducible bottleneck)
# ---------------------------------------------------------------------------

def phase_online(args):
    """Run prequential loop WITHOUT GA. Save meta-learner snapshots at hazard=0 points.

    The GA results don't feed back into online_update, so we can separate:
    1. This phase: fast sequential loop saving model state at each hazard=0 point
    2. phase_online_ga: embarrassingly parallel GA using saved snapshots

    Output is per-threshold: one snapshot file and one diagnostics file per
    drift-detection threshold being swept.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    tag = threshold_tag(args.drift_threshold)

    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    stacking = data['stacking_online']
    # Override drift threshold for this sweep point
    stacking.drift_threshold = args.drift_threshold
    X_stream = data['X_stream']
    y_stream = data['y_stream']
    hazard0_set = set(data['hazard0_indices'])

    print(f"=== RQ2 Online A [{tag}]: {len(y_stream)} samples, "
          f"{len(hazard0_set)} snapshots (no GA) ===")
    t0 = time.time()

    # Run prequential loop, save meta-learner state at each hazard=0 point
    snapshots = {}
    for t in range(len(y_stream)):
        x_raw = X_stream.iloc[t:t+1]
        if t in hazard0_set:
            # Save meta-learner state BEFORE this sample's update
            # (this is the model state used for prediction at time t)
            snapshots[t] = (
                stacking.meta_learner.coef_.copy(),
                stacking.meta_learner.intercept_.copy(),
            )
        stacking.online_update(x_raw, float(y_stream[t]))

    elapsed = time.time() - t0
    diag = stacking.get_diagnostics()
    print(f"Done: {elapsed:.1f}s, snapshots={len(snapshots)}, "
          f"drifts={diag['drift_count']}")

    # Save snapshots (tiny: just coef/intercept arrays per hazard=0 point)
    snap_path = os.path.join(cd, f'rq2_A_snapshots_{tag}.pkl')
    with open(snap_path, 'wb') as f:
        pickle.dump(snapshots, f)

    # Save diagnostics
    diag_path = os.path.join(cd, f'rq2_A_online_diag_{tag}.json')
    with open(diag_path, 'w') as f:
        json.dump({'elapsed': round(elapsed, 2),
                   'drift_threshold': args.drift_threshold,
                   'diagnostics': {'drift_count': diag['drift_count'],
                                   'update_count': diag['update_count']}}, f)
    print(f"Saved: {snap_path}, {diag_path}")


# ---------------------------------------------------------------------------
# Phase: online_ga  (chunked parallel GA using saved model snapshots)
# ---------------------------------------------------------------------------

def phase_online_ga(args):
    """Run GA for a chunk of hazard=0 samples using per-timestep model snapshots.

    Per-threshold: snapshot file and output chunk are keyed on the threshold.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    tag = threshold_tag(args.drift_threshold)

    # Load setup data (for base_models and hazard0 rows)
    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Load per-threshold meta-learner snapshots
    snap_path = os.path.join(cd, f'rq2_A_snapshots_{tag}.pkl')
    with open(snap_path, 'rb') as f:
        snapshots = pickle.load(f)

    model = data['stacking_online']  # has correct base_models
    hazard0_indices = data['hazard0_indices']
    hazard0_rows = data['hazard0_rows']

    # Retrain Oracle (BMA can't be pickled due to mpmath)
    print("Retraining Oracle BMA...")
    oracle = BMA(data['y_valid'], add_constant(data['X_valid']),
                 RegType='Logit', Verbose=False).fit()

    # Chunk selection
    total = len(hazard0_indices)
    chunk_size = (total + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    my_indices = hazard0_indices[start:end]

    print(f"=== RQ2 online_ga A [{tag}] chunk {args.chunk}/{args.n_chunks}: "
          f"samples {start}-{end-1} ({len(my_indices)} GA runs) ===")
    t0 = time.time()
    results = {}

    for i, t in enumerate(my_indices):
        # Restore meta-learner state for this timestamp
        coef, intercept = snapshots[t]
        model.meta_learner.coef_ = coef
        model.meta_learner.intercept_ = intercept

        row_wc = hazard0_rows[t]
        pred, new_row = run_single_ga(model, row_wc)
        oracle_pred = oracle.predict(np.asarray(new_row))
        oracle_val = float(np.ravel(oracle_pred)[0])
        re, success = compute_re_success(pred, oracle_val)
        results[str(t)] = {
            'pred': round(pred, 6), 'oracle': round(oracle_val, 6),
            'RE': round(re, 6) if not np.isnan(re) else None,
            'success': success}
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(my_indices)}  ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s")

    out = os.path.join(cd, f'rq2_A_online_{tag}_chunk{args.chunk}.json')
    with open(out, 'w') as f:
        json.dump({'results': results, 'elapsed': round(elapsed, 2),
                   'chunk': args.chunk, 'n_chunks': args.n_chunks,
                   'drift_threshold': args.drift_threshold}, f)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Phase: offline / bma  (chunked, embarrassingly parallel)
# ---------------------------------------------------------------------------

def phase_batch(args, method_key):
    """Run GA for a chunk of hazard=0 samples using frozen model.

    Threshold-independent: offline stacking and BMA do not perform online
    updates, so these outputs are shared across the threshold sweep.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Retrain BMA/Oracle (mpmath matrices can't be pickled)
    if method_key == 'offline':
        model = data['stacking_offline']
    else:
        print("Retraining BMA...")
        bma_model = BMA(data['y_warmup'], add_constant(data['X_warmup']),
                        RegType='Logit', Verbose=False).fit()
        model = BMAPredictor(bma_model)

    print("Retraining Oracle BMA...")
    oracle = BMA(data['y_valid'], add_constant(data['X_valid']),
                 RegType='Logit', Verbose=False).fit()
    hazard0_indices = data['hazard0_indices']
    hazard0_rows = data['hazard0_rows']

    # Chunk selection
    total = len(hazard0_indices)
    chunk_size = (total + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    my_indices = hazard0_indices[start:end]

    print(f"=== RQ2 {method_key} A chunk {args.chunk}/{args.n_chunks}: "
          f"samples {start}-{end-1} ({len(my_indices)} GA runs) ===")
    t0 = time.time()
    results = {}

    for i, t in enumerate(my_indices):
        row_wc = hazard0_rows[t]
        pred, new_row = run_single_ga(model, row_wc)
        oracle_pred = oracle.predict(np.asarray(new_row))
        oracle_val = float(np.ravel(oracle_pred)[0])
        re, success = compute_re_success(pred, oracle_val)
        results[str(t)] = {
            'pred': round(pred, 6), 'oracle': round(oracle_val, 6),
            'RE': round(re, 6) if not np.isnan(re) else None,
            'success': success}
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(my_indices)}  ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s")

    out = os.path.join(cd, f'rq2_A_{method_key}_chunk{args.chunk}.json')
    with open(out, 'w') as f:
        json.dump({'results': results, 'elapsed': round(elapsed, 2),
                   'chunk': args.chunk, 'n_chunks': args.n_chunks}, f)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Phase: merge  (combine all chunks into final output)
# ---------------------------------------------------------------------------

def phase_merge(args):
    """Merge online (per-threshold), offline, and BMA chunks for one threshold."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    tag = threshold_tag(args.drift_threshold)

    # Load setup metadata
    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        setup = pickle.load(f)

    warmup_size = setup['warmup_size']
    stream_size = len(setup['y_stream'])
    hazard0_count = len(setup['hazard0_indices'])

    # Load and merge per-threshold online GA chunks
    online_prefix = f'rq2_A_online_{tag}_chunk'
    online_results = {}
    for fname in sorted(os.listdir(cd)):
        if fname.startswith(online_prefix) and fname.endswith('.json'):
            with open(os.path.join(cd, fname)) as f:
                chunk = json.load(f)
            online_results.update(chunk['results'])

    # Load per-threshold online diagnostics
    diag_path = os.path.join(cd, f'rq2_A_online_diag_{tag}.json')
    online_diag = {}
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            online_diag = json.load(f).get('diagnostics', {})

    # Load and merge (shared) offline chunks
    offline_results = {}
    for fname in sorted(os.listdir(cd)):
        if fname.startswith('rq2_A_offline_chunk') and fname.endswith('.json'):
            with open(os.path.join(cd, fname)) as f:
                chunk = json.load(f)
            offline_results.update(chunk['results'])

    # Load and merge (shared) BMA chunks
    bma_results = {}
    for fname in sorted(os.listdir(cd)):
        if fname.startswith('rq2_A_bma_chunk') and fname.endswith('.json'):
            with open(os.path.join(cd, fname)) as f:
                chunk = json.load(f)
            bma_results.update(chunk['results'])

    print(f"=== RQ2 Merge A [{tag}]: online={len(online_results)}, "
          f"offline={len(offline_results)}, bma={len(bma_results)} ===")

    # Build final output
    methods = {}
    for name, data in [('Stacking-Online', online_results),
                       ('Stacking-Offline', offline_results),
                       ('BMA', bma_results)]:
        re_vals = [v['RE'] for v in data.values() if v.get('RE') is not None]
        succ_vals = [v['success'] for v in data.values()]
        per_sample = [{'t': int(t), **v} for t, v in sorted(data.items(), key=lambda x: int(x[0]))]

        med_re = float(np.median(re_vals)) if re_vals else float('nan')
        mean_re = float(np.mean(re_vals)) if re_vals else float('nan')
        succ_rate = float(np.mean(succ_vals)) if succ_vals else 0.0

        methods[name] = {
            'median_RE': round(med_re, 6), 'mean_RE': round(mean_re, 6),
            'success_rate': round(succ_rate, 6), 'n_samples': len(data),
            'per_sample': per_sample}
        print(f"  {name:<20s}  medRE={med_re:.4f}  success={succ_rate:.4f}")

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    output = {
        'setting': 'A', 'drift_threshold': args.drift_threshold,
        'warmup_size': warmup_size,
        'stream_size': stream_size, 'hazard0_count': hazard0_count,
        'methods': methods, 'diagnostics': online_diag}

    json_path = os.path.join(log_dir, f'rq2_A_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {json_path}")

    tsv_path = os.path.join(log_dir, f'rq2_A_{tag}_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write('method\tmedian_RE\tmean_RE\tsuccess_rate\tn_samples\n')
        for m, r in methods.items():
            f.write(f"{m}\t{r['median_RE']}\t{r['mean_RE']}\t"
                    f"{r['success_rate']}\t{r['n_samples']}\n")
    print(f"Saved: {tsv_path}")


# ---------------------------------------------------------------------------
# Single-node mode (runs all phases sequentially, with local parallelism)
# ---------------------------------------------------------------------------

def run_single_node(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    tag = threshold_tag(args.drift_threshold)
    t_start = time.time()

    X_warmup, y_warmup, X_stream, y_stream, X_valid, y_valid = \
        build_data_splits(base_dir)

    print(f"=== RQ2 Setting A [{tag}] (single-node) ===")
    print(f"Warmup: {len(y_warmup)}, Stream: {len(y_stream)}, "
          f"GA workers: {args.ga_workers}")

    # Train
    stacking_online = StackingEnsemble(
        n_folds=10, random_state=args.seed,
        online_lr=args.online_lr, mini_batch_size=args.mini_batch,
        drift_threshold=args.drift_threshold)
    stacking_online.fit(X_warmup, y_warmup)
    stacking_offline = copy.deepcopy(stacking_online)
    bma_model = BMA(y_warmup, add_constant(X_warmup),
                    RegType='Logit', Verbose=False).fit()
    bma_predictor = BMAPredictor(bma_model)
    oracle = BMA(y_valid, add_constant(X_valid),
                 RegType='Logit', Verbose=False).fit()

    hazard0_indices = [t for t in range(len(y_stream)) if y_stream[t] == 0]
    if args.limit:
        hazard0_indices = hazard0_indices[:args.limit]
    hazard0_set = set(hazard0_indices)
    hazard0_rows = {t: add_constant(X_stream.iloc[t:t+1], has_constant='add') for t in hazard0_indices}
    print(f"Hazard=0: {len(hazard0_indices)}")

    # Online sequential
    print(f"\n--- Online ({len(hazard0_indices)} GA) ---")
    t1 = time.time()
    online_results = {}
    ga_count = 0
    for t in range(len(y_stream)):
        x_raw = X_stream.iloc[t:t+1]
        if t in hazard0_set:
            pred, new_row = run_single_ga(stacking_online, hazard0_rows[t])
            ov = float(np.ravel(oracle.predict(np.asarray(new_row)))[0])
            re, succ = compute_re_success(pred, ov)
            online_results[str(t)] = {
                'pred': round(pred, 6), 'oracle': round(ov, 6),
                'RE': round(re, 6) if not np.isnan(re) else None,
                'success': succ}
            ga_count += 1
            if ga_count % 20 == 0:
                print(f"  Online GA: {ga_count}/{len(hazard0_indices)}")
        stacking_online.online_update(x_raw, float(y_stream[t]))
    p1 = time.time() - t1
    print(f"  Done: {p1:.1f}s")

    # Offline batch
    print(f"\n--- Offline batch ({args.ga_workers} workers) ---")
    t2 = time.time()
    h0_list = [(t, hazard0_rows[t]) for t in hazard0_indices]
    offline_preds = run_batch_ga(stacking_offline, h0_list, args.ga_workers)
    p2 = time.time() - t2
    print(f"  Done: {p2:.1f}s")

    # BMA batch
    print(f"\n--- BMA batch ({args.ga_workers} workers) ---")
    t3 = time.time()
    bma_preds = run_batch_ga(bma_predictor, h0_list, args.ga_workers)
    p3 = time.time() - t3
    print(f"  Done: {p3:.1f}s")

    # Build results
    methods = {}
    for name, src in [('Stacking-Online', None), ('Stacking-Offline', offline_preds),
                      ('BMA', bma_preds)]:
        data = {}
        for t in hazard0_indices:
            if name == 'Stacking-Online':
                data[str(t)] = online_results[str(t)]
            else:
                pred, new_row = src[t]
                ov = float(np.ravel(oracle.predict(np.asarray(new_row)))[0])
                re, succ = compute_re_success(pred, ov)
                data[str(t)] = {'pred': round(pred, 6), 'oracle': round(ov, 6),
                                'RE': round(re, 6) if not np.isnan(re) else None,
                                'success': succ}

        re_vals = [v['RE'] for v in data.values() if v.get('RE') is not None]
        succ_vals = [v['success'] for v in data.values()]
        per_sample = [{'t': int(k), **v} for k, v in sorted(data.items(), key=lambda x: int(x[0]))]
        med = float(np.median(re_vals)) if re_vals else float('nan')
        mn = float(np.mean(re_vals)) if re_vals else float('nan')
        sr = float(np.mean(succ_vals)) if succ_vals else 0.0
        methods[name] = {'median_RE': round(med, 6), 'mean_RE': round(mn, 6),
                         'success_rate': round(sr, 6), 'n_samples': len(data),
                         'per_sample': per_sample}
        print(f"  {name:<20s}  medRE={med:.4f}  success={sr:.4f}")

    elapsed = time.time() - t_start
    diag = stacking_online.get_diagnostics()
    print(f"\nTotal: {elapsed:.1f}s (online={p1:.0f}s offline={p2:.0f}s bma={p3:.0f}s)")

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    output = {
        'setting': 'A', 'drift_threshold': args.drift_threshold,
        'warmup_size': len(y_warmup),
        'stream_size': len(y_stream), 'hazard0_count': len(hazard0_indices),
        'elapsed_seconds': round(elapsed, 2),
        'phase_times': {'online': round(p1, 2), 'offline': round(p2, 2), 'bma': round(p3, 2)},
        'methods': methods,
        'diagnostics': {'drift_count': diag['drift_count'],
                        'update_count': diag['update_count']}}
    with open(os.path.join(log_dir, f'rq2_A_{tag}.json'), 'w') as f:
        json.dump(output, f, indent=2)
    tsv_path = os.path.join(log_dir, f'rq2_A_{tag}_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write('method\tmedian_RE\tmean_RE\tsuccess_rate\tn_samples\n')
        for m, r in methods.items():
            f.write(f"{m}\t{r['median_RE']}\t{r['mean_RE']}\t{r['success_rate']}\t{r['n_samples']}\n")
    print(f"Saved: logs/rq2_A_{tag}.json, logs/rq2_A_{tag}_summary.tsv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RQ2: Adaptation decision quality (Setting A)')
    parser.add_argument('--setting', choices=['A'], default='A',
                        help='Only Setting A is supported; flag kept for compatibility.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mini-batch', type=int, default=5)
    parser.add_argument('--drift-threshold', type=float, default=0.15,
                        help='ADWIN drift threshold (sweep this in parallel).')
    parser.add_argument('--online-lr', type=float, default=0.001)
    parser.add_argument('--limit', type=int, default=None)
    # Single-node mode
    parser.add_argument('--ga-workers', type=int, default=4)
    # Distributed mode
    parser.add_argument('--phase', choices=['setup', 'online', 'online_ga', 'offline', 'bma', 'merge'],
                        default=None, help='Distributed phase (omit for single-node)')
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--n-chunks', type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.phase is None:
        run_single_node(args)
    elif args.phase == 'setup':
        phase_setup(args)
    elif args.phase == 'online':
        phase_online(args)
    elif args.phase == 'online_ga':
        phase_online_ga(args)
    elif args.phase in ('offline', 'bma'):
        phase_batch(args, args.phase)
    elif args.phase == 'merge':
        phase_merge(args)


if __name__ == '__main__':
    mp.freeze_support()
    main()
