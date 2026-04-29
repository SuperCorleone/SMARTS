#!/usr/bin/env python3
"""
run_rq2.py
RQ2: Adaptation decision quality -- Stacking-Online vs Stacking-Offline vs BMA.

BMA is the base-paper TAAS2024 implementation (see bma.py) with MAPE-K online
updates. Only Setting A is supported. The online path sweeps drift-detection
thresholds in parallel: setup/offline are threshold-independent and shared,
while online snapshots, online_ga chunks, bma_online snapshots, bma_online_ga
chunks, and the merged output are per-threshold.

Single-node:
    python run_rq2.py --seed 42 --ga-workers 4 --drift-threshold 0.15

Distributed (multi-node SLURM):
    python run_rq2.py --phase setup                                                  # train & pickle (shared)
    python run_rq2.py --phase online       --drift-threshold 0.15                    # Stacking snapshots per threshold
    python run_rq2.py --phase online_ga    --drift-threshold 0.15 --chunk 0 --n-chunks 10
    python run_rq2.py --phase offline      --chunk 0 --n-chunks 10                   # Stacking-Offline (shared)
    python run_rq2.py --phase bma_online   --retrain-every 20                        # BMA snapshots (threshold-independent)
    python run_rq2.py --phase bma_online_ga --drift-threshold 0.15 --chunk 0 --n-chunks 10
    python run_rq2.py --phase merge        --drift-threshold 0.15                    # per threshold
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from stacking import StackingEnsemble
from bma import BMA
from run_rq1 import find_best_logit_ms, predict_logit_ms

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
    """Wraps BMA so it exposes the same predict_single(row_with_const) interface
    as StackingEnsemble, but strips the const column internally since BMA
    normalizes via its own StandardScaler (TAAS2024 design)."""

    def __init__(self, bma_model):
        self.bma = bma_model

    def predict_single(self, row_data_with_const):
        feature_cols = [c for c in row_data_with_const.columns if c != 'const']
        row_raw = row_data_with_const[feature_cols]
        pred = self.bma.predict(row_raw)
        return float(np.ravel(pred)[0]) if hasattr(pred, '__len__') else float(pred)


class BestLogitPredictor:
    """Wraps the TAAS2024 best-logit (model-selection) tuple so it exposes the
    same predict_single(row_with_const) interface as StackingEnsemble. Strips
    the 'const' column internally because find_best_logit_ms uses its own
    StandardScaler over the original feature columns."""

    def __init__(self, ms_result):
        self.ms_result = ms_result  # (idx_set, fitted_model, fitted_scaler) or None

    def predict_single(self, row_data_with_const):
        if self.ms_result is None:
            return 0.5
        feature_cols = [c for c in row_data_with_const.columns if c != 'const']
        row_raw = row_data_with_const[feature_cols]
        return float(predict_logit_ms(self.ms_result, row_raw))


# ---------------------------------------------------------------------------
# GA core
# ---------------------------------------------------------------------------

def make_fitness(model, row_data, initial_values):
    """GA fitness for self-adaptation (paper: maximize p(requirement satisfied), minimize cost).
    Model predicts p(hazard)=p(Y=1 here). Paper's Y=1 is "satisfied" — opposite label convention.
    So we MINIMIZE model's p(hazard) and minimize config-change cost.
    """
    def fitness(X):
        data = row_data.copy()
        for i, k in enumerate(INDEX_SET):
            data.iloc[0, data.columns.get_loc(VARS_DICT[k][0])] = X[i]
        prediction = model.predict_single(data)
        # Bias GA toward configurations that pull p(hazard) below 0.5 (safe).
        if prediction > 0.49:
            prediction *= 10
        delta_change = sum(
            VARS_DICT[k][3] * abs(X[i] - initial_values[i]) /
            (VARS_DICT[k][2][1] - VARS_DICT[k][2][0])
            for i, k in enumerate(INDEX_SET)
        )
        return delta_change + prediction
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
    """Frozen-model GA worker. Returns (idx, pred_at_new, pred_at_initial, new_row)."""
    global _worker_model
    stream_idx, row_values, row_columns = args
    row_data = pd.DataFrame([row_values], columns=row_columns)
    pred_initial = float(_worker_model.predict_single(row_data))
    pred, new_row = run_single_ga(_worker_model, row_data)
    return stream_idx, pred, pred_initial, new_row

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
    return {idx: (pred, pred_initial, new_row)
            for idx, pred, pred_initial, new_row in results}


def _ga_worker_with_coef(args):
    """Per-snapshot meta-learner GA worker. Returns (idx, pred_at_new,
    pred_at_initial, new_row). pred_initial is computed AFTER restoring the
    snapshot's coef/intercept so it reflects the time-t online state."""
    global _worker_model
    stream_idx, row_values, row_columns, coef, intercept = args
    _worker_model.meta_learner.coef_ = coef
    _worker_model.meta_learner.intercept_ = intercept
    row_data = pd.DataFrame([row_values], columns=row_columns)
    pred_initial = float(_worker_model.predict_single(row_data))
    pred, new_row = run_single_ga(_worker_model, row_data)
    return stream_idx, pred, pred_initial, new_row

def run_batch_ga_snapshots(model, samples, n_workers):
    """Parallel GA with per-sample meta-learner state.
    samples: list of (idx, row_df, coef, intercept)
    """
    args_list = [
        (idx, row.values[0].tolist(), list(row.columns), coef.copy(), intercept.copy())
        for idx, row, coef, intercept in samples
    ]
    if n_workers <= 1:
        global _worker_model
        _worker_model = model
        results = [_ga_worker_with_coef(a) for a in args_list]
    else:
        ctx = mp.get_context('fork') if platform.system() != 'Windows' else mp.get_context('spawn')
        with ctx.Pool(processes=n_workers, initializer=_init_worker,
                      initargs=(model,)) as pool:
            results = pool.map(_ga_worker_with_coef, args_list)
    return {idx: (pred, pred_initial, new_row)
            for idx, pred, pred_initial, new_row in results}


def _ga_worker_bma_coef(args):
    """Per-snapshot BMA-coefficients GA worker. Returns (idx, pred_at_new,
    pred_at_initial, new_row). pred_initial is computed AFTER restoring the
    snapshot's BMA coefficients."""
    global _worker_model
    stream_idx, row_values, row_columns, coefficients = args
    _worker_model.bma.coefficients = coefficients
    row_data = pd.DataFrame([row_values], columns=row_columns)
    pred_initial = float(_worker_model.predict_single(row_data))
    pred, new_row = run_single_ga(_worker_model, row_data)
    return stream_idx, pred, pred_initial, new_row

def run_batch_ga_bma_snapshots(predictor, samples, n_workers):
    """Parallel GA with per-sample BMA coefficient state.
    samples: list of (idx, row_df, coefficients)
    """
    args_list = [
        (idx, row.values[0].tolist(), list(row.columns), coefficients.copy())
        for idx, row, coefficients in samples
    ]
    if n_workers <= 1:
        global _worker_model
        _worker_model = predictor
        results = [_ga_worker_bma_coef(a) for a in args_list]
    else:
        ctx = mp.get_context('fork') if platform.system() != 'Windows' else mp.get_context('spawn')
        with ctx.Pool(processes=n_workers, initializer=_init_worker,
                      initargs=(predictor,)) as pool:
            results = pool.map(_ga_worker_bma_coef, args_list)
    return {idx: (pred, pred_initial, new_row)
            for idx, pred, pred_initial, new_row in results}


# ---------------------------------------------------------------------------
# Data / helpers
# ---------------------------------------------------------------------------

def load_and_encode(path):
    df = pd.read_csv(path)
    df['firm'] = (df['firm'] == 'Yes') * 1
    return df

class _RFOracle:
    """Calibrated Random-Forest oracle (option D): cross-family neutral
    baseline with isotonic post-hoc calibration to fix the over-confident
    leaf-vote behaviour of vanilla RF.

    Hyperparameters tuned for 450-row regime: bounded depth and minimum leaf
    size keep individual trees from memorising; 5-fold isotonic calibration
    smooths the marginal predictions. References:
      - Niculescu-Mizil & Caruana, 2005 (calibration).
      - Friedman 2001 (overfitting in unbounded RF on small data).
    """
    kind = 'rf'

    def __init__(self, X, y, random_state=42):
        base = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=1,
        )
        cal = CalibratedClassifierCV(base, method='isotonic', cv=5)
        cal.fit(X, np.asarray(y, dtype=int))
        self._cal = cal
        self._cols = list(X.columns)

    def predict(self, row_raw):
        return float(self._cal.predict_proba(row_raw[self._cols])[:, 1][0])


class _GPOracle:
    """Gaussian-Process classifier oracle (option F): explicitly probabilistic,
    well-calibrated, smooth out-of-distribution extrapolation. The standard
    "small-data calibrated oracle" choice (Rasmussen & Williams, 2006). Uses
    an RBF kernel scaled by a constant; lengthscale is optimised by the
    classifier on fit.

    StandardScaler is applied because RBF kernel distances assume features
    on comparable scales — without scaling the unit-disparate sensor channels
    (illuminance ∈ thousands vs. firm ∈ {0,1}) would dominate the kernel.
    """
    kind = 'gp'

    def __init__(self, X, y, random_state=42):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gpc = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=2,
            random_state=random_state,
            n_jobs=1,
        )
        gpc.fit(X_s, np.asarray(y, dtype=int))
        self._gpc = gpc
        self._scaler = scaler
        self._cols = list(X.columns)

    def predict(self, row_raw):
        row_s = self._scaler.transform(row_raw[self._cols])
        return float(self._gpc.predict_proba(row_s)[:, 1][0])


class _EnsembleOracle:
    """Cross-family ensemble oracle (option H): mean of {logit, rf, gp}
    predictions. Robust to any single oracle's idiosyncrasies; the standard
    "ensemble selection" approach (Caruana et al., 2004) applied to the
    oracle-construction problem.

    The component oracles share a single fit on (X, y) — building this oracle
    therefore costs sum-of-components.
    """
    kind = 'ensemble'

    def __init__(self, X, y, random_state=42):
        self._members = [
            _LogitOracle(X, y, random_state=random_state),
            _RFOracle(X, y, random_state=random_state),
            _GPOracle(X, y, random_state=random_state),
        ]

    def predict(self, row_raw):
        ps = [m.predict(row_raw) for m in self._members]
        return float(np.mean(ps))


class _LogitOracle:
    """TUNE-aligned oracle (SEAMS'22 §5.1; bma_adaptation.py:163-164).

    Verbatim TUNE construction:
        oracle = BMA(y_oracle, add_constant(X_oracle),
                     RegType='Logit', Verbose=True).fit()
    where (X_oracle, y_oracle) is the full validation set. The oracle is BMA
    over all 2^9 logit subset models, weighted by BIC posterior — NOT a single
    full-feature logit. We use the same `BMA` class as the predictor (bma.py),
    fit on the (held-out) validation split.

    Caveat (analysis for stacking applicability): because BMA-as-predictor
    shares the inductive bias of this oracle (averaged over the same Logit
    family), comparing BMA's RE against Stacking-Online's RE is biased in
    BMA's favour — Stacking lives in a different hypothesis class
    (CV-stacked logit subsets + SGD meta-learner) and will incur higher RE
    even when its decisions are equally good. Use this oracle when you want
    to reproduce TUNE's protocol exactly; pair with --oracle rf for a
    cross-family sanity check.
    """
    kind = 'logit'

    def __init__(self, X, y, random_state=42):
        # BMA expects raw features; it adds add_constant + StandardScaler
        # internally (see bma.py). TUNE's call passes add_constant(X) so the
        # constant is added explicitly; our BMA class adds it for us, so we
        # pass X directly to match the TUNE *semantics* (full-feature BMA-Logit
        # on validation) without double-adding the constant.
        self._bma = BMA(np.asarray(y, dtype=int), X,
                        RegType='Logit', Verbose=False).fit()
        self._cols = list(X.columns)

    def predict(self, row_raw):
        pred = self._bma.predict(row_raw[self._cols])
        return float(np.ravel(pred)[0])


# Map oracle kind → metric key. RE only for the TUNE-aligned BMA(Logit) oracle
# (it's the formula TUNE published, with the polarity-corrected denominator).
# Cross-family oracles use linear |pred - oracle| because dividing by an oracle
# from a different family has no theoretical grounding and produces unstable
# numerics when the oracle disagrees about scale with the predictor's family.
_KIND_TO_METRIC = {
    'logit':    'RE',
    'rf':       'abs_err',
    'gp':       'abs_err',
    'ensemble': 'abs_err',
}


def metric_key_for(kind):
    if kind not in _KIND_TO_METRIC:
        raise ValueError(f"unknown oracle kind: {kind!r}")
    return _KIND_TO_METRIC[kind]


def build_oracle(X, y, kind='logit', random_state=42):
    """Build an oracle of the requested kind.

    kind='logit'    → TUNE-aligned BMA(Logit) on validation. Metric: RE.
    kind='rf'       → calibrated Random Forest. Metric: abs_err.
    kind='gp'       → Gaussian Process classifier. Metric: abs_err.
    kind='ensemble' → mean of {logit, rf, gp}. Metric: abs_err.
    """
    factory = {
        'logit':    _LogitOracle,
        'rf':       _RFOracle,
        'gp':       _GPOracle,
        'ensemble': _EnsembleOracle,
    }
    if kind not in factory:
        raise ValueError(f"unknown oracle kind: {kind!r}")
    return factory[kind](X, y, random_state=random_state)


def oracle_predict(oracle, row_with_const_or_raw):
    """Predict p(hazard=1) at one config. Strips a 'const' column if present."""
    cols = [c for c in row_with_const_or_raw.columns if c != 'const']
    raw = row_with_const_or_raw[cols]
    return oracle.predict(raw)


def compute_metrics(pred, oracle_val, oracle_kind):
    """Compact RQ2 metric panel — fields depend on oracle kind.

    Common:
        pred, oracle, success
        success: TUNE-aligned dual-agreement criterion. TUNE uses
        `prediction > 0.5 AND pred_oracle > 0.5` (Y=1 = "satisfied"). Our
        codebase has the inverse polarity (Y=1 = "hazard"), so the
        polarity-flipped equivalent is `pred < 0.5 AND oracle < 0.5` —
        BOTH the model and the oracle must judge the new config safe.
        success_rate (= mean(success)) is the "precision" used in RQ2 plots.

    rf oracle:
        abs_err = |pred - oracle|  (cross-family RF baseline; sensitivity check)

    logit oracle (TUNE-aligned, bma_adaptation.py:331):
        RE = |pred - oracle| / oracle   (verbatim TUNE; no epsilon, no clip)
        When oracle == 0 exactly, RE is set to inf to surface the pathology
        rather than mask it.
    """
    common = {
        'pred':    float(pred),
        'oracle':  float(oracle_val),
        'success': bool(pred < 0.5 and oracle_val < 0.5),
    }
    metric = metric_key_for(oracle_kind)
    if metric == 'RE':
        if oracle_val == 0.0:
            common['RE'] = float('inf')
        else:
            common['RE'] = float(abs(pred - oracle_val) / oracle_val)
    else:  # abs_err for rf, gp, ensemble
        common['abs_err'] = float(abs(pred - oracle_val))
    return common

def cache_dir(base_dir):
    d = os.path.join(base_dir, 'cache')
    os.makedirs(d, exist_ok=True)
    return d

def threshold_tag(th):
    """Canonical filename fragment for a drift threshold, e.g. 0.015 -> 'th0.015'."""
    return f"th{float(th):.3f}"


def oracle_tag(kind):
    """Canonical filename fragment for an oracle kind, e.g. 'logit' -> 'oraclelogit'."""
    return f"oracle{kind}"

def build_data_splits(base_dir):
    """Setting A: full training set as warmup, full validation set as stream."""
    df_train = load_and_encode(os.path.join(base_dir, 'data', 'training_rescueRobot_450.csv'))
    df_valid = load_and_encode(os.path.join(base_dir, 'data', 'validation_rescueRobot_450.csv'))
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL].values
    X_valid = df_valid[FEATURE_COLS]
    y_valid = df_valid[TARGET_COL].values

    X_warmup = X_train.reset_index(drop=True)
    y_warmup = y_train
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
        online_lr=args.online_lr, mini_batch_size=args.mini_batch,
        alpha=args.alpha)
    stacking.fit(X_warmup, y_warmup)
    stacking_offline = copy.deepcopy(stacking)

    # Identify hazard=1 trigger samples (paper: adaptation fires when hazard detected)
    hazard0_indices = [t for t in range(len(y_stream)) if y_stream[t] == 1]
    if args.limit is not None:
        hazard0_indices = hazard0_indices[:args.limit]
    print(f"Hazard=1 trigger samples: {len(hazard0_indices)}")

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

    print(f"Building {args.oracle} oracle...")
    oracle = build_oracle(data['X_valid'], data['y_valid'],
                          kind=args.oracle, random_state=args.seed)

    # Chunk selection
    total = len(hazard0_indices)
    chunk_size = (total + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    my_indices = hazard0_indices[start:end]

    print(f"=== RQ2 online_ga A [{tag}] chunk {args.chunk}/{args.n_chunks}: "
          f"samples {start}-{end-1} ({len(my_indices)} GA runs, {args.ga_workers} workers) ===")
    t0 = time.time()
    results = {}

    samples = [(t, hazard0_rows[t], *snapshots[t]) for t in my_indices]
    batch = run_batch_ga_snapshots(model, samples, args.ga_workers)
    for t in my_indices:
        pred, _pred_initial, new_row = batch[t]
        oracle_val = oracle_predict(oracle, new_row)
        m = compute_metrics(pred, oracle_val, args.oracle)
        results[str(t)] = {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m.items()}

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s")

    otag = oracle_tag(args.oracle)
    out = os.path.join(cd, f'rq2_A_online_{tag}_{otag}_chunk{args.chunk}.json')
    with open(out, 'w') as f:
        json.dump({'results': results, 'elapsed': round(elapsed, 2),
                   'chunk': args.chunk, 'n_chunks': args.n_chunks,
                   'drift_threshold': args.drift_threshold,
                   'oracle': args.oracle}, f)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Phase: offline / bma  (chunked, embarrassingly parallel)
# ---------------------------------------------------------------------------

def phase_batch(args, method_key):
    """Run GA for a chunk of hazard=0 samples using a frozen model.

    Threshold-independent: offline Stacking does not perform online updates,
    so output is shared across the threshold sweep.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    model = data['stacking_offline']

    print(f"Building {args.oracle} oracle...")
    oracle = build_oracle(data['X_valid'], data['y_valid'],
                          kind=args.oracle, random_state=args.seed)
    hazard0_indices = data['hazard0_indices']
    hazard0_rows = data['hazard0_rows']

    # Chunk selection
    total = len(hazard0_indices)
    chunk_size = (total + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    my_indices = hazard0_indices[start:end]

    print(f"=== RQ2 {method_key} A chunk {args.chunk}/{args.n_chunks}: "
          f"samples {start}-{end-1} ({len(my_indices)} GA runs, {args.ga_workers} workers) ===")
    t0 = time.time()
    results = {}

    h0_list = [(t, hazard0_rows[t]) for t in my_indices]
    batch = run_batch_ga(model, h0_list, args.ga_workers)
    for t in my_indices:
        pred, _pred_initial, new_row = batch[t]
        oracle_val = oracle_predict(oracle, new_row)
        m = compute_metrics(pred, oracle_val, args.oracle)
        results[str(t)] = {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m.items()}

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s")

    otag = oracle_tag(args.oracle)
    out = os.path.join(cd, f'rq2_A_{method_key}_{otag}_chunk{args.chunk}.json')
    with open(out, 'w') as f:
        json.dump({'results': results, 'elapsed': round(elapsed, 2),
                   'chunk': args.chunk, 'n_chunks': args.n_chunks,
                   'oracle': args.oracle}, f)
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

    def _load_chunks(prefix):
        out = {}
        for fname in sorted(os.listdir(cd)):
            if fname.startswith(prefix) and fname.endswith('.json'):
                with open(os.path.join(cd, fname)) as f:
                    chunk = json.load(f)
                out.update(chunk['results'])
        return out

    otag = oracle_tag(args.oracle)
    online_results     = _load_chunks(f'rq2_A_online_{tag}_{otag}_chunk')
    bma_online_results = _load_chunks(f'rq2_A_bma_online_{tag}_{otag}_chunk')
    best_logit_results = _load_chunks(f'rq2_A_best_logit_{otag}_chunk')

    diag_path = os.path.join(cd, f'rq2_A_online_diag_{tag}.json')
    online_diag = {}
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            online_diag = json.load(f).get('diagnostics', {})

    print(f"=== RQ2 Merge A [{tag}]: online={len(online_results)}, "
          f"bma_online={len(bma_online_results)}, "
          f"best_logit={len(best_logit_results)} ===")

    # Detect oracle kind from any populated chunk (fall back to args).
    def _detect_oracle(results):
        for v in results.values():
            if 'RE' in v:
                return 'logit'
            if 'abs_err' in v:
                return 'rf'
        return None

    oracle_kind = (_detect_oracle(online_results)
                   or _detect_oracle(bma_online_results)
                   or _detect_oracle(best_logit_results)
                   or args.oracle)
    err_key = 'RE' if oracle_kind == 'logit' else 'abs_err'

    methods = {}
    for name, data in [('Stacking-Online', online_results),
                       ('BMA',             bma_online_results),
                       ('Best-Logit',      best_logit_results)]:
        if not data:
            continue
        per_sample = [{'t': int(t), **v}
                      for t, v in sorted(data.items(), key=lambda x: int(x[0]))]

        def _vec(key, _data=data):
            return [v[key] for v in _data.values() if key in v]

        succ_vals = _vec('success')
        err_vals  = _vec(err_key)

        def _stat(vals, fn):
            return float(fn(vals)) if vals else float('nan')

        methods[name] = {
            'precision':        round(_stat(succ_vals, np.mean), 6),
            f'mean_{err_key}':   round(_stat(err_vals, np.mean), 6),
            f'median_{err_key}': round(_stat(err_vals, np.median), 6),
            'n_samples':        len(data),
            'per_sample':       per_sample,
        }
        print(f"  {name:<20s}  precision={methods[name]['precision']:.3f}  "
              f"{err_key}(mean)={methods[name][f'mean_{err_key}']:.4f}")

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    output = {
        'setting': 'A', 'drift_threshold': args.drift_threshold,
        'warmup_size': warmup_size,
        'stream_size': stream_size, 'hazard0_count': hazard0_count,
        'oracle': oracle_kind,
        'metric_key': err_key,
        'methods': methods, 'diagnostics': online_diag}

    json_path = os.path.join(log_dir, f'rq2_A_{tag}_{otag}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {json_path}")

    tsv_path = os.path.join(log_dir, f'rq2_A_{tag}_{otag}_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write(f'method\tprecision\tmean_{err_key}\tmedian_{err_key}\tn_samples\n')
        for m, r in methods.items():
            f.write(f"{m}\t{r['precision']}\t"
                    f"{r[f'mean_{err_key}']}\t{r[f'median_{err_key}']}\t"
                    f"{r['n_samples']}\n")
    print(f"Saved: {tsv_path}")


# ---------------------------------------------------------------------------
# Phase: bma_online  (BMA prequential loop — threshold-independent, run once)
# ---------------------------------------------------------------------------

def phase_bma_online(args):
    """Run BMA prequential loop; save coefficient snapshots at hazard=0 points.

    Threshold-independent: BMA uses no drift detection. Run once and reuse
    across all threshold sweeps via phase_bma_online_ga.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)

    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    X_stream = data['X_stream']
    y_stream = data['y_stream']
    hazard0_set = set(data['hazard0_indices'])

    bma_snap_path = os.path.join(cd, f'rq2_A_bma_snapshots_re{args.retrain_every}.pkl')

    print(f"=== RQ2 BMA-Online A (retrain_every={args.retrain_every}): "
          f"{len(y_stream)} samples, {len(hazard0_set)} snapshots ===")
    t0 = time.time()

    print("Training BMA on warmup...")
    bma_model = BMA(data['y_warmup'], data['X_warmup'],
                    RegType='Logit', Verbose=False,
                    retrain_every=args.retrain_every).fit()

    bma_snapshots = {}
    for t in range(len(y_stream)):
        x_raw = X_stream.iloc[t:t + 1]
        if t in hazard0_set:
            # Save BMA coefficients BEFORE this sample's update
            bma_snapshots[t] = bma_model.coefficients.copy()
        bma_model.online_update(x_raw, float(y_stream[t]))

    elapsed = time.time() - t0
    retrains = bma_model._stream_count // bma_model.retrain_every
    print(f"Done: {elapsed:.1f}s, snapshots={len(bma_snapshots)}, retrains={retrains}")

    with open(bma_snap_path, 'wb') as f:
        pickle.dump(bma_snapshots, f)
    print(f"Saved: {bma_snap_path}")


# ---------------------------------------------------------------------------
# Phase: bma_online_ga  (chunked parallel GA using BMA coefficient snapshots)
# ---------------------------------------------------------------------------

def phase_bma_online_ga(args):
    """Run GA for a chunk of hazard=0 samples using BMA coefficient snapshots.

    Threshold-independent: restores per-timestep BMA state from snapshots
    produced by phase_bma_online, then runs GA exactly as phase_online_ga
    does for Stacking-Online.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    tag = threshold_tag(args.drift_threshold)

    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    bma_snap_path = os.path.join(cd, f'rq2_A_bma_snapshots_re{args.retrain_every}.pkl')
    with open(bma_snap_path, 'rb') as f:
        bma_snapshots = pickle.load(f)

    # Base BMA model supplies the warmup-fitted scaler; coefficients overridden per snapshot
    print("Training base BMA (warmup scaler)...")
    bma_model = BMA(data['y_warmup'], data['X_warmup'],
                    RegType='Logit', Verbose=False).fit()
    bma_predictor = BMAPredictor(bma_model)

    print(f"Building {args.oracle} oracle...")
    oracle = build_oracle(data['X_valid'], data['y_valid'],
                          kind=args.oracle, random_state=args.seed)

    hazard0_indices = data['hazard0_indices']
    hazard0_rows = data['hazard0_rows']

    total = len(hazard0_indices)
    chunk_size = (total + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    my_indices = hazard0_indices[start:end]

    print(f"=== RQ2 bma_online_ga A [{tag}] chunk {args.chunk}/{args.n_chunks}: "
          f"samples {start}-{end - 1} ({len(my_indices)} GA runs, {args.ga_workers} workers) ===")
    t0 = time.time()
    results = {}

    fallback_coef = bma_model.coefficients
    samples = [
        (t, hazard0_rows[t], bma_snapshots.get(t, fallback_coef))
        for t in my_indices
    ]
    batch = run_batch_ga_bma_snapshots(bma_predictor, samples, args.ga_workers)
    for t in my_indices:
        pred, _pred_initial, new_row = batch[t]
        oracle_val = oracle_predict(oracle, new_row)
        m = compute_metrics(pred, oracle_val, args.oracle)
        results[str(t)] = {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m.items()}

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s")

    otag = oracle_tag(args.oracle)
    out = os.path.join(cd, f'rq2_A_bma_online_{tag}_{otag}_chunk{args.chunk}.json')
    with open(out, 'w') as f:
        json.dump({'results': results, 'elapsed': round(elapsed, 2),
                   'chunk': args.chunk, 'n_chunks': args.n_chunks,
                   'drift_threshold': args.drift_threshold,
                   'oracle': args.oracle}, f)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Phase: best_logit  (TUNE-aligned MS baseline; threshold-independent)
# ---------------------------------------------------------------------------

def phase_best_logit(args):
    """Run GA for a chunk of hazard=0 samples using a frozen Best-Logit (MS)
    model trained on the warmup set. Threshold-independent: Best-Logit does
    not perform online updates."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cd = cache_dir(base_dir)
    pkl_path = os.path.join(cd, 'rq2_A_setup.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print("Selecting Best-Logit (MS) on warmup...")
    ms_result = find_best_logit_ms(data['X_warmup'], data['y_warmup'],
                                   random_state=args.seed)
    bl_predictor = BestLogitPredictor(ms_result)

    print(f"Building {args.oracle} oracle...")
    oracle = build_oracle(data['X_valid'], data['y_valid'],
                          kind=args.oracle, random_state=args.seed)

    hazard0_indices = data['hazard0_indices']
    hazard0_rows = data['hazard0_rows']

    total = len(hazard0_indices)
    chunk_size = (total + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    my_indices = hazard0_indices[start:end]

    print(f"=== RQ2 best_logit A chunk {args.chunk}/{args.n_chunks}: "
          f"samples {start}-{end-1} ({len(my_indices)} GA runs, {args.ga_workers} workers) ===")
    t0 = time.time()
    results = {}

    h0_list = [(t, hazard0_rows[t]) for t in my_indices]
    batch = run_batch_ga(bl_predictor, h0_list, args.ga_workers)
    for t in my_indices:
        pred, _pred_initial, new_row = batch[t]
        oracle_val = oracle_predict(oracle, new_row)
        m = compute_metrics(pred, oracle_val, args.oracle)
        results[str(t)] = {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m.items()}

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s")

    otag = oracle_tag(args.oracle)
    out = os.path.join(cd, f'rq2_A_best_logit_{otag}_chunk{args.chunk}.json')
    with open(out, 'w') as f:
        json.dump({'results': results, 'elapsed': round(elapsed, 2),
                   'chunk': args.chunk, 'n_chunks': args.n_chunks,
                   'oracle': args.oracle}, f)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Single-node mode (runs all phases sequentially, with local parallelism)
# ---------------------------------------------------------------------------

def run_single_node(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tag = threshold_tag(args.drift_threshold)
    t_start = time.time()
    err_key = 'RE' if args.oracle == 'logit' else 'abs_err'

    X_warmup, y_warmup, X_stream, y_stream, X_valid, y_valid = \
        build_data_splits(base_dir)

    print(f"=== RQ2 Setting A [{tag}] (single-node, oracle={args.oracle}) ===")
    print(f"Warmup: {len(y_warmup)}, Stream: {len(y_stream)}, "
          f"GA workers: {args.ga_workers}, retrain_every={args.retrain_every}")

    # Train models
    stacking_online = StackingEnsemble(
        n_folds=10, random_state=args.seed,
        online_lr=args.online_lr, mini_batch_size=args.mini_batch,
        alpha=args.alpha,
        drift_threshold=args.drift_threshold)
    stacking_online.fit(X_warmup, y_warmup)

    # BMA: base-paper implementation with MAPE-K updates in the prequential loop
    bma_online = BMA(y_warmup, X_warmup, RegType='Logit', Verbose=False,
                     retrain_every=args.retrain_every).fit()
    bma_online_predictor = BMAPredictor(bma_online)

    # Best-Logit (MS) on warmup — frozen TUNE-style baseline.
    print("Selecting Best-Logit (MS) on warmup...")
    ms_result = find_best_logit_ms(X_warmup, y_warmup, random_state=args.seed)
    bl_predictor = BestLogitPredictor(ms_result)

    print(f"Building {args.oracle} oracle...")
    oracle = build_oracle(X_valid, y_valid, kind=args.oracle, random_state=args.seed)

    # Trigger adaptation on hazard=1 samples (paper: hazard detected -> plan)
    hazard0_indices = [t for t in range(len(y_stream)) if y_stream[t] == 1]
    if args.limit:
        hazard0_indices = hazard0_indices[:args.limit]
    hazard0_set = set(hazard0_indices)
    hazard0_rows = {t: add_constant(X_stream.iloc[t:t+1], has_constant='add')
                    for t in hazard0_indices}
    print(f"Hazard=1 triggers: {len(hazard0_indices)}")

    # Online pass 1: sequential prequential loop, collect snapshots (no GA)
    print(f"\n--- Online prequential pass ({len(hazard0_indices)} trigger points) ---")
    t1 = time.time()
    stacking_snaps = {}
    bma_snaps = {}
    for t in range(len(y_stream)):
        x_raw = X_stream.iloc[t:t+1]
        if t in hazard0_set:
            stacking_snaps[t] = (
                stacking_online.meta_learner.coef_.copy(),
                stacking_online.meta_learner.intercept_.copy(),
            )
            bma_snaps[t] = bma_online.coefficients.copy()
        stacking_online.online_update(x_raw, float(y_stream[t]))
        bma_online.online_update(x_raw, float(y_stream[t]))
    print(f"  Pass done: {time.time()-t1:.1f}s")

    # Online pass 2: parallel GA using per-timestep snapshots
    print(f"\n--- Online GA batch ({len(hazard0_indices)} x 2 runs, {args.ga_workers} workers) ---")
    stacking_samples = [(t, hazard0_rows[t], *stacking_snaps[t]) for t in hazard0_indices]
    bma_samples = [(t, hazard0_rows[t], bma_snaps[t]) for t in hazard0_indices]
    stacking_batch = run_batch_ga_snapshots(stacking_online, stacking_samples, args.ga_workers)
    bma_batch = run_batch_ga_bma_snapshots(bma_online_predictor, bma_samples, args.ga_workers)

    def _row_to_metrics(pred, new_row):
        ov = oracle_predict(oracle, new_row)
        m = compute_metrics(pred, ov, args.oracle)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}

    stacking_online_results = {}
    bma_online_results = {}
    for t in hazard0_indices:
        pred_s, _pred_si, new_row_s = stacking_batch[t]
        stacking_online_results[str(t)] = _row_to_metrics(pred_s, new_row_s)
        pred_b, _pred_bi, new_row_b = bma_batch[t]
        bma_online_results[str(t)] = _row_to_metrics(pred_b, new_row_b)
    p1 = time.time() - t1
    print(f"  Done: {p1:.1f}s")

    # Best-Logit batch (parallel, frozen model — replaces Stacking-Offline path)
    print(f"\n--- Best-Logit batch ({args.ga_workers} workers) ---")
    t2 = time.time()
    h0_list = [(t, hazard0_rows[t]) for t in hazard0_indices]
    bl_preds = run_batch_ga(bl_predictor, h0_list, args.ga_workers)
    best_logit_results = {}
    for t in hazard0_indices:
        pred, _pred_init, new_row = bl_preds[t]
        best_logit_results[str(t)] = _row_to_metrics(pred, new_row)
    p2 = time.time() - t2
    print(f"  Done: {p2:.1f}s")

    methods = {}
    for name, data_dict in [
        ('Stacking-Online', stacking_online_results),
        ('BMA',             bma_online_results),
        ('Best-Logit',      best_logit_results),
    ]:
        per_sample = [{'t': int(k), **v}
                      for k, v in sorted(data_dict.items(), key=lambda x: int(x[0]))]

        def _vec(key, _data=data_dict):
            return [v[key] for v in _data.values() if key in v]

        def _stat(vals, fn):
            return float(fn(vals)) if vals else float('nan')

        methods[name] = {
            'precision':         round(_stat(_vec('success'), np.mean), 6),
            f'mean_{err_key}':   round(_stat(_vec(err_key), np.mean), 6),
            f'median_{err_key}': round(_stat(_vec(err_key), np.median), 6),
            'n_samples':         len(data_dict),
            'per_sample':        per_sample,
        }
        print(f"  {name:<20s}  precision={methods[name]['precision']:.3f}  "
              f"{err_key}(mean)={methods[name][f'mean_{err_key}']:.4f}")

    elapsed = time.time() - t_start
    diag = stacking_online.get_diagnostics()
    bma_retrains = bma_online._stream_count // bma_online.retrain_every
    print(f"\nTotal: {elapsed:.1f}s  (online={p1:.0f}s best_logit={p2:.0f}s)")
    print(f"Stacking drifts={diag['drift_count']}  BMA retrains={bma_retrains}")

    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    output = {
        'setting': 'A', 'drift_threshold': args.drift_threshold,
        'retrain_every': args.retrain_every,
        'oracle': args.oracle,
        'metric_key': err_key,
        'warmup_size': len(y_warmup),
        'stream_size': len(y_stream), 'hazard0_count': len(hazard0_indices),
        'elapsed_seconds': round(elapsed, 2),
        'phase_times': {'online': round(p1, 2), 'best_logit': round(p2, 2)},
        'methods': methods,
        'diagnostics': {'drift_count': diag['drift_count'],
                        'update_count': diag['update_count'],
                        'bma_retrains': bma_retrains}}
    otag = oracle_tag(args.oracle)
    with open(os.path.join(log_dir, f'rq2_A_{tag}_{otag}.json'), 'w') as f:
        json.dump(output, f, indent=2)
    tsv_path = os.path.join(log_dir, f'rq2_A_{tag}_{otag}_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write(f'method\tprecision\tmean_{err_key}\tmedian_{err_key}\tn_samples\n')
        for m, r in methods.items():
            f.write(f"{m}\t{r['precision']}\t"
                    f"{r[f'mean_{err_key}']}\t{r[f'median_{err_key}']}\t"
                    f"{r['n_samples']}\n")
    print(f"Saved: logs/rq2_A_{tag}_{otag}.json, logs/rq2_A_{tag}_{otag}_summary.tsv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RQ2: Adaptation decision quality (Setting A)')
    parser.add_argument('--setting', choices=['A'], default='A',
                        help='Only Setting A is supported; flag kept for compatibility.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mini-batch', type=int, default=1)
    parser.add_argument('--drift-threshold', type=float, default=0.15,
                        help='Windowed mean-shift drift threshold (sweep this in parallel).')
    parser.add_argument('--online-lr', type=float, default=0.05,
                        help='SGD eta0 (online learning rate; Pareto-optimal default)')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='SGD L2 regularization strength')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--retrain-every', type=int, default=1,
                        help='BMA re-fit interval (1 = TAAS2024 paper-strict eq.5)')
    parser.add_argument('--tag-suffix', type=str, default='',
                        help='Optional suffix for ablation runs (appended to threshold tag)')
    parser.add_argument('--oracle',
                        choices=['logit', 'rf', 'gp', 'ensemble'],
                        default='logit',
                        help='Oracle for RQ2 metrics. '
                             'logit (default, TUNE-aligned BMA(Logit) on validation; metric=RE). '
                             'rf (calibrated Random Forest, cross-family sensitivity; metric=abs_err). '
                             'gp (Gaussian Process w/ RBF kernel, calibrated cross-family; metric=abs_err). '
                             'ensemble (mean of {logit, rf, gp}, cross-family robust; metric=abs_err).')
    # Single-node mode
    parser.add_argument('--ga-workers', type=int, default=4,
                        help='Parallel GA workers (single-node and all distributed GA phases).')
    # Distributed mode
    parser.add_argument('--phase',
                        choices=['setup', 'online', 'online_ga', 'offline',
                                 'bma_online', 'bma_online_ga', 'best_logit',
                                 'merge'],
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
    elif args.phase == 'offline':
        phase_batch(args, 'offline')
    elif args.phase == 'bma_online':
        phase_bma_online(args)
    elif args.phase == 'bma_online_ga':
        phase_bma_online_ga(args)
    elif args.phase == 'best_logit':
        phase_best_logit(args)
    elif args.phase == 'merge':
        phase_merge(args)


if __name__ == '__main__':
    mp.freeze_support()
    main()
