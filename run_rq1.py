#!/usr/bin/env python3
"""
run_rq1.py
RQ1: Prediction accuracy comparison -- Stacking-Online vs Stacking-Offline vs BMA vs Best-Logit.

Setting A only: warmup = full training set (450), stream = full validation set (462).
Uses PREQUENTIAL (test-then-train) evaluation.

Usage:
    python run_rq1.py --seed 42
    python run_rq1.py --seed 42 --drift-threshold 0.10
"""

import copy
import json
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.tools import add_constant

from stacking import StackingEnsemble
from bma import BMA

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = ['illuminance', 'smoke', 'size', 'distance', 'firm',
                'power', 'band', 'quality', 'speed']
TARGET_COL = 'hazard'


def load_and_encode(path):
    """Load CSV and encode the 'firm' column to binary."""
    df = pd.read_csv(path)
    df['firm'] = (df['firm'] == 'Yes') * 1
    return df


def cumulative_metrics(y_true, y_pred, threshold=0.5):
    """Compute cumulative precision, recall, F1, and Brier score."""
    labels = (np.array(y_pred) >= threshold).astype(int)
    true = np.array(y_true, dtype=int)
    prec = precision_score(true, labels, zero_division=0)
    rec = recall_score(true, labels, zero_division=0)
    f1 = f1_score(true, labels, zero_division=0)
    brier = float(np.mean((np.array(y_pred) - true) ** 2))
    return prec, rec, f1, brier


def sliding_window_f1(y_true, y_pred, window=50, threshold=0.5):
    """Compute F1 over the most recent `window` samples."""
    if len(y_true) < 1:
        return 0.0
    start = max(0, len(y_true) - window)
    labels = (np.array(y_pred[start:]) >= threshold).astype(int)
    true = np.array(y_true[start:], dtype=int)
    return float(f1_score(true, labels, zero_division=0))


def find_best_logit(stacking_model, X_warmup_with_const, y_warmup):
    """
    Find the single best logistic regression base model by F1 on warmup data.
    Returns (index_set, model) tuple.
    """
    best_f1 = -1.0
    best_entry = None
    y_arr = np.array(y_warmup, dtype=int)

    for idx_set, model in stacking_model.base_models:
        if model is None:
            continue
        try:
            preds = model.predict(X_warmup_with_const.iloc[:, list(idx_set)])
            labels = (np.array(preds) >= 0.5).astype(int)
            score = f1_score(y_arr, labels, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_entry = (idx_set, model)
        except Exception:
            continue

    if best_entry is None:
        # Fallback: use first valid model
        for idx_set, model in stacking_model.base_models:
            if model is not None:
                best_entry = (idx_set, model)
                break
    return best_entry


def predict_logit(model_entry, row_with_const):
    """Predict probability with a single logistic regression model."""
    idx_set, model = model_entry
    try:
        pred = model.predict(row_with_const.iloc[:, list(idx_set)])
        return float(pred.values[0] if hasattr(pred, 'values') else pred[0])
    except Exception:
        return 0.5


def predict_bma(bma_model, row_with_const):
    """Predict probability with BMA, handling scalar/array returns."""
    try:
        result = bma_model.predict(np.asarray(row_with_const))
        if hasattr(result, '__len__'):
            return float(np.ravel(result)[0])
        return float(result)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RQ1: Prediction accuracy comparison (Setting A)')
    parser.add_argument('--setting', choices=['A'], default='A',
                        help='Only Setting A is supported; flag kept for compatibility.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mini-batch', type=int, default=5,
                        help='Mini-batch size for StackingEnsemble')
    parser.add_argument('--drift-threshold', type=float, default=0.15,
                        help='ADWIN drift threshold for online warm restart.')
    parser.add_argument('--online-lr', type=float, default=0.001,
                        help='Online learning rate')
    args = parser.parse_args()

    np.random.seed(args.seed)
    t_start = time.time()
    tag = f"th{float(args.drift_threshold):.3f}"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_train = load_and_encode(os.path.join(base_dir, 'data', 'training_rescueRobot_450.csv'))
    df_valid = load_and_encode(os.path.join(base_dir, 'data', 'validation_rescueRobot_450.csv'))

    X_train_all = df_train[FEATURE_COLS]
    y_train_all = df_train[TARGET_COL].values

    X_valid_all = df_valid[FEATURE_COLS]
    y_valid_all = df_valid[TARGET_COL].values

    # ------------------------------------------------------------------
    # 2. Setting A: warmup = 450 training, stream = 462 validation
    # ------------------------------------------------------------------
    X_warmup = X_train_all.iloc[:450].reset_index(drop=True)
    y_warmup = y_train_all[:450]
    X_stream = X_valid_all.reset_index(drop=True)
    y_stream = y_valid_all

    warmup_size = len(y_warmup)
    stream_size = len(y_stream)
    print(f"=== RQ1 Setting A [{tag}] ===")
    print(f"Warmup: {warmup_size} samples, Stream: {stream_size} samples")
    print(f"Config: mini_batch={args.mini_batch}, drift_threshold={args.drift_threshold}, "
          f"lr={args.online_lr}, seed={args.seed}")

    # ------------------------------------------------------------------
    # 3. Train Stacking on warmup
    # ------------------------------------------------------------------
    print("\n--- Training Stacking Ensemble ---")
    stacking_online = StackingEnsemble(
        n_folds=10,
        random_state=args.seed,
        online_lr=args.online_lr,
        mini_batch_size=args.mini_batch,
        drift_threshold=args.drift_threshold,
    )
    stacking_online.fit(X_warmup, y_warmup)

    # Offline copy: deep copy AFTER fit, BEFORE streaming
    stacking_offline = copy.deepcopy(stacking_online)

    # ------------------------------------------------------------------
    # 4. Train BMA on warmup
    # ------------------------------------------------------------------
    print("\n--- Training BMA ---")
    X_warmup_const = add_constant(X_warmup)
    bma_model = BMA(y_warmup, X_warmup_const, RegType='Logit', Verbose=False).fit()
    print(f"  BMA fitted with {len(bma_model.likelihoods_all)} models")

    # ------------------------------------------------------------------
    # 5. Find best individual logistic regression
    # ------------------------------------------------------------------
    print("\n--- Selecting Best-Logit ---")
    best_logit_entry = find_best_logit(stacking_online, add_constant(X_warmup), y_warmup)
    if best_logit_entry is not None:
        bl_idx_set, _ = best_logit_entry
        print(f"  Best-Logit uses features: {bl_idx_set}")
    else:
        print("  WARNING: No valid logistic model found; Best-Logit will return 0.5")

    # ------------------------------------------------------------------
    # 6. Prequential loop
    # ------------------------------------------------------------------
    print(f"\n--- Prequential Evaluation ({stream_size} samples) ---")

    methods = ['Stacking-Online', 'Stacking-Offline', 'BMA', 'Best-Logit']
    preds = {m: [] for m in methods}
    trues = []

    per_sample = {m: [] for m in methods}

    for t in range(stream_size):
        x_t_raw = X_stream.iloc[t:t + 1]
        y_t = float(y_stream[t])
        x_t_with_const = add_constant(x_t_raw, has_constant='add')

        # ---- Predict BEFORE update ----
        pred_online = stacking_online.predict_single(x_t_with_const)
        pred_offline = stacking_offline.predict_single(x_t_with_const)
        pred_bma_val = predict_bma(bma_model, x_t_with_const)
        pred_logit_val = (predict_logit(best_logit_entry, x_t_with_const)
                          if best_logit_entry is not None else 0.5)

        # ---- Record ----
        trues.append(int(y_t))
        preds['Stacking-Online'].append(float(pred_online))
        preds['Stacking-Offline'].append(float(pred_offline))
        preds['BMA'].append(float(pred_bma_val))
        preds['Best-Logit'].append(float(pred_logit_val))

        # Per-sample metrics (cumulative and sliding-window)
        for m in methods:
            p, r, f1_c, br = cumulative_metrics(trues, preds[m])
            f1_w = sliding_window_f1(trues, preds[m], window=50)
            per_sample[m].append({
                't': t,
                'pred': round(preds[m][-1], 6),
                'true': int(y_t),
                'f1_cum': round(f1_c, 6),
                'f1_window': round(f1_w, 6),
                'brier_cum': round(br, 6),
            })

        # ---- Knowledge update: only online model ----
        stacking_online.online_update(x_t_raw, y_t)

        # Progress
        if (t + 1) % 50 == 0 or t == stream_size - 1:
            _, _, f1_on, br_on = cumulative_metrics(trues, preds['Stacking-Online'])
            _, _, f1_off, br_off = cumulative_metrics(trues, preds['Stacking-Offline'])
            _, _, f1_bma, br_bma = cumulative_metrics(trues, preds['BMA'])
            _, _, f1_bl, br_bl = cumulative_metrics(trues, preds['Best-Logit'])
            print(f"  t={t + 1:>4d}/{stream_size}  "
                  f"F1: On={f1_on:.3f} Off={f1_off:.3f} BMA={f1_bma:.3f} BL={f1_bl:.3f}  "
                  f"Brier: On={br_on:.3f} Off={br_off:.3f} BMA={br_bma:.3f} BL={br_bl:.3f}")

    # ------------------------------------------------------------------
    # 7. Final metrics
    # ------------------------------------------------------------------
    print("\n--- Final Results ---")
    results_methods = {}
    for m in methods:
        prec, rec, f1_val, brier = cumulative_metrics(trues, preds[m])
        results_methods[m] = {
            'precision': round(prec, 6),
            'recall': round(rec, 6),
            'f1': round(f1_val, 6),
            'brier': round(brier, 6),
            'per_sample': per_sample[m],
        }
        print(f"  {m:<20s}  P={prec:.4f}  R={rec:.4f}  F1={f1_val:.4f}  Brier={brier:.4f}")

    # Diagnostics from online model
    diag = stacking_online.get_diagnostics()
    diagnostics = {
        'drift_count': diag['drift_count'],
        'update_count': diag['update_count'],
        'samples_seen': diag['samples_seen'],
        'buffer_size': diag['buffer_size'],
        'pending_mini_batch': diag['pending_mini_batch'],
    }

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s  |  Drifts: {diagnostics['drift_count']}  "
          f"|  Updates: {diagnostics['update_count']}")

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

    output_json = {
        'setting': 'A',
        'warmup_size': warmup_size,
        'stream_size': stream_size,
        'seed': args.seed,
        'mini_batch': args.mini_batch,
        'drift_threshold': args.drift_threshold,
        'online_lr': args.online_lr,
        'elapsed_seconds': round(elapsed, 2),
        'methods': results_methods,
        'diagnostics': diagnostics,
    }

    json_path = os.path.join(base_dir, 'logs', f'rq1_A_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved: {json_path}")

    # TSV summary
    tsv_path = os.path.join(base_dir, 'logs', f'rq1_A_{tag}_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write('method\tprecision\trecall\tF1\tbrier\n')
        for m in methods:
            r = results_methods[m]
            f.write(f"{m}\t{r['precision']}\t{r['recall']}\t{r['f1']}\t{r['brier']}\n")
    print(f"Saved: {tsv_path}")


if __name__ == '__main__':
    main()
