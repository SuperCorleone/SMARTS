#!/usr/bin/env python3
"""
run_rq4.py
RQ4: Concept drift experiment -- same as RQ1 but with injected label flipping in the validation stream.

Usage:
    python run_rq4.py --seed 42 --drift-scenario 1x
    python run_rq4.py --seed 42 --drift-scenario 2x --drift-threshold 0.10
"""

import copy
import json
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant
import statsmodels.api as sm

from stacking import StackingEnsemble
from bma import BMA

warnings.filterwarnings('ignore')

FEATURE_COLS = ['illuminance', 'smoke', 'size', 'distance', 'firm',
                'power', 'band', 'quality', 'speed']
TARGET_COL = 'hazard'

DRIFT_SCENARIOS = {
    # ----- Transient bursts (50-sample intervals): tests detector sensitivity.
    # Online and offline often look similar at the cumulative level because
    # the stream reverts after each burst. Keep these for detector ablation.
    '1x':     [(150, 200)],
    '2x':     [(115, 165), (300, 350)],
    '3x':     [(92, 142), (231, 281), (370, 420)],
    '4x':     [(52, 102), (154, 204), (256, 306), (358, 408)],
    '5x':     [(35, 85), (120, 170), (205, 255), (290, 340), (375, 425)],
    '6x':     [(23, 73), (96, 146), (169, 219), (242, 292), (315, 365), (388, 438)],

    # ----- Permanent flips: maximal separation between online and offline.
    # The stream never reverts after the flip, so offline (frozen at warmup)
    # cannot recover; online has time to relearn. Stream length = 462.
    'perm_late':  [(225, 462)],   # flip back-half: ~half the stream is post-drift
    'perm_mid':   [(150, 462)],   # flip from 1/3 onward: longer recovery window
    'perm_early': [(75, 462)],    # flip from 1/6 onward: stress test for online

    # ----- Recurring drift: A → B → A → B → A. Tests both detection AND
    # whether online learning catastrophically forgets the original concept.
    # Each segment ~75 samples with ~75-sample stationary gaps.
    'recurring':  [(75, 150), (225, 300), (375, 450)],
    # # 10-sample intervals
    # '1x_w10': [(226, 236)],
    # '2x_w10': [(147, 157), (304, 314)],
    # '3x_w10': [(108, 118), (226, 236), (344, 354)],
    # '4x_w10': [(84, 94), (178, 188), (272, 282), (366, 376)],
    # '5x_w10': [(68, 78), (146, 156), (224, 234), (302, 312), (380, 390)],
    # '6x_w10': [(57, 67), (124, 134), (191, 201), (258, 268), (325, 335), (392, 402)],
    # # 30-sample intervals
    # '1x_w30': [(216, 246)],
    # '2x_w30': [(134, 164), (298, 328)],
    # '3x_w30': [(93, 123), (216, 246), (339, 369)],
    # '4x_w30': [(68, 98), (166, 196), (264, 294), (362, 392)],
    # '5x_w30': [(52, 82), (134, 164), (216, 246), (298, 328), (380, 410)],
    # '6x_w30': [(40, 70), (110, 140), (180, 210), (250, 280), (320, 350), (390, 420)],
}


def load_and_encode(path):
    df = pd.read_csv(path)
    df['firm'] = (df['firm'] == 'Yes') * 1
    return df


def cumulative_metrics(y_true, y_pred, threshold=0.5):
    labels = (np.array(y_pred) >= threshold).astype(int)
    true = np.array(y_true, dtype=int)
    prec = precision_score(true, labels, zero_division=0)
    rec = recall_score(true, labels, zero_division=0)
    f1 = f1_score(true, labels, zero_division=0)
    brier = float(np.mean((np.array(y_pred) - true) ** 2))
    return prec, rec, f1, brier


def sliding_window_f1(y_true, y_pred, window=50, threshold=0.5):
    if len(y_true) < 1:
        return 0.0
    start = max(0, len(y_true) - window)
    labels = (np.array(y_pred[start:]) >= threshold).astype(int)
    true = np.array(y_true[start:], dtype=int)
    return float(f1_score(true, labels, zero_division=0))


def find_best_logit_ms(X, y, random_state=42):
    """MS baseline aligned with TAAS2024 (helpers.py)."""
    y_arr = np.asarray(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=0.2, stratify=y_arr, random_state=random_state)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    y_train_s = pd.DataFrame(y_train)
    y_test_s = pd.DataFrame(y_test)

    nCols = X_train_s.shape[1]
    maxVars = nCols
    best_f1 = -1.0
    best_idx_set = None

    models_previous = []
    for num_elements in range(1, maxVars + 1):
        models_next = list(combinations(list(range(nCols)), num_elements))
        if num_elements == 1:
            models_current = models_next
            models_previous = []
        else:
            idx_keep = np.zeros(len(models_next))
            for m_new, idx in zip(models_next, range(len(models_next))):
                for m_good in models_previous:
                    if all(x in m_new for x in m_good):
                        idx_keep[idx] = 1
                        break
            models_current = np.asarray(models_next)[np.where(idx_keep == 1)].tolist()
            models_previous = []

        for model_index_set in models_current:
            model_X = X_train_s.iloc[:, list(model_index_set)]
            test_X = X_test_s.iloc[:, list(model_index_set)]
            try:
                local_model = sm.Logit(y_train_s, model_X).fit(disp=0)
                models_previous.append(model_index_set)
                y_pred = np.round(local_model.predict(test_X)).astype(int)
                score = f1_score(y_test_s, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_idx_set = model_index_set
            except Exception:
                continue

    if best_idx_set is None:
        return None

    scaler_full = StandardScaler()
    X_all_s = pd.DataFrame(scaler_full.fit_transform(X), columns=X.columns)
    y_all = pd.DataFrame(y_arr)
    best_model = sm.Logit(y_all, X_all_s.iloc[:, list(best_idx_set)]).fit(disp=0)

    return best_idx_set, best_model, scaler_full


def predict_logit_ms(ms_entry, row_raw):
    idx_set, model, scaler = ms_entry
    try:
        row_scaled = pd.DataFrame(scaler.transform(row_raw), columns=row_raw.columns)
        pred = model.predict(row_scaled.iloc[:, list(idx_set)])
        return float(pred.values[0] if hasattr(pred, 'values') else pred[0])
    except Exception:
        return 0.5


def predict_bma(bma_model, row_raw):
    try:
        result = bma_model.predict(row_raw)
        if hasattr(result, '__len__'):
            return float(np.ravel(result)[0])
        return float(result)
    except Exception:
        return 0.5


def threshold_tag(th):
    return f"th{float(th):.3f}"


def compute_segment_metrics(methods, preds, trues, drift_intervals, stream_size):
    segment_metrics = {}
    for m in methods:
        method_segments = {}
        for i, (s, e) in enumerate(drift_intervals):
            pre_start = max(0, s - 50)
            post_end = min(e + 50, stream_size)

            seg_key = f"interval_{i}"
            method_segments[seg_key] = {}

            # pre_drift
            if pre_start < s:
                y_seg = trues[pre_start:s]
                p_seg = preds[m][pre_start:s]
                if len(y_seg) > 0:
                    labs = (np.array(p_seg) >= 0.5).astype(int)
                    method_segments[seg_key]['pre_drift'] = round(float(
                        f1_score(np.array(y_seg, dtype=int), labs, zero_division=0)), 6)
                else:
                    method_segments[seg_key]['pre_drift'] = None
            else:
                method_segments[seg_key]['pre_drift'] = None

            # during_drift
            y_seg = trues[s:e]
            p_seg = preds[m][s:e]
            if len(y_seg) > 0:
                labs = (np.array(p_seg) >= 0.5).astype(int)
                method_segments[seg_key]['during_drift'] = round(float(
                    f1_score(np.array(y_seg, dtype=int), labs, zero_division=0)), 6)
            else:
                method_segments[seg_key]['during_drift'] = None

            # post_drift
            if e < post_end:
                y_seg = trues[e:post_end]
                p_seg = preds[m][e:post_end]
                if len(y_seg) > 0:
                    labs = (np.array(p_seg) >= 0.5).astype(int)
                    method_segments[seg_key]['post_drift'] = round(float(
                        f1_score(np.array(y_seg, dtype=int), labs, zero_division=0)), 6)
                else:
                    method_segments[seg_key]['post_drift'] = None
            else:
                method_segments[seg_key]['post_drift'] = None

        segment_metrics[m] = method_segments
    return segment_metrics


def main():
    parser = argparse.ArgumentParser(description='RQ4: Concept drift experiment (Setting A)')
    parser.add_argument('--setting', choices=['A'], default='A',
                        help='Only Setting A is supported; flag kept for compatibility.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mini-batch', type=int, default=1,
                        help='Mini-batch size for StackingEnsemble (Pareto-optimal default)')
    parser.add_argument('--drift-threshold', type=float, default=0.15,
                        help='Windowed mean-shift drift threshold for warm restart.')
    parser.add_argument('--online-lr', type=float, default=0.05,
                        help='SGD eta0 (online learning rate; Pareto-optimal default)')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='SGD L2 regularization strength')
    parser.add_argument('--drift-scenario', choices=list(DRIFT_SCENARIOS.keys()),
                        required=True,
                        help='Drift scenario key (e.g. 1x, 3x_w10, 6x_w30)')
    parser.add_argument('--retrain-every', type=int, default=1,
                        help='BMA re-fit interval (1 = TAAS2024 paper-strict eq.5)')
    parser.add_argument('--tag-suffix', type=str, default='',
                        help='Optional suffix appended to log filenames for ablation runs')
    args = parser.parse_args()

    np.random.seed(args.seed)
    t_start = time.time()
    tag = threshold_tag(args.drift_threshold)
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    scenario = args.drift_scenario
    drift_intervals = DRIFT_SCENARIOS[scenario]

    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_train = load_and_encode(os.path.join(base_dir, 'data', 'training_rescueRobot_450.csv'))
    df_valid = load_and_encode(os.path.join(base_dir, 'data', 'validation_rescueRobot_450.csv'))

    X_train_all = df_train[FEATURE_COLS]
    y_train_all = df_train[TARGET_COL].values

    X_valid_all = df_valid[FEATURE_COLS]
    y_valid_all = df_valid[TARGET_COL].values

    # Setting A: warmup = full training set, stream = full validation set (both 462 rows)
    X_warmup = X_train_all.reset_index(drop=True)
    y_warmup = y_train_all
    X_stream = X_valid_all.reset_index(drop=True)
    y_stream = y_valid_all

    # Inject concept drift via label flipping
    y_drifted = y_stream.copy()
    for s, e in drift_intervals:
        y_drifted[s:e] = 1 - y_drifted[s:e]

    warmup_size = len(y_warmup)
    stream_size = len(y_stream)
    print(f"=== RQ4 Setting A [{tag}] drift={scenario} ===")
    print(f"Warmup: {warmup_size} samples, Stream: {stream_size} samples")
    print(f"Drift intervals: {drift_intervals}")
    print(f"Config: mini_batch={args.mini_batch}, drift_threshold={args.drift_threshold}, "
          f"lr={args.online_lr}, alpha={args.alpha}, "
          f"retrain_every={args.retrain_every}, seed={args.seed}")

    # Train Stacking on warmup (no drift in training)
    print("\n--- Training Stacking Ensemble ---")
    stacking_online = StackingEnsemble(
        n_folds=10,
        random_state=args.seed,
        online_lr=args.online_lr,
        mini_batch_size=args.mini_batch,
        alpha=args.alpha,
        drift_threshold=args.drift_threshold,
    )
    stacking_online.fit(X_warmup, y_warmup)
    stacking_offline = copy.deepcopy(stacking_online)

    # Train BMA on warmup (TAAS2024: no add_constant, scaler inside)
    print("\n--- Training BMA ---")
    bma_model = BMA(y_warmup, X_warmup, RegType='Logit', Verbose=False,
                    retrain_every=args.retrain_every).fit()
    print(f"  BMA fitted with {len(bma_model.likelihoods_all)} models")

    # Model Selection baseline (TAAS2024: StandardScaler + held-out F1)
    print("\n--- Selecting Best-Logit (MS) ---")
    ms_result = find_best_logit_ms(X_warmup, y_warmup, random_state=args.seed)
    if ms_result is not None:
        bl_idx_set, _, _ = ms_result
        bl_feature_names = [FEATURE_COLS[i] for i in bl_idx_set]
        print(f"  Best-Logit uses features: {bl_idx_set} {bl_feature_names}")
    else:
        print("  WARNING: No valid logistic model found; Best-Logit will return 0.5")

    # Prequential loop (uses y_drifted for evaluation and online update)
    print(f"\n--- Prequential Evaluation ({stream_size} samples) ---")

    methods = ['Stacking-Online', 'Stacking-Offline', 'BMA', 'Best-Logit']
    preds = {m: [] for m in methods}
    trues = []
    per_sample = {m: [] for m in methods}

    for t in range(stream_size):
        x_t_raw = X_stream.iloc[t:t + 1]
        y_t = float(y_drifted[t])
        x_t_with_const = add_constant(x_t_raw, has_constant='add')

        # Predict BEFORE update
        pred_online = stacking_online.predict_single(x_t_with_const)
        pred_offline = stacking_offline.predict_single(x_t_with_const)
        pred_bma_val = predict_bma(bma_model, x_t_raw)
        pred_logit_val = (predict_logit_ms(ms_result, x_t_raw)
                          if ms_result is not None else 0.5)

        # Record
        trues.append(int(y_t))
        preds['Stacking-Online'].append(float(pred_online))
        preds['Stacking-Offline'].append(float(pred_offline))
        preds['BMA'].append(float(pred_bma_val))
        preds['Best-Logit'].append(float(pred_logit_val))

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

        # Knowledge update: both online models (uses drifted label)
        stacking_online.online_update(x_t_raw, y_t)
        bma_model.online_update(x_t_raw, y_t)

        # Progress
        if (t + 1) % 50 == 0 or t == stream_size - 1:
            _, _, f1_on, br_on = cumulative_metrics(trues, preds['Stacking-Online'])
            _, _, f1_off, br_off = cumulative_metrics(trues, preds['Stacking-Offline'])
            _, _, f1_bma, br_bma = cumulative_metrics(trues, preds['BMA'])
            _, _, f1_bl, br_bl = cumulative_metrics(trues, preds['Best-Logit'])
            print(f"  t={t + 1:>4d}/{stream_size}  "
                  f"F1: On={f1_on:.3f} Off={f1_off:.3f} BMA={f1_bma:.3f} BL={f1_bl:.3f}  "
                  f"Brier: On={br_on:.3f} Off={br_off:.3f} BMA={br_bma:.3f} BL={br_bl:.3f}")

    # Final metrics
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

    # Segment metrics
    segment_metrics = compute_segment_metrics(methods, preds, trues, drift_intervals, stream_size)
    print("\n--- Segment Metrics ---")
    for m in methods:
        print(f"  {m}:")
        for seg_key, segs in segment_metrics[m].items():
            print(f"    {seg_key}: pre={segs['pre_drift']}  during={segs['during_drift']}  post={segs['post_drift']}")

    # Diagnostics
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

    # Save outputs
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

    output_json = {
        'setting': 'A',
        'drift_scenario': scenario,
        'drift_intervals': [[s, e] for s, e in drift_intervals],
        'warmup_size': warmup_size,
        'stream_size': stream_size,
        'seed': args.seed,
        'mini_batch': args.mini_batch,
        'drift_threshold': args.drift_threshold,
        'online_lr': args.online_lr,
        'elapsed_seconds': round(elapsed, 2),
        'methods': results_methods,
        'segment_metrics': segment_metrics,
        'diagnostics': diagnostics,
    }

    json_path = os.path.join(base_dir, 'logs', f'rq4_A_{tag}_drift{scenario}.json')
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved: {json_path}")

    tsv_path = os.path.join(base_dir, 'logs', f'rq4_A_{tag}_drift{scenario}_summary.tsv')
    with open(tsv_path, 'w') as f:
        f.write('method\tprecision\trecall\tF1\tbrier\n')
        for m in methods:
            r = results_methods[m]
            f.write(f"{m}\t{r['precision']}\t{r['recall']}\t{r['f1']}\t{r['brier']}\n")
    print(f"Saved: {tsv_path}")


if __name__ == '__main__':
    main()
