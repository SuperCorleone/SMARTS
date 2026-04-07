#!/usr/bin/env python3
"""
experiment_online.py
Online-first Stacking experiment with prequential evaluation.

Architecture:
    warm-up (first N training samples) → online stream (remaining training + validation)
    No offline/online hard split — one continuous data flow.

RQ1: Does prediction performance improve over time? (windowed Precision/Recall/F1)
RQ2: Can Online Stacking support effective adaptation? (RE / Success Rate via GA)
RQ3: Computational efficiency vs BMA retrain (cumulative timing)
RQ4: Drift detection ablation (threshold / learning rate sensitivity)

Usage:
    python experiment_online.py                  # run all RQs
    python experiment_online.py --rq 1           # run only RQ1
    python experiment_online.py --warmup 100     # custom warm-up size
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpmath import mp
from statsmodels.tools import add_constant
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
import os
import json
import copy
import time
import argparse
from datetime import datetime

from stacking import StackingEnsemble
from ewa_ensemble import EWAEnsemble

warnings.filterwarnings('ignore')
mp.dps = 50


# ============================================================
# BMA — used as comparison baseline, retrained at checkpoints
# ============================================================
class BMA:
    def __init__(self, y, X, **kwargs):
        self.y = y
        self.X = X
        self.nRows, self.nCols = np.shape(X)
        self.likelihoods = mp.zeros(self.nCols, 1)
        self.likelihoods_all = {}
        self.coefficients_mp = mp.zeros(self.nCols, 1)
        self.coefficients = np.zeros(self.nCols)
        self.probabilities = np.zeros(self.nCols)
        self.MaxVars = kwargs.get('MaxVars', self.nCols)
        priors = kwargs.get('Priors', None)
        if priors is not None and np.size(priors) == self.nCols:
            self.Priors = priors
        else:
            self.Priors = np.ones(self.nCols)
        self.RegType = kwargs.get('RegType', 'LS')

    def fit(self):
        likelighood_sum = 0
        max_likelihood = 0
        for num_elements in range(1, self.MaxVars + 1):
            Models_next = list(combinations(list(range(self.nCols)), num_elements))
            if num_elements == 1:
                Models_current = Models_next
                Models_previous = []
            else:
                idx_keep = np.zeros(len(Models_next))
                for M_new, idx in zip(Models_next, range(len(Models_next))):
                    for M_good in Models_previous:
                        if all(x in M_new for x in M_good):
                            idx_keep[idx] = 1
                            break
                Models_current = np.asarray(Models_next)[np.where(idx_keep == 1)].tolist()
                Models_previous = []
            for model_index_set in Models_current:
                model_X = self.X.iloc[:, list(model_index_set)]
                if self.RegType == 'Logit':
                    model_regr = sm.Logit(self.y, model_X).fit(disp=0)
                else:
                    from statsmodels.regression.linear_model import OLS
                    model_regr = OLS(self.y, model_X).fit()
                model_likelihood = mp.exp(-model_regr.bic / 2) * np.prod(self.Priors[list(model_index_set)])
                if model_likelihood > max_likelihood / 20:
                    self.likelihoods_all[str(model_index_set)] = model_likelihood
                    likelighood_sum = mp.fadd(likelighood_sum, model_likelihood)
                    for idx, i in zip(model_index_set, range(num_elements)):
                        self.likelihoods[idx] = mp.fadd(self.likelihoods[idx], model_likelihood, prec=1000)
                        self.coefficients_mp[idx] = mp.fadd(self.coefficients_mp[idx],
                                                            model_regr.params[i] * model_likelihood, prec=1000)
                    Models_previous.append(model_index_set)
                    max_likelihood = np.max([max_likelihood, model_likelihood])
        self.likelighood_sum = likelighood_sum
        for idx in range(self.nCols):
            self.probabilities[idx] = mp.fdiv(self.likelihoods[idx], likelighood_sum, prec=1000)
            self.coefficients[idx] = mp.fdiv(self.coefficients_mp[idx], likelighood_sum, prec=1000)
        return self

    def predict(self, data):
        data = np.asarray(data)
        if self.RegType == 'Logit':
            try:
                result = 1 / (1 + np.exp(-1 * np.dot(self.coefficients, data)))
            except:
                result = 1 / (1 + np.exp(-1 * np.dot(self.coefficients, data.T)))
        else:
            try:
                result = np.dot(self.coefficients, data)
            except:
                result = np.dot(self.coefficients, data.T)
        return result

    def predict_proba_batch(self, X_with_const):
        """Predict probabilities for multiple rows."""
        probas = np.zeros(X_with_const.shape[0])
        for i in range(X_with_const.shape[0]):
            row = X_with_const.iloc[i:i + 1]
            probas[i] = self.predict(row)[0]
        return probas


# ============================================================
# Genetic Algorithm for adaptation (RQ2)
# ============================================================
tmp_model = None
tmp_vars = None
tmp_data = None
tmp_index_set = None
tmp_initial_values = None


def fitness(X):
    global tmp_model, tmp_vars, tmp_data, tmp_index_set, tmp_initial_values
    i = 0
    for k in tmp_index_set:
        tmp_data.iloc[0, tmp_data.columns.get_loc(tmp_vars[k][0])] = X[i]
        i += 1
    prediction = tmp_model.predict_single(tmp_data)
    if prediction < 0.51:
        prediction = prediction / 10
    delta_change = 0.0
    i = 0
    for k in tmp_index_set:
        delta_change += tmp_vars[k][3] * abs(X[i] - tmp_initial_values[i]) / (
                tmp_vars[k][2][1] - tmp_vars[k][2][0])
        i += 1
    return delta_change - prediction


def run_adaptation(model, vars_dict, index_set, row_data, fitness_fn):
    from geneticalgorithm import geneticalgorithm as ga
    vartype = np.array([vars_dict[k][1] for k in index_set])
    varbound = np.array([vars_dict[k][2] for k in index_set])
    params = {
        'max_num_iteration': 50,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }
    ga_model = ga(
        function=fitness_fn,
        dimension=len(index_set),
        variable_type_mixed=vartype,
        variable_boundaries=varbound,
        convergence_curve=False,
        progress_bar=False,
        algorithm_parameters=params
    )
    ga_model.run()
    assignment = ga_model.output_dict['variable']
    new_row = row_data.copy()
    for i, k in enumerate(index_set):
        new_row.iloc[0, new_row.columns.get_loc(vars_dict[k][0])] = assignment[i]
    return new_row


# ============================================================
# Utility: compute windowed metrics
# ============================================================
def compute_window_metrics(y_true, y_pred_proba):
    """Compute classification metrics for a window of predictions."""
    y_true = np.array(y_true, dtype=int)
    y_pred_proba = np.array(y_pred_proba)
    y_pred_label = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'accuracy': float(np.mean(y_pred_label == y_true)),
        'mean_abs_error': float(np.mean(np.abs(y_pred_proba - y_true))),
    }

    # Precision/Recall/F1 need at least one positive prediction or true label
    if y_true.sum() > 0 and y_pred_label.sum() > 0:
        metrics['precision'] = float(precision_score(y_true, y_pred_label, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred_label, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred_label, zero_division=0))
    else:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0

    # AUC needs both classes present
    if len(set(y_true)) > 1:
        metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba))
    else:
        metrics['auc'] = float('nan')

    return metrics


# ============================================================
# Data loading and stream construction
# ============================================================
def load_data(data_dir):
    """Load and preprocess training + validation data."""
    df = pd.read_csv(os.path.join(data_dir, 'training_rescueRobot_450.csv'))
    dfv = pd.read_csv(os.path.join(data_dir, 'validation_rescueRobot_450.csv'))
    df["firm"] = (df["firm"] == "Yes") * 1
    dfv["firm"] = (dfv["firm"] == "Yes") * 1
    return df, dfv


def build_stream(df_train, df_val, warmup_size):
    """
    Split data into warm-up + online stream.

    Returns:
        X_warmup, y_warmup: warm-up data for initial fit
        X_stream, y_stream: continuous online stream (remaining train + all val)
        stream_source: list of 'train'/'val' labels for each stream sample
    """
    X_train = df_train.drop(["hazard"], axis=1)
    y_train = df_train["hazard"]
    X_val = df_val.drop(["hazard"], axis=1)
    y_val = df_val["hazard"]

    X_warmup = X_train.iloc[:warmup_size].reset_index(drop=True)
    y_warmup = y_train.iloc[:warmup_size].reset_index(drop=True)

    # Online stream = remaining training + all validation
    X_remaining = X_train.iloc[warmup_size:].reset_index(drop=True)
    y_remaining = y_train.iloc[warmup_size:].reset_index(drop=True)

    X_stream = pd.concat([X_remaining, X_val], ignore_index=True)
    y_stream = pd.concat([y_remaining, y_val], ignore_index=True)

    n_remaining = len(X_remaining)
    n_val = len(X_val)
    stream_source = ['train'] * n_remaining + ['val'] * n_val

    return X_warmup, y_warmup, X_stream, y_stream, stream_source


# ============================================================
# RQ1: Prediction performance over time
# ============================================================
def run_online_stream(stacking, X_warmup, y_warmup, X_stream, y_stream,
                      stream_source, window_size=50, rq2_checkpoints=None):
    """
    统一在线循环：prequential 评估 + checkpoint 快照。

    - RQ1 数据：每个样本的预测/标签，按窗口汇总 Precision/Recall/F1
    - RQ2 快照：在指定 checkpoint 保存模型深拷贝 + 累积数据（供 run_rq2 做适应评估）

    Parameters
    ----------
    stacking : StackingEnsemble (已完成 warm-up fit)
    X_warmup, y_warmup : warm-up 数据（用于 BMA 累积训练集）
    X_stream, y_stream : 在线流
    stream_source : 每个样本来源标记 ('train'/'val')
    window_size : 窗口大小
    rq2_checkpoints : list of int, 在线流中的 checkpoint 步骤号（如 [100,200,300,400]）

    Returns
    -------
    rq1_results : dict  (overall, windows, per_sample)
    rq2_snapshots : list of dict  (step, model, cum_X, cum_y)
    """
    if rq2_checkpoints is None:
        rq2_checkpoints = []
    checkpoint_set = set(rq2_checkpoints)

    print("\n=== Online Stream: Prequential Evaluation (Precision / Recall / F1) ===")

    n_stream = len(X_stream)
    X_stream_const = add_constant(X_stream)

    # 累积数据追踪（warm-up 是起点）
    cum_X_list = list(X_warmup.values)
    cum_y_list = list(np.asarray(y_warmup, dtype=float))

    # Per-sample results
    predictions = []
    labels = []
    update_times = []
    rq2_snapshots = []

    for t in range(n_stream):
        row = X_stream_const.iloc[t:t + 1]
        y_t = float(y_stream.iloc[t])

        # 1. Predict with current model
        p_t = stacking.predict_single(row)
        predictions.append(p_t)
        labels.append(int(y_t))

        # 2. Update model with true label
        x_raw = X_stream.iloc[t:t + 1]
        t0 = time.time()
        stacking.online_update(x_raw, y_t)
        update_times.append(time.time() - t0)

        # 3. 累积数据
        cum_X_list.append(X_stream.iloc[t].values)
        cum_y_list.append(y_t)

        # 4. RQ2 checkpoint snapshot
        if t in checkpoint_set:
            snap = {
                'step': t,
                'samples_seen': stacking._samples_seen,
                'model': copy.deepcopy(stacking),
                'cum_X': pd.DataFrame(np.array(cum_X_list),
                                      columns=X_warmup.columns),
                'cum_y': np.array(cum_y_list),
            }
            rq2_snapshots.append(snap)
            print(f"  [Checkpoint t={t}] snapshot saved, "
                  f"samples_seen={stacking._samples_seen}, "
                  f"cum_data={len(cum_y_list)}")

        # Progress every 100 samples
        if (t + 1) % 100 == 0:
            recent_preds = predictions[-100:]
            recent_labels = labels[-100:]
            m = compute_window_metrics(recent_labels, recent_preds)
            print(f"  [{t + 1}/{n_stream}] last-100  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                  f"samples_seen={stacking._samples_seen}")

    # ---- RQ1: Compute windowed metrics ----
    windows = []
    for start in range(0, n_stream, window_size):
        end = min(start + window_size, n_stream)
        if end - start < 10:
            break
        w_labels = labels[start:end]
        w_preds = predictions[start:end]
        m = compute_window_metrics(w_labels, w_preds)
        m['window_start'] = start
        m['window_end'] = end
        m['cumulative_samples'] = stacking._samples_seen - (n_stream - end)
        m['source'] = stream_source[start]
        m['mean_update_time'] = float(np.mean(update_times[start:end]))
        windows.append(m)

    per_sample = []
    for t in range(n_stream):
        per_sample.append({
            'step': t,
            'prediction': float(predictions[t]),
            'y_true': labels[t],
            'abs_error': float(abs(predictions[t] - labels[t])),
            'correct': int((predictions[t] > 0.5) == (labels[t] > 0.5)),
            'source': stream_source[t],
            'update_time_s': float(update_times[t]),
        })

    overall = compute_window_metrics(labels, predictions)
    overall['drift_count'] = stacking._drift_count
    overall['total_update_time'] = float(sum(update_times))

    print(f"\n  RQ1 Overall:  Precision={overall['precision']:.4f}  "
          f"Recall={overall['recall']:.4f}  F1={overall['f1']:.4f}  "
          f"Drift={overall['drift_count']}")

    rq1_results = {
        'overall': overall,
        'windows': windows,
        'per_sample': per_sample,
    }
    return rq1_results, rq2_snapshots


# ============================================================
# RQ2: Adaptation quality at checkpoints (RE / Success Rate)
# ============================================================
def run_rq2(rq2_snapshots, df_val, n_test_samples=20):
    """
    RQ2: 在线 Stacking 能否支持有效的运行时适应决策？

    在在线流的不同阶段，用当前模型做 GA 适应：
    - 适应目标：调整 (power, band, quality, speed) 使 hazard > 0.5
    - Ground truth：用截至当前的数据训练 BMA 作为 oracle
    - 指标：Relative Error (RE) 和 Success Rate

    Parameters
    ----------
    rq2_snapshots : list of dict from run_online_stream
        每个包含 step, model, cum_X, cum_y
    df_val : DataFrame (原始验证集，用于取 hazard=0 的测试样本)
    n_test_samples : int, 每个 checkpoint 用多少 hazard=0 样本做适应测试

    Returns
    -------
    dict with per-checkpoint and overall results
    """
    global tmp_model, tmp_vars, tmp_data, tmp_index_set, tmp_initial_values

    print("\n=== RQ2: Adaptation Quality at Checkpoints (RE / Success Rate) ===")

    # GA 可调变量
    vars_dict = {
        5: ('power', ['int'], [13, 78], 0.8),
        6: ('band', ['real'], [14.7, 46.58], 0.4),
        7: ('quality', ['real'], [0, 147.19], 0.2),
        8: ('speed', ['int'], [15, 64], 0.1)
    }
    ga_index_set = [5, 6, 7, 8]

    # 固定测试样本：验证集中 hazard=0 的前 n_test_samples 行
    X_val = df_val.drop(["hazard"], axis=1)
    test_rows = [i for i in df_val.index if df_val.loc[i, 'hazard'] == 0][:n_test_samples]
    print(f"  Test samples: {len(test_rows)} hazard=0 rows from validation set")

    checkpoint_results = []

    for snap in rq2_snapshots:
        step = snap['step']
        model = snap['model']
        cum_X = snap['cum_X']
        cum_y = snap['cum_y']
        n_seen = snap['samples_seen']

        print(f"\n  --- Checkpoint t={step} (samples_seen={n_seen}) ---")

        # 训练 BMA oracle（用截至当前的全部数据）
        print(f"    Training BMA oracle on {len(cum_y)} samples...")
        t0 = time.time()
        bma_oracle = BMA(cum_y, add_constant(cum_X), RegType='Logit').fit()
        bma_time = time.time() - t0
        print(f"    BMA trained in {bma_time:.2f}s")

        # 对每个测试样本做 GA 适应
        re_vals = []
        successes = []

        for r in test_rows:
            row_data = add_constant(X_val, has_constant='add').iloc[r:r + 1]

            # GA globals
            tmp_model = model
            tmp_vars = vars_dict
            tmp_data = row_data.copy()
            tmp_index_set = ga_index_set
            tmp_initial_values = [row_data[vars_dict[k][0]].values[0] for k in ga_index_set]

            # GA 适应
            new_data = run_adaptation(model, vars_dict, ga_index_set, row_data, fitness)

            # Stacking 预测适应后的配置
            prediction = model.predict_single(new_data)

            # BMA oracle 预测适应后的配置
            pred_oracle = bma_oracle.predict(new_data.values[0])
            if hasattr(pred_oracle, '__len__'):
                pred_oracle = pred_oracle[0]

            # RE 和 Success
            if pred_oracle > 1e-7:
                re_val = abs(prediction - pred_oracle) / pred_oracle
            else:
                re_val = float('nan')
            success = prediction > 0.5 and pred_oracle > 0.5

            re_vals.append(re_val)
            successes.append(success)

        # 汇总
        valid_re = [r for r in re_vals if not np.isnan(r)]
        median_re = float(np.median(valid_re)) if valid_re else float('nan')
        mean_re = float(np.mean(valid_re)) if valid_re else float('nan')
        success_rate = float(np.mean(successes))

        print(f"    Median RE={median_re:.4f}  Mean RE={mean_re:.4f}  "
              f"Success Rate={success_rate:.2%}  BMA time={bma_time:.2f}s")

        checkpoint_results.append({
            'step': step,
            'samples_seen': n_seen,
            'n_test': len(test_rows),
            'median_RE': median_re,
            'mean_RE': mean_re,
            'success_rate': success_rate,
            'success_count': int(sum(successes)),
            'bma_train_time': bma_time,
            're_values': [float(r) for r in re_vals],
        })

    # Overall summary
    print(f"\n  RQ2 Summary:")
    print(f"  {'Checkpoint':>12} {'Samples':>8} {'Median RE':>10} {'Success':>10}")
    print(f"  {'-' * 44}")
    for cr in checkpoint_results:
        print(f"  {cr['step']:>12} {cr['samples_seen']:>8} "
              f"{cr['median_RE']:>10.4f} {cr['success_rate']:>9.2%}")

    return {'checkpoints': checkpoint_results}


# ============================================================
# RQ3: Computational efficiency (partial_fit vs BMA retrain)
# ============================================================
def run_rq3(X_warmup, y_warmup, X_stream, y_stream):
    """
    RQ3: 在线 Stacking 的计算效率是否优于 BMA 的重训策略？

    测量：
    - Online Stacking 单步更新时间 (partial_fit + drift detection)
    - BMA 全量重训时间（随数据量增长）
    在不同累积数据量下对比。

    指标：累计计算时间 vs 样本数、单次更新延迟。
    """
    print("\n=== RQ3: Computational Efficiency (Online Stacking vs BMA Retrain) ===")

    # 评估点：累积数据量
    eval_sizes = [100, 200, 300, 400, 600, 800]
    eval_sizes = [s for s in eval_sizes if s <= len(X_warmup) + len(X_stream)]

    # 构建累积数据
    all_X = pd.concat([X_warmup, X_stream], ignore_index=True)
    all_y = np.concatenate([np.asarray(y_warmup, dtype=float),
                            np.asarray(y_stream, dtype=float)])

    results = []

    # --- Online Stacking: warm-up + stream ---
    print("  Measuring Online Stacking update times...")
    stacking = StackingEnsemble(n_folds=10, random_state=42)

    t0 = time.time()
    stacking.fit(X_warmup, y_warmup)
    warmup_time = time.time() - t0

    online_cumulative = warmup_time
    online_times_per_sample = []
    X_stream_const = add_constant(X_stream)

    for t in range(len(X_stream)):
        x_raw = X_stream.iloc[t:t + 1]
        y_t = float(y_stream.iloc[t])
        t0 = time.time()
        stacking.online_update(x_raw, y_t)
        dt = time.time() - t0
        online_times_per_sample.append(dt)
        online_cumulative += dt

    # --- BMA: full retrain at each eval_size ---
    print("  Measuring BMA retrain times at different data sizes...")
    bma_results = []
    for size in eval_sizes:
        X_sub = add_constant(all_X.iloc[:size])
        y_sub = all_y[:size]
        t0 = time.time()
        BMA(y_sub, X_sub, RegType='Logit').fit()
        bma_time = time.time() - t0
        bma_results.append({'n_samples': size, 'bma_retrain_time': bma_time})
        print(f"    BMA retrain n={size}: {bma_time:.4f}s")

    # --- 汇总：在相同数据量下对比 ---
    print(f"\n  {'Data Size':>10} {'Online Cumul':>14} {'BMA Retrain':>14} {'Speedup':>10}")
    print(f"  {'-' * 52}")

    comparison = []
    warmup_n = len(X_warmup)
    for bma_r in bma_results:
        size = bma_r['n_samples']
        bma_t = bma_r['bma_retrain_time']
        # Online cumulative time up to this data size
        stream_steps = max(0, size - warmup_n)
        online_cum = warmup_time + sum(online_times_per_sample[:stream_steps])
        speedup = bma_t / online_cum if online_cum > 0 else float('inf')

        # Online 单步平均
        if stream_steps > 0:
            avg_step = sum(online_times_per_sample[:stream_steps]) / stream_steps
        else:
            avg_step = 0.0

        print(f"  {size:>10} {online_cum:>13.4f}s {bma_t:>13.4f}s {speedup:>9.2f}x")

        comparison.append({
            'n_samples': size,
            'online_cumulative_time': online_cum,
            'online_avg_step_time': avg_step,
            'bma_retrain_time': bma_t,
            'speedup': speedup,
        })

    return {
        'warmup_time': warmup_time,
        'online_mean_step_time': float(np.mean(online_times_per_sample)),
        'comparison': comparison,
    }


# ============================================================
# RQ4: Drift detection ablation
# ============================================================
def run_rq4(X_warmup, y_warmup, X_stream, y_stream,
            thresholds=(0.05, 0.10, 0.15, 0.20),
            learning_rates=(0.001, 0.005, 0.01, 0.05)):
    """
    RQ4: 漂移检测的敏感性分析（消融实验）。

    对比三种在线更新策略：
      a. 纯 partial_fit（无漂移检测）
      b. partial_fit + 漂移检测（默认方案）
      c. 固定周期重训（每 K 步重训一次）

    并对漂移阈值和学习率做敏感性分析。
    指标：预测误差、漂移触发次数、累计计算时间。
    """
    print("\n=== RQ4: Drift Detection Ablation ===")

    X_stream_const = add_constant(X_stream)

    def eval_online(stacking_model, label):
        """Run prequential eval on stream, return summary."""
        preds, labels_list = [], []
        t0_total = time.time()
        for t in range(len(X_stream)):
            row = X_stream_const.iloc[t:t + 1]
            y_t = float(y_stream.iloc[t])
            p_t = stacking_model.predict_single(row)
            preds.append(p_t)
            labels_list.append(int(y_t))
            x_raw = X_stream.iloc[t:t + 1]
            stacking_model.online_update(x_raw, y_t)
        total_time = time.time() - t0_total
        m = compute_window_metrics(labels_list, preds)
        return {
            'label': label,
            'f1': m['f1'],
            'precision': m['precision'],
            'recall': m['recall'],
            'mean_abs_error': m['mean_abs_error'],
            'drift_count': stacking_model._drift_count,
            'time': total_time,
        }

    def make_fresh_stacking():
        s = StackingEnsemble(n_folds=10, random_state=42)
        s.fit(X_warmup, y_warmup)
        return s

    results = []

    # --- (a) 纯 partial_fit（禁用漂移检测：阈值设为极大） ---
    print("  (a) Pure partial_fit (no drift)...")
    s = make_fresh_stacking()
    s._detect_drift = lambda *a, **kw: False  # 禁用漂移
    r = eval_online(s, 'no_drift')
    results.append(r)
    print(f"      F1={r['f1']:.4f}  MAE={r['mean_abs_error']:.4f}  drift={r['drift_count']}  time={r['time']:.2f}s")

    # --- (b) partial_fit + 漂移检测（默认参数） ---
    print("  (b) partial_fit + drift detection (default threshold=0.15, eta0=0.01)...")
    s = make_fresh_stacking()
    r = eval_online(s, 'drift_default')
    results.append(r)
    print(f"      F1={r['f1']:.4f}  MAE={r['mean_abs_error']:.4f}  drift={r['drift_count']}  time={r['time']:.2f}s")

    # --- (c) 固定周期重训 ---
    for K in [50, 100]:
        print(f"  (c) Fixed-period retrain every K={K} steps...")
        s = make_fresh_stacking()
        s._detect_drift = lambda *a, **kw: False  # 禁用自动漂移
        preds, labels_list = [], []
        t0_total = time.time()
        for t in range(len(X_stream)):
            row = X_stream_const.iloc[t:t + 1]
            y_t = float(y_stream.iloc[t])
            p_t = s.predict_single(row)
            preds.append(p_t)
            labels_list.append(int(y_t))
            x_raw = X_stream.iloc[t:t + 1]
            s.online_update(x_raw, y_t)
            # 固定周期强制重训
            if (t + 1) % K == 0 and len(s._retrain_buffer_X) >= 20:
                from sklearn.linear_model import SGDClassifier
                buf_X = np.array(s._retrain_buffer_X)
                buf_y = np.array(s._retrain_buffer_y)
                s.meta_learner = SGDClassifier(
                    loss='log_loss', penalty='l2', alpha=0.1,
                    learning_rate='constant', eta0=0.01,
                    random_state=42, max_iter=2000, tol=1e-4)
                s.meta_learner.fit(buf_X, buf_y)
                s._drift_count += 1
        total_time = time.time() - t0_total
        m = compute_window_metrics(labels_list, preds)
        r = {
            'label': f'fixed_K{K}',
            'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall'],
            'mean_abs_error': m['mean_abs_error'],
            'drift_count': s._drift_count, 'time': total_time,
        }
        results.append(r)
        print(f"      F1={r['f1']:.4f}  MAE={r['mean_abs_error']:.4f}  retrains={r['drift_count']}  time={r['time']:.2f}s")

    # --- 漂移阈值敏感性 ---
    print("\n  Drift threshold sensitivity:")
    threshold_results = []
    for th in thresholds:
        s = make_fresh_stacking()
        # Monkey-patch threshold
        orig_detect = s._detect_drift.__func__ if hasattr(s._detect_drift, '__func__') else None
        s._detect_drift = lambda min_window=30, threshold=th, _s=s: \
            StackingEnsemble._detect_drift(_s, min_window=min_window, threshold=threshold)
        r = eval_online(s, f'threshold_{th}')
        threshold_results.append(r)
        print(f"    threshold={th:.2f}: F1={r['f1']:.4f}  MAE={r['mean_abs_error']:.4f}  drift={r['drift_count']}")

    # --- 学习率敏感性 ---
    print("\n  Learning rate sensitivity:")
    lr_results = []
    for lr in learning_rates:
        s = make_fresh_stacking()
        s.meta_learner.eta0 = lr
        s.meta_learner.learning_rate = 'constant'
        r = eval_online(s, f'eta0_{lr}')
        lr_results.append(r)
        print(f"    eta0={lr:.4f}: F1={r['f1']:.4f}  MAE={r['mean_abs_error']:.4f}  drift={r['drift_count']}")

    return {
        'strategies': results,
        'threshold_sensitivity': threshold_results,
        'learning_rate_sensitivity': lr_results,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Online Stacking Experiment')
    parser.add_argument('--warmup', type=int, default=100, help='Warm-up size (default: 100)')
    parser.add_argument('--window', type=int, default=50, help='Window size for metrics (default: 50)')
    parser.add_argument('--rq', type=int, default=None, help='Run specific RQ only (1-4)')
    args = parser.parse_args()

    WARMUP_SIZE = args.warmup
    WINDOW_SIZE = args.window

    start_time = datetime.now()
    print(f"=== Online Stacking Experiment Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"    Warm-up size: {WARMUP_SIZE}, Window size: {WINDOW_SIZE}")

    # ---- Load data ----
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    df_train, df_val_raw = load_data(data_dir)

    # ---- Build stream ----
    X_warmup, y_warmup, X_stream, y_stream, stream_source = build_stream(
        df_train, df_val_raw, WARMUP_SIZE
    )
    print(f"\n  Warm-up: {len(X_warmup)} samples")
    print(f"  Online stream: {len(X_stream)} samples "
          f"({sum(1 for s in stream_source if s == 'train')} train + "
          f"{sum(1 for s in stream_source if s == 'val')} val)")
    print(f"  Class balance (warm-up): {y_warmup.mean():.2%} positive")
    print(f"  Class balance (stream):  {y_stream.mean():.2%} positive")

    # ---- Warm-up: initialize Stacking ----
    print(f"\n--- Warm-up Phase: fitting on {WARMUP_SIZE} samples ---")
    stacking = StackingEnsemble(n_folds=10, random_state=42)
    stacking.fit(X_warmup, y_warmup)

    print(f"\n  Meta-learner: {type(stacking.meta_learner).__name__}")
    print(f"  Base models: {len(stacking.base_models)}")
    print(f"  Samples seen after warm-up: {stacking._samples_seen}")

    # ---- RQ2 checkpoints: 在线流中的评估时间点 ----
    n_stream = len(X_stream)
    rq2_checkpoints = [i for i in [99, 199, 299, 399, 499, 599, 699, 799]
                       if i < n_stream]

    # ---- Run online stream (RQ1 + RQ2 snapshots) ----
    results = {}
    run_rq2_flag = (args.rq is None or args.rq == 2)

    rq1_results, rq2_snapshots = run_online_stream(
        stacking, X_warmup, y_warmup, X_stream, y_stream, stream_source,
        window_size=WINDOW_SIZE,
        rq2_checkpoints=rq2_checkpoints if run_rq2_flag else [],
    )
    results['rq1'] = rq1_results

    # ---- Run RQ2: adaptation evaluation at checkpoints ----
    if run_rq2_flag and rq2_snapshots:
        rq2_results = run_rq2(rq2_snapshots, df_val_raw, n_test_samples=20)
        results['rq2'] = rq2_results

    # ---- Run RQ3: computational efficiency ----
    if args.rq is None or args.rq == 3:
        rq3_results = run_rq3(X_warmup, y_warmup, X_stream, y_stream)
        results['rq3'] = rq3_results

    # ---- Run RQ4: drift detection ablation ----
    if args.rq is None or args.rq == 4:
        rq4_results = run_rq4(X_warmup, y_warmup, X_stream, y_stream)
        results['rq4'] = rq4_results

    # ---- Save results ----
    end_time = datetime.now()
    duration = end_time - start_time

    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    output = {
        'experiment': 'Online Stacking (warm-up + continuous stream)',
        'config': {
            'warmup_size': WARMUP_SIZE,
            'window_size': WINDOW_SIZE,
            'n_base_models': len(stacking.base_models),
            'total_samples_seen': stacking._samples_seen,
        },
        'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': str(duration),
    }

    # RQ1 summary (without per_sample to keep JSON small)
    if 'rq1' in results:
        output['rq1'] = {
            'overall': results['rq1']['overall'],
            'windows': results['rq1']['windows'],
        }

    # RQ2 summary
    if 'rq2' in results:
        output['rq2'] = results['rq2']

    # RQ3 summary
    if 'rq3' in results:
        output['rq3'] = results['rq3']

    # RQ4 summary
    if 'rq4' in results:
        output['rq4'] = results['rq4']

    summary_file = os.path.join(log_dir, 'online_experiment_results.json')
    with open(summary_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_file}")

    # Per-sample TSV for plotting
    if 'rq1' in results:
        tsv_file = os.path.join(log_dir, 'online_rq1_per_sample.tsv')
        with open(tsv_file, 'w') as f:
            f.write("step\tprediction\ty_true\tabs_error\tcorrect\tsource\tupdate_time_s\n")
            for r in results['rq1']['per_sample']:
                f.write(f"{r['step']}\t{r['prediction']:.6f}\t{r['y_true']}\t"
                        f"{r['abs_error']:.6f}\t{r['correct']}\t{r['source']}\t"
                        f"{r['update_time_s']:.6f}\n")
        print(f"Per-sample saved: {tsv_file}")

    print(f"\nTotal runtime: {duration}")
    print(f"Total samples seen: {stacking._samples_seen}")
    print(f"Drift retrains: {stacking._drift_count}")


if __name__ == '__main__':
    main()
