#!/usr/bin/env python3
"""
experiment_online.py
Unified online learning comparison experiment.
Runs three methods on the RQ2 adaptation task:
  1. Static Stacking (no online updates)
  2. Online Stacking with Sliding Window meta-learner retraining
  3. EWA Baseline (exponential weight aggregation)

Compares online vs offline performance against the BMA oracle.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpmath import mp
from geneticalgorithm import geneticalgorithm as ga
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
import os
import json
import copy
from datetime import datetime

from stacking import StackingEnsemble
from ewa_ensemble import EWAEnsemble

warnings.filterwarnings('ignore')
mp.dps = 50


# ============================================================
# Oracle model (BMA) - used to compute ground-truth probability
# ============================================================
class BMA:
    def __init__(self, y, X, **kwargs):
        self.y = y
        self.X = X
        self.names = list(X.columns)
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
        self.Verbose = kwargs.get('Verbose', False)
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


# ============================================================
# Genetic Algorithm globals and functions
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
# Main experiment
# ============================================================
def main():
    global tmp_model, tmp_vars, tmp_data, tmp_index_set, tmp_initial_values

    start_time = datetime.now()
    print(f"=== Online Learning Comparison Experiment Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # ---- Load data ----
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    df = pd.read_csv(os.path.join(data_dir, 'training_rescueRobot_450.csv'))
    dfv = pd.read_csv(os.path.join(data_dir, 'validation_rescueRobot_450.csv'))
    df["firm"] = (df["firm"] == "Yes") * 1
    dfv["firm"] = (dfv["firm"] == "Yes") * 1

    # 离线训练用 df，在线测试用 dfv（Stacking 从未见过验证集）
    X_train = df.drop(["hazard"], axis=1)
    y_train = df["hazard"]
    X_val = dfv.drop(["hazard"], axis=1)
    y_val = dfv["hazard"]

    # ---- Build Oracle BMA (trained on validation set) ----
    print("\nBuilding Oracle model (BMA)...")
    oracle = BMA(y_val, add_constant(X_val), RegType='Logit', Verbose=False).fit()

    # ---- Train shared Stacking model offline (on training set only) ----
    print("\nBuilding Stacking model (offline on training set)...")
    stacking = StackingEnsemble(n_folds=10, random_state=42)
    stacking.fit(X_train, y_train)

    print(f"\nMeta-learner coefficient dimensions: {stacking.meta_learner.coef_.shape}")
    print(f"Meta-learner regularization C: {stacking.meta_learner.C}")
    print(f"Log-odds space: {stacking.use_log_odds}")
    top_idx = np.argsort(np.abs(stacking.meta_learner.coef_[0]))[::-1][:10]
    print("Top 10 base models by weight:")
    for idx in top_idx:
        print(f"  Model {stacking.model_index_sets[idx]}: w={stacking.meta_learner.coef_[0][idx]:.4f}")

    # ---- Create three method instances ----
    # Static: original stacking, no updates
    static_stacking = stacking

    # Online Stacking: deep copy so online_update does not affect static
    online_stacking = copy.deepcopy(stacking)

    # EWA: wraps the original stacking's base models (does not modify them)
    ewa = EWAEnsemble(stacking, eta=0.5, use_log_odds=True)

    # ---- Configurable variables for GA ----
    vars_dict = {
        5: ('power', ['int'], [13, 78], 0.8),
        6: ('band', ['real'], [14.7, 46.58], 0.4),
        7: ('quality', ['real'], [0, 147.19], 0.2),
        8: ('speed', ['int'], [15, 64], 0.1)
    }

    # 在线阶段：用验证集中 hazard=0 的行（Stacking 从未见过这些数据）
    selected_rows = [i for i in dfv.index if dfv.loc[i, 'hazard'] == 0]
    print(f"\nOnline phase: {len(selected_rows)} unseen samples from validation set")

    # ---- Define methods ----
    methods = {
        'Static Stacking': {'model': static_stacking, 'online': False},
        'Online Stacking (SW)': {'model': online_stacking, 'online': True, 'type': 'stacking'},
        'EWA Baseline': {'model': ewa, 'online': True, 'type': 'ewa'},
    }

    results = {}

    # ---- Run each method sequentially (GA uses globals, cannot parallelize) ----
    for method_name, config in methods.items():
        print(f"\n--- Running: {method_name} ---")
        results[method_name] = []
        model = config['model']

        for count, r in enumerate(selected_rows):
            # 从验证集取数据（Stacking 离线阶段从未见过）
            row_data = dfv.drop(["hazard"], axis=1)
            row_data = add_constant(row_data)[r:(r + 1)]

            # Set GA globals
            tmp_model = model
            tmp_vars = vars_dict
            tmp_data = row_data
            tmp_index_set = [5, 6, 7, 8]
            tmp_initial_values = [row_data[vars_dict[k][0]].values[0] for k in tmp_index_set]

            # Run GA adaptation
            new_data = run_adaptation(model, vars_dict, tmp_index_set, row_data, fitness)
            prediction = model.predict_single(new_data)
            pred_oracle = oracle.predict(row_data)[0]

            re_val = abs(prediction - pred_oracle) / pred_oracle
            success = prediction > 0.5 and pred_oracle > 0.5

            results[method_name].append({
                'row': r, 'RE': re_val, 'success': success,
                'prediction': prediction, 'oracle': pred_oracle
            })

            # Online update (if applicable)
            if config['online']:
                y_feedback = 1 if pred_oracle > 0.5 else 0
                if config['type'] == 'stacking':
                    X_raw_row = dfv.drop(["hazard"], axis=1)[r:(r + 1)]
                    model.online_update(X_raw_row, y_feedback, window_size=100)
                elif config['type'] == 'ewa':
                    model.online_update(row_data, y_feedback)

            # Progress every 50
            if (count + 1) % 50 == 0:
                re_vals = [x['RE'] for x in results[method_name]]
                succ_vals = [x['success'] for x in results[method_name]]
                print(f"  [{method_name}] {count + 1}/{len(selected_rows)} "
                      f"Median RE: {np.median(re_vals):.4f}, "
                      f"Success: {np.mean(succ_vals):.4f}")

    end_time = datetime.now()
    duration = end_time - start_time

    # ---- Print comparison table ----
    print("\n" + "=" * 65)
    print("=== Online Learning Comparison Results ===")
    print("=" * 65)
    print(f"{'Method':<22} {'Median RE':>10} {'Mean RE':>10} {'Success Rate':>14}")
    print("-" * 65)
    for method_name in methods:
        re_vals = [x['RE'] for x in results[method_name]]
        succ_vals = [x['success'] for x in results[method_name]]
        median_re = np.median(re_vals)
        mean_re = np.mean(re_vals)
        success_rate = np.mean(succ_vals)
        print(f"{method_name:<22} {median_re:>10.4f} {mean_re:>10.4f} {success_rate:>13.2%}")
    print(f"{'BMA (paper)':<22} {'0.0242':>10} {'-':>10} {'75.82%':>14}")
    print("=" * 65)
    print(f"Total runtime: {duration}")

    # ---- Save results ----
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # TSV file with all per-sample results
    tsv_file = os.path.join(log_dir, 'online_comparison_results.tsv')
    with open(tsv_file, 'w') as f:
        f.write("method\trow\tRE\tsuccess\tprediction\toracle\n")
        for method_name in methods:
            for r in results[method_name]:
                f.write(f"{method_name}\t{r['row']}\t{r['RE']}\t"
                        f"{'TRUE' if r['success'] else 'FALSE'}\t"
                        f"{r['prediction']}\t{r['oracle']}\n")

    # JSON summary
    summary = {
        'experiment': 'Online Learning Comparison (RQ4)',
        'design': 'Offline train on df (training set), online test on dfv (validation set, unseen)',
        'n_offline_train': len(X_train),
        'n_online_samples': len(selected_rows),
        'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': str(duration),
        'methods': {},
        'baseline_BMA': {'median_RE': 0.0242, 'success_rate': 0.7582}
    }
    for method_name in methods:
        re_vals = [x['RE'] for x in results[method_name]]
        succ_vals = [x['success'] for x in results[method_name]]
        summary['methods'][method_name] = {
            'median_RE': float(np.median(re_vals)),
            'mean_RE': float(np.mean(re_vals)),
            'success_rate': float(np.mean(succ_vals)),
            'success_count': int(sum(succ_vals)),
            'total_count': len(succ_vals)
        }

    summary_file = os.path.join(log_dir, 'online_comparison_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved: {tsv_file}")
    print(f"Summary saved: {summary_file}")


if __name__ == '__main__':
    main()
