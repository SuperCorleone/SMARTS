#!/usr/bin/env python3
"""
stacking_adaptation.py
运行 Stacking 的 RQ2 适应决策实验，与 BMA 基线对比。
可直接运行: python stacking_adaptation.py
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
from datetime import datetime

# 导入 Stacking 类
from stacking import StackingEnsemble

warnings.filterwarnings('ignore')
mp.dps = 50


# ============================================================
# Oracle 模型 (BMA) – 用于计算真实概率
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
# 遗传算法相关（全局变量）
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
# 主实验
# ============================================================
def main():
    global tmp_model, tmp_vars, tmp_data, tmp_index_set, tmp_initial_values

    start_time = datetime.now()
    print(f"=== Stacking RQ2 Experiment Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 数据路径（假设脚本在 stacking-package 同级目录下）
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    df = pd.read_csv(os.path.join(data_dir, 'training_rescueRobot_450.csv'))
    dfv = pd.read_csv(os.path.join(data_dir, 'validation_rescueRobot_450.csv'))
    df["firm"] = (df["firm"] == "Yes") * 1
    dfv["firm"] = (dfv["firm"] == "Yes") * 1

    LIMIT = 450
    X_oracle = dfv.drop(["hazard"], axis=1)
    y_oracle = dfv["hazard"]
    X = df.drop(["hazard"], axis=1)[0:LIMIT]
    y = df["hazard"][0:LIMIT]

    print("\nBuilding Oracle model (BMA)...")
    oracle = BMA(y_oracle, add_constant(X_oracle), RegType='Logit', Verbose=False).fit()

    print("\nBuilding Stacking model...")
    stacking = StackingEnsemble(n_folds=10, random_state=42)
    stacking.fit(X, y)

    # 打印元学习器信息
    print(f"\nMeta-learner coefficient dimensions: {stacking.meta_learner.coef_.shape}")
    print(f"Meta-learner regularization C: {stacking.meta_learner.C}")
    print(f"Log-odds space: {stacking.use_log_odds}")
    top_idx = np.argsort(np.abs(stacking.meta_learner.coef_[0]))[::-1][:10]
    print("Top 10 base models by weight:")
    for idx in top_idx:
        print(f"  Model {stacking.model_index_sets[idx]}: w={stacking.meta_learner.coef_[0][idx]:.4f}")

    # 可配置变量
    vars_dict = {
        5: ('power', ['int'], [13, 78], 0.8),
        6: ('band', ['real'], [14.7, 46.58], 0.4),
        7: ('quality', ['real'], [0, 147.19], 0.2),
        8: ('speed', ['int'], [15, 64], 0.1)
    }

    print('\n=== Adaptation with Stacking ===')
    selected_rows = [i for i in df.index if df.loc[i, 'hazard'] == 0]
    print(f"Number of samples: {len(selected_rows)}")

    results = []
    for count, r in enumerate(selected_rows):
        row_data = df.drop(["hazard"], axis=1)
        row_data = add_constant(row_data)[r:(r + 1)]

        tmp_model = stacking
        tmp_vars = vars_dict
        tmp_data = row_data
        tmp_index_set = [5, 6, 7, 8]
        tmp_initial_values = [row_data[vars_dict[k][0]].values[0] for k in tmp_index_set]

        new_data = run_adaptation(stacking, vars_dict, tmp_index_set, row_data, fitness)
        prediction = stacking.predict_single(new_data)
        pred_oracle = oracle.predict(row_data)[0]

        re_val = abs(prediction - pred_oracle) / pred_oracle
        success = prediction > 0.5 and pred_oracle > 0.5

        results.append({'row': r, 'RE': re_val, 'success': success,
                        'prediction': prediction, 'oracle': pred_oracle})
        print(f'Stacking RE: {re_val} Success: {success}')

        if (count + 1) % 50 == 0:
            elapsed = datetime.now() - start_time
            re_vals = [r['RE'] for r in results]
            succ_vals = [r['success'] for r in results]
            print(f"  Progress: {count + 1}/{len(selected_rows)} "
                  f"(Median RE: {np.median(re_vals):.4f}, "
                  f"Success Rate: {np.mean(succ_vals):.4f}, "
                  f"Time Elapsed: {elapsed})")

    end_time = datetime.now()
    re_values = [r['RE'] for r in results]
    success_values = [r['success'] for r in results]

    print("\n=== Experiment Results Summary ===")
    print(f"Number of samples: {len(results)}")
    print(f"Median RE: {np.median(re_values):.6f}")
    print(f"Mean RE: {np.mean(re_values):.6f}")
    print(f"Std RE: {np.std(re_values):.6f}")
    print(f"Success Rate: {np.mean(success_values):.4f}")
    print(f"Success Count: {sum(success_values)}/{len(success_values)}")
    print(f"Runtime: {end_time - start_time}")

    print("\n=== Comparison with BMA Baseline ===")
    print(f"BMA Median RE (paper): 0.0242  |  Success Rate: 0.7582")
    print(f"Stacking:        {np.median(re_values):.4f}  |  Success Rate: {np.mean(success_values):.4f}")

    # 保存结果
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'stacking_rq2_results.tsv')
    with open(log_file, 'w') as f:
        f.write("type\tmodel\tRE\tsuccess\n")
        for r in results:
            f.write(f"Stacking\tStacking\t{r['RE']}\t{'TRUE' if r['success'] else 'FALSE'}\n")

    summary_file = os.path.join(log_dir, 'stacking_rq2_summary.json')
    summary = {
        'experiment': 'Stacking RQ2 with same GLM as BMA',
        'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': str(end_time - start_time),
        'n_samples': len(results),
        'n_folds': 10,
        'n_base_models': len(stacking.base_models),
        'meta_C': 0.1,
        'log_odds_space': True,
        'metrics': {
            'median_RE': float(np.median(re_values)),
            'mean_RE': float(np.mean(re_values)),
            'std_RE': float(np.std(re_values)),
            'success_rate': float(np.mean(success_values)),
            'success_count': int(sum(success_values)),
            'total_count': len(success_values)
        },
        'baseline_BMA': {'median_RE': 0.0242, 'success_rate': 0.7582},
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved: {log_file}")
    print(f"Summary saved: {summary_file}")


if __name__ == '__main__':
    main()