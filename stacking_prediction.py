#!/usr/bin/env python3
"""
stacking_prediction.py
运行 Stacking 的 RQ1 预测精度实验，与 BMA 基线对比。
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpmath import mp
from statsmodels.tools import add_constant
from itertools import combinations
from sklearn.metrics import recall_score, precision_score, f1_score
import time
import warnings
import os
import json
from stacking import StackingEnsemble

warnings.filterwarnings('ignore')
mp.dps = 50


# ============================================================
# BMA 模型（与 stacking_adaptation.py 中相同）
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
# 评估函数
# ============================================================
def precision(predictions, true_labels):
    return precision_score(true_labels, (predictions > 0.5))

def recall(predictions, true_labels):
    return recall_score(true_labels, (predictions > 0.5))

def f_measure(predictions, true_labels):
    return f1_score(true_labels, (predictions > 0.5))


# ============================================================
# 主程序
# ============================================================
def main():
    print("=== Stacking RQ1 Experiment Start ===")

    # 数据路径（假设脚本在 stacking-package 同级目录下）
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    df = pd.read_csv(os.path.join(data_dir, 'training_rescueRobot_450.csv'))
    dfv = pd.read_csv(os.path.join(data_dir, 'validation_rescueRobot_450.csv'))
    df["firm"] = (df["firm"] == "Yes") * 1
    dfv["firm"] = (dfv["firm"] == "Yes") * 1

    # 训练集（450 个样本）
    X_train = df.drop(["hazard"], axis=1)[:450]
    y_train = df["hazard"][:450]
    # 验证集（作为测试集）
    X_val = dfv.drop(["hazard"], axis=1)
    y_val = dfv["hazard"]

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

    # ----------------------------- BMA 基线 -----------------------------
    print("\n训练 BMA 模型...")
    start = time.time()
    bma = BMA(y_train, add_constant(X_train), RegType='Logit', Verbose=False).fit()
    bma_time = time.time() - start
    pred_bma = bma.predict(add_constant(X_val))
    bma_prec = precision(pred_bma, y_val)
    bma_rec = recall(pred_bma, y_val)
    bma_f1 = f_measure(pred_bma, y_val)
    print(f"BMA: precision={bma_prec:.4f}, recall={bma_rec:.4f}, F1={bma_f1:.4f}, train_time={bma_time:.2f}s")

    # ----------------------------- Stacking 模型 -----------------------------
    print("\n训练 Stacking 模型 (n_folds=10)...")
    start = time.time()
    stacking = StackingEnsemble(n_folds=10, random_state=42)
    stacking.fit(X_train, y_train)
    stacking_time = time.time() - start
    pred_stacking = stacking.predict_proba(X_val)
    stack_prec = precision(pred_stacking, y_val)
    stack_rec = recall(pred_stacking, y_val)
    stack_f1 = f_measure(pred_stacking, y_val)
    print(f"Stacking: precision={stack_prec:.4f}, recall={stack_rec:.4f}, F1={stack_f1:.4f}, train_time={stacking_time:.2f}s")

    # ----------------------------- 保存结果 -----------------------------
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 保存 RQ1 结果
    results_df = pd.DataFrame({
        'model': ['BMA', 'Stacking'],
        'precision': [bma_prec, stack_prec],
        'recall': [bma_rec, stack_rec],
        'f1': [bma_f1, stack_f1],
        'train_time': [bma_time, stacking_time]
    })
    results_df.to_csv(os.path.join(log_dir, 'stacking_rq1_results.tsv'), sep='\t', index=False)
    print(f"RQ1 结果已保存至: {os.path.join(log_dir, 'stacking_rq1_results.tsv')}")

    print("\n=== Summary ===")
    print(f"              Precision   Recall   F1      Train Time")
    print(f"BMA           {bma_prec:.4f}      {bma_rec:.4f}   {bma_f1:.4f}   {bma_time:.2f}s")
    print(f"Stacking      {stack_prec:.4f}      {stack_rec:.4f}   {stack_f1:.4f}   {stacking_time:.2f}s")


if __name__ == '__main__':
    main()