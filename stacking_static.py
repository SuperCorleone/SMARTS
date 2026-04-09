"""
stacking.py
StackingEnsemble 类实现，用于替代 BMA 的模型集成。
使用与 BMA 相同的变量子集逻辑回归作为基模型，在 log-odds 空间进行元学习。
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpmath import mp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from statsmodels.tools import add_constant
from itertools import combinations

mp.dps = 50


class StackingEnsemble:
    """
    Stacking 集成模型，基模型为所有可能的变量子集上的 Logistic 回归（经过 Occam's window 剪枝）。
    元学习器在 log-odds 空间训练，使用更强的正则化。
    """

    def __init__(self, n_folds=10, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = []          # 存储 (特征索引集, 训练好的模型)
        self.meta_learner = None       # 元学习器（逻辑回归）
        self.model_index_sets = []     # 所有基模型的特征索引集
        self.n_features = None
        self.use_log_odds = True       # 在对数几率空间进行组合

    def _enumerate_models_occam(self, X, y):
        """使用 Occam's window 枚举所有变量子集模型"""
        nCols = X.shape[1]
        max_likelihood = 0
        selected_models = []
        for num_elements in range(1, nCols + 1):
            Models_next = list(combinations(list(range(nCols)), num_elements))
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
                model_X = X.iloc[:, list(model_index_set)]
                try:
                    model_regr = sm.Logit(y, model_X).fit(disp=0)
                    model_likelihood = mp.exp(-model_regr.bic / 2)
                    if model_likelihood > max_likelihood / 20:
                        selected_models.append(tuple(model_index_set))
                        Models_previous.append(model_index_set)
                        max_likelihood = max(max_likelihood, model_likelihood)
                except Exception:
                    pass
        return selected_models

    @staticmethod
    def _prob_to_logodds(p):
        """概率转对数几率，避免 log(0)"""
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    @staticmethod
    def _logodds_to_prob(lo):
        """对数几率转概率"""
        return 1 / (1 + np.exp(-lo))

    def fit(self, X_raw, y):
        """
        训练 Stacking 模型
        X_raw : 原始特征（不含常数项）
        y     : 目标变量 (0/1)
        """
        X_with_const = add_constant(X_raw)
        self.n_features = X_with_const.shape[1]
        y_arr = np.asarray(y, dtype=float)

        # 步骤1：枚举所有基模型（变量子集）
        print("  Step 1: Enumerating model space...")
        self.model_index_sets = self._enumerate_models_occam(X_with_const, y)
        print(f"  Number of base models: {len(self.model_index_sets)}")

        n_models = len(self.model_index_sets)
        n_samples = len(y_arr)

        # 步骤2：K 折交叉验证生成元特征（基模型预测概率）
        print(f"  Step 2: {self.n_folds}-fold cross validation...")
        meta_probs = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_with_const)):
            X_train_fold = X_with_const.iloc[train_idx]
            y_train_fold = y_arr[train_idx]
            X_val_fold = X_with_const.iloc[val_idx]
            for i, idx_set in enumerate(self.model_index_sets):
                model_X_train = X_train_fold.iloc[:, list(idx_set)]
                model_X_val = X_val_fold.iloc[:, list(idx_set)]
                try:
                    model = sm.Logit(y_train_fold, model_X_train).fit(disp=0)
                    meta_probs[val_idx, i] = model.predict(model_X_val)
                except Exception:
                    meta_probs[val_idx, i] = 0.5

        # 步骤3：训练元学习器（在 log-odds 空间）
        print("  Step 3: Training meta-learner (log-odds space)...")
        if self.use_log_odds:
            meta_features = self._prob_to_logodds(meta_probs)
        else:
            meta_features = meta_probs

        self.meta_learner = LogisticRegression(
            max_iter=2000,
            C=0.1,          # 较强正则化，防止过拟合
            random_state=self.random_state,
            solver='lbfgs',
            penalty='l2'
        )
        self.meta_learner.fit(meta_features, y_arr)

        # 步骤4：在全部训练数据上重新训练所有基模型
        print("  Step 4: Full training...")
        self.base_models = []
        for idx_set in self.model_index_sets:
            model_X = X_with_const.iloc[:, list(idx_set)]
            try:
                model = sm.Logit(y_arr, model_X).fit(disp=0)
                self.base_models.append((idx_set, model))
            except Exception:
                self.base_models.append((idx_set, None))

        valid = sum(1 for _, m in self.base_models if m is not None)
        print(f"  Done. Valid models: {valid}/{n_models}")
        return self

    def predict_single(self, row_data_with_const):
        """
        预测单个样本的概率
        row_data_with_const : 已经添加常数项的单行 DataFrame
        """
        n_models = len(self.base_models)
        meta_probs = np.zeros((1, n_models))

        for i, (idx_set, model) in enumerate(self.base_models):
            if model is not None:
                model_X = row_data_with_const.iloc[:, list(idx_set)]
                try:
                    pred = model.predict(model_X)
                    meta_probs[0, i] = pred.values[0] if hasattr(pred, 'values') else pred[0]
                except Exception:
                    meta_probs[0, i] = 0.5
            else:
                meta_probs[0, i] = 0.5

        if self.use_log_odds:
            meta_features = self._prob_to_logodds(meta_probs)
        else:
            meta_features = meta_probs

        proba = self.meta_learner.predict_proba(meta_features)[:, 1]
        return proba[0]

    def predict_proba(self, X_raw):
        """
        批量预测概率。
        X_raw: DataFrame, 原始特征（不含常数项）
        返回: 一维数组，每个样本的正类概率
        """
        X_with_const = add_constant(X_raw)
        n_samples = X_with_const.shape[0]
        probas = np.zeros(n_samples)
        for i in range(n_samples):
            row = X_with_const.iloc[i:i+1]
            probas[i] = self.predict_single(row)
        return probas