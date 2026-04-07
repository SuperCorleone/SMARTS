"""
stacking.py
StackingEnsemble 类实现，用于替代 BMA 的模型集成。
使用与 BMA 相同的变量子集逻辑回归作为基模型，在 log-odds 空间进行元学习。
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpmath import mp
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold
from statsmodels.tools import add_constant
from itertools import combinations
from collections import deque

mp.dps = 50


class StackingEnsemble:
    """
    Stacking 集成模型，基模型为所有可能的变量子集上的 Logistic 回归（经过 Occam's window 剪枝）。
    元学习器在 log-odds 空间训练，使用更强的正则化。
    """

    def __init__(self, n_folds=10, random_state=42, max_vars=None):
        self.n_folds = n_folds
        self.random_state = random_state
        self.max_vars = max_vars       # 基模型最大变量数，None 则自动取 min(nCols, 15)
        self.base_models = []          # 存储 (特征索引集, 训练好的模型)
        self.meta_learner = None       # 元学习器（逻辑回归）
        self.model_index_sets = []     # 所有基模型的特征索引集
        self.n_features = None
        self.use_log_odds = True       # 在对数几率空间进行组合
        # Online learning state
        self._online_buffer_X = []
        self._online_buffer_y = []
        self._online_window_size = 100
        self._samples_seen = 0        # 累计已见样本数
        # Drift detection state (ADWIN-inspired)
        self._drift_error_window = deque(maxlen=200)
        self._drift_detected = False
        self._drift_count = 0
        self._retrain_buffer_X = []
        self._retrain_buffer_y = []
        self._retrain_buffer_max = 150

    def _enumerate_models_occam(self, X, y):
        """使用 Occam's window 枚举所有变量子集模型。
        从上一层好模型向上扩展生成候选，避免枚举所有 C(n,k) 组合导致 OOM。
        """
        nCols = X.shape[1]
        if self.max_vars is not None:
            max_size = min(self.max_vars, nCols)
        else:
            max_size = min(nCols, 15)
        max_likelihood = 0
        selected_models = []
        Models_previous = []  # 上一层通过 Occam's window 的好模型
        for num_elements in range(1, max_size + 1):
            if num_elements == 1:
                Models_current = [(i,) for i in range(nCols)]
            else:
                # 从上一层好模型扩展：每个好模型加一个新变量，去重
                candidates = set()
                for M_good in Models_previous:
                    for new_var in range(nCols):
                        if new_var not in M_good:
                            candidates.add(tuple(sorted(M_good + (new_var,))))
                Models_current = list(candidates)
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
            # 当前层没有任何模型通过 Occam's window，提前终止
            if not Models_previous:
                break
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
        训练 Stacking 模型（可用于全量训练或 warm-up 冷启动）。
        对小数据集自动降低 CV 折数。

        X_raw : 原始特征（不含常数项）
        y     : 目标变量 (0/1)
        """
        X_with_const = add_constant(X_raw)
        self.n_features = X_with_const.shape[1]
        y_arr = np.asarray(y, dtype=float)
        n_samples = len(y_arr)

        # 自动调整 CV 折数：确保每折至少有 10 个样本
        actual_folds = min(self.n_folds, max(2, n_samples // 10))
        if actual_folds < self.n_folds:
            print(f"  Auto-adjusted n_folds: {self.n_folds} → {actual_folds} (small dataset: {n_samples} samples)")

        # 步骤1：枚举所有基模型（变量子集）
        print("  Step 1: Enumerating model space...")
        self.model_index_sets = self._enumerate_models_occam(X_with_const, y)
        print(f"  Number of base models: {len(self.model_index_sets)}")

        n_models = len(self.model_index_sets)

        # 步骤2：K 折交叉验证生成元特征（基模型预测概率）
        print(f"  Step 2: {actual_folds}-fold cross validation...")
        meta_probs = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=actual_folds, shuffle=True, random_state=self.random_state)

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

        # 步骤3：训练元学习器（SGDClassifier with log_loss in log-odds space）
        print("  Step 3: Training meta-learner (SGDClassifier, log-odds space)...")
        if self.use_log_odds:
            meta_features = self._prob_to_logodds(meta_probs)
        else:
            meta_features = meta_probs

        self.meta_learner = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.1,         # L2 regularization strength
            learning_rate='constant',
            eta0=0.01,         # conservative learning rate for stability
            random_state=self.random_state,
            max_iter=2000,
            tol=1e-4,
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
        self._samples_seen = n_samples
        print(f"  Done. Valid models: {valid}/{n_models}, samples_seen={self._samples_seen}")
        return self

    def _compute_meta_features(self, row_data_with_const):
        """计算单个样本的元特征向量（基模型预测经 log-odds 变换）"""
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

        return meta_features

    def predict_single(self, row_data_with_const):
        """
        预测单个样本的概率
        row_data_with_const : 已经添加常数项的单行 DataFrame
        """
        meta_features = self._compute_meta_features(row_data_with_const)
        proba = self.meta_learner.predict_proba(meta_features)[:, 1]
        return proba[0]

    def online_update(self, X_new_raw, y_new, window_size=100):
        """
        Online update of meta-learner using SGDClassifier.partial_fit() + drift detection.

        Uses partial_fit() for incremental weight updates one sample at a time.
        Monitors prediction error with an ADWIN-inspired drift detector:
        when drift is detected, retrain on the recent buffer to adapt faster.

        Parameters
        ----------
        X_new_raw : DataFrame
            Raw features (without constant) for one or more new samples.
        y_new : int/float or array-like
            True label(s) for the new sample(s).
        window_size : int
            Max size of the retrain buffer (used when drift triggers full retrain).
        """
        self._online_window_size = window_size
        X_new_with_const = add_constant(X_new_raw, has_constant='add')

        if X_new_with_const.ndim == 1 or X_new_with_const.shape[0] == 1:
            rows = [X_new_with_const.iloc[0:1] if hasattr(X_new_with_const, 'iloc') else X_new_with_const]
            labels = [float(y_new) if np.isscalar(y_new) else float(np.asarray(y_new).ravel()[0])]
        else:
            rows = [X_new_with_const.iloc[i:i+1] for i in range(X_new_with_const.shape[0])]
            labels = list(np.asarray(y_new, dtype=float).ravel())

        for row, label in zip(rows, labels):
            self._samples_seen += 1
            mf = self._compute_meta_features(row)

            # --- Drift detection: monitor prediction error ---
            pred_proba = self.meta_learner.predict_proba(mf)[:, 1][0]
            error = abs(pred_proba - label)
            self._drift_error_window.append(error)
            drift = self._detect_drift()

            # --- Maintain retrain buffer for drift recovery ---
            self._retrain_buffer_X.append(mf[0])
            self._retrain_buffer_y.append(label)
            if len(self._retrain_buffer_X) > self._retrain_buffer_max:
                self._retrain_buffer_X = self._retrain_buffer_X[-self._retrain_buffer_max:]
                self._retrain_buffer_y = self._retrain_buffer_y[-self._retrain_buffer_max:]

            if drift and len(self._retrain_buffer_X) >= 20:
                # Drift detected: retrain on recent buffer
                buf_X = np.array(self._retrain_buffer_X)
                buf_y = np.array(self._retrain_buffer_y)
                self.meta_learner = SGDClassifier(
                    loss='log_loss',
                    penalty='l2',
                    alpha=0.1,
                    learning_rate='constant',
                    eta0=0.01,
                    random_state=self.random_state,
                    max_iter=2000,
                    tol=1e-4,
                )
                self.meta_learner.fit(buf_X, buf_y)
                self._drift_error_window.clear()
                self._drift_count += 1
            else:
                # Normal: incremental update via partial_fit
                self.meta_learner.partial_fit(mf, [label], classes=[0, 1])

    def _detect_drift(self, min_window=30, threshold=0.15):
        """
        ADWIN-inspired drift detector.

        Compares the mean error in the recent half of the window vs the older half.
        If the difference exceeds the threshold, drift is signaled.

        Parameters
        ----------
        min_window : int
            Minimum number of error observations before drift detection activates.
        threshold : float
            Absolute difference in mean error between halves to trigger drift.

        Returns
        -------
        bool
            True if drift is detected.
        """
        errors = self._drift_error_window
        if len(errors) < min_window:
            return False
        n = len(errors)
        mid = n // 2
        old_mean = np.mean(list(errors)[:mid])
        new_mean = np.mean(list(errors)[mid:])
        # Drift if recent errors are significantly higher than older errors
        if new_mean - old_mean > threshold:
            self._drift_detected = True
            return True
        self._drift_detected = False
        return False

    def reset_online_state(self):
        """清除在线缓冲区和漂移检测状态，恢复为纯离线状态"""
        self._online_buffer_X = []
        self._online_buffer_y = []
        self._drift_error_window.clear()
        self._drift_detected = False
        self._drift_count = 0
        self._retrain_buffer_X = []
        self._retrain_buffer_y = []
        self._samples_seen = 0

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