"""
stacking.py
StackingEnsemble: replaces BMA in the Analyze component of MAPE-K loop.

Base models: variable-subset logistic regression (Occam's window pruning).
Meta-learner: SGDClassifier in log-odds space.

Online update supports:
- Mini-batch accumulation to reduce single-sample SGD noise
- ADWIN-inspired drift detection
- Warm restart (preserves weights, multiple epochs on buffer)
"""

import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from mpmath import mp
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from statsmodels.tools import add_constant

# sklearn < 1.1 uses 'log', >= 1.1 uses 'log_loss'
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
_SGD_LOG_LOSS = 'log_loss' if _sklearn_version >= (1, 1) else 'log'
from collections import deque

mp.dps = 50


class StackingEnsemble:

    def __init__(self, n_folds=10, random_state=42, max_vars=None,
                 online_lr=0.001, mini_batch_size=5,
                 warm_restart_epochs=3,
                 drift_threshold=0.15, drift_window=200,
                 retrain_buffer_max=150):
        """
        Parameters
        ----------
        n_folds : int
            CV folds for meta-feature generation.
        random_state : int
            Random seed.
        max_vars : int or None
            Max variables per base model. None = min(nCols, 15).
        online_lr : float
            SGD learning rate for both initial fit and online updates.
            Default 0.001 (10x lower than previous 0.01) to reduce
            single-sample gradient noise.
        mini_batch_size : int
            Accumulate this many samples before calling partial_fit.
            Reduces gradient variance by ~sqrt(mini_batch_size).
            Set to 1 for single-sample updates (noisier but faster adaptation).
        warm_restart_epochs : int
            Number of passes over buffer on warm restart.
        drift_threshold : float
            ADWIN threshold for triggering drift (mean error difference).
        drift_window : int
            Max size of error observation window.
        retrain_buffer_max : int
            Max samples retained for drift recovery.
        """
        # Architecture
        self.n_folds = n_folds
        self.random_state = random_state
        self.max_vars = max_vars
        self.base_models = []
        self.meta_learner = None
        self.model_index_sets = []
        self.n_features = None
        self.use_log_odds = True

        # Online config
        self.online_lr = online_lr
        self.mini_batch_size = mini_batch_size
        self.warm_restart_epochs = warm_restart_epochs
        self.drift_threshold = drift_threshold

        # Online state
        self._mini_batch_X = []
        self._mini_batch_y = []
        self._samples_seen = 0
        self._drift_error_window = deque(maxlen=drift_window)
        self._drift_detected = False
        self._drift_count = 0
        self._retrain_buffer_X = []
        self._retrain_buffer_y = []
        self._retrain_buffer_max = retrain_buffer_max
        self._update_count = 0
        self._brier_history = []

    # ================================================================
    # Model enumeration
    # ================================================================

    def _enumerate_models_occam(self, X, y):
        """Enumerate variable-subset models passing Occam's window (BIC-based)."""
        nCols = X.shape[1]
        max_size = min(self.max_vars or 15, nCols)
        max_likelihood = 0
        selected_models = []
        Models_previous = []

        for num_elements in range(1, max_size + 1):
            if num_elements == 1:
                Models_current = [(i,) for i in range(nCols)]
            else:
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
            if not Models_previous:
                break

        return selected_models

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def _prob_to_logodds(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    @staticmethod
    def _logodds_to_prob(lo):
        return 1 / (1 + np.exp(-lo))

    # ================================================================
    # Offline training
    # ================================================================

    def fit(self, X_raw, y):
        """
        Offline training: enumerate models, CV meta-features, train meta-learner.

        Parameters
        ----------
        X_raw : DataFrame
            Raw features without constant column.
        y : array-like
            Binary labels (0/1).
        """
        X_with_const = add_constant(X_raw)
        self.n_features = X_with_const.shape[1]
        y_arr = np.asarray(y, dtype=float)
        n_samples = len(y_arr)

        actual_folds = min(self.n_folds, max(2, n_samples // 10))
        if actual_folds < self.n_folds:
            print(f"  Auto-adjusted folds: {self.n_folds} -> {actual_folds} "
                  f"(small dataset: {n_samples})")

        # Step 1: Enumerate base models
        print("  Step 1: Enumerating model space...")
        self.model_index_sets = self._enumerate_models_occam(X_with_const, y)
        n_models = len(self.model_index_sets)
        print(f"  Base models: {n_models}")

        # Step 2: CV meta-features
        print(f"  Step 2: {actual_folds}-fold CV...")
        meta_probs = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=actual_folds, shuffle=True,
                    random_state=self.random_state)

        for _, (train_idx, val_idx) in enumerate(kf.split(X_with_const)):
            X_tr = X_with_const.iloc[train_idx]
            y_tr = y_arr[train_idx]
            X_va = X_with_const.iloc[val_idx]
            for i, idx_set in enumerate(self.model_index_sets):
                try:
                    model = sm.Logit(y_tr, X_tr.iloc[:, list(idx_set)]).fit(disp=0)
                    meta_probs[val_idx, i] = model.predict(X_va.iloc[:, list(idx_set)])
                except Exception:
                    meta_probs[val_idx, i] = 0.5

        # Step 3: Train meta-learner (SGD in log-odds space)
        print("  Step 3: Training meta-learner...")
        meta_features = (self._prob_to_logodds(meta_probs) if self.use_log_odds
                         else meta_probs)

        self.meta_learner = SGDClassifier(
            loss=_SGD_LOG_LOSS, penalty='l2', alpha=0.1,
            learning_rate='constant', eta0=self.online_lr,
            random_state=self.random_state, max_iter=2000, tol=1e-4,
        )
        self.meta_learner.fit(meta_features, y_arr)

        # Step 4: Refit base models on full data
        print("  Step 4: Full training...")
        self.base_models = []
        for idx_set in self.model_index_sets:
            try:
                model = sm.Logit(y_arr, X_with_const.iloc[:, list(idx_set)]).fit(disp=0)
                self.base_models.append((idx_set, model))
            except Exception:
                self.base_models.append((idx_set, None))

        valid = sum(1 for _, m in self.base_models if m is not None)
        self._samples_seen = n_samples
        print(f"  Done. Valid: {valid}/{n_models}, samples={n_samples}")
        return self

    # ================================================================
    # Prediction
    # ================================================================

    def _compute_meta_features(self, row_data_with_const):
        """Compute meta-feature vector (base model predictions in log-odds)."""
        n_models = len(self.base_models)
        meta_probs = np.zeros((1, n_models))
        for i, (idx_set, model) in enumerate(self.base_models):
            if model is not None:
                try:
                    pred = model.predict(row_data_with_const.iloc[:, list(idx_set)])
                    meta_probs[0, i] = (pred.values[0] if hasattr(pred, 'values')
                                        else pred[0])
                except Exception:
                    meta_probs[0, i] = 0.5
            else:
                meta_probs[0, i] = 0.5
        return (self._prob_to_logodds(meta_probs) if self.use_log_odds
                else meta_probs)

    def predict_single(self, row_data_with_const):
        """
        Predict hazard probability for one sample.

        Parameters
        ----------
        row_data_with_const : DataFrame
            Single row with constant column already added.
        """
        mf = self._compute_meta_features(row_data_with_const)
        return self.meta_learner.predict_proba(mf)[:, 1][0]

    def predict_proba(self, X_raw):
        """
        Batch prediction.

        Parameters
        ----------
        X_raw : DataFrame
            Raw features without constant column.
        """
        X_wc = add_constant(X_raw)
        return np.array([self.predict_single(X_wc.iloc[i:i+1])
                         for i in range(X_wc.shape[0])])

    # ================================================================
    # Online update
    # ================================================================

    def online_update(self, X_new_raw, y_new):
        """
        Prequential online update: mini-batch SGD + drift detection.

        Call AFTER predict_single() in the MAPE-K loop (Knowledge phase).

        Parameters
        ----------
        X_new_raw : DataFrame
            Raw features (no constant) for one or more new samples.
        y_new : int/float or array-like
            True label(s).
        """
        X_wc = add_constant(X_new_raw, has_constant='add')

        if X_wc.ndim == 1 or X_wc.shape[0] == 1:
            rows = [X_wc.iloc[0:1] if hasattr(X_wc, 'iloc') else X_wc]
            labels = [float(y_new) if np.isscalar(y_new)
                      else float(np.asarray(y_new).ravel()[0])]
        else:
            rows = [X_wc.iloc[i:i+1] for i in range(X_wc.shape[0])]
            labels = list(np.asarray(y_new, dtype=float).ravel())

        for row, label in zip(rows, labels):
            self._samples_seen += 1
            mf = self._compute_meta_features(row)

            # Track prediction error for drift detection
            pred_proba = self.meta_learner.predict_proba(mf)[:, 1][0]
            error = abs(pred_proba - label)
            self._drift_error_window.append(error)
            self._brier_history.append(error ** 2)

            # Maintain retrain buffer
            self._retrain_buffer_X.append(mf[0])
            self._retrain_buffer_y.append(label)
            if len(self._retrain_buffer_X) > self._retrain_buffer_max:
                self._retrain_buffer_X = self._retrain_buffer_X[-self._retrain_buffer_max:]
                self._retrain_buffer_y = self._retrain_buffer_y[-self._retrain_buffer_max:]

            drift = self._detect_drift()

            if drift:
                self._handle_drift()
                self._mini_batch_X.clear()
                self._mini_batch_y.clear()
            else:
                self._mini_batch_X.append(mf[0])
                self._mini_batch_y.append(label)
                if len(self._mini_batch_X) >= self.mini_batch_size:
                    self._flush_mini_batch()

    def _flush_mini_batch(self):
        """Flush accumulated mini-batch via partial_fit."""
        if not self._mini_batch_X:
            return
        batch_X = np.array(self._mini_batch_X)
        batch_y = np.array(self._mini_batch_y)
        self.meta_learner.partial_fit(batch_X, batch_y, classes=[0, 1])
        self._update_count += 1
        self._mini_batch_X.clear()
        self._mini_batch_y.clear()

    def _handle_drift(self):
        """Warm restart: preserve existing weights, do multiple passes
        over buffer. Gradual adaptation instead of catastrophic reset."""
        if not self._retrain_buffer_X:
            return
        buf_X = np.array(self._retrain_buffer_X)
        buf_y = np.array(self._retrain_buffer_y)

        for epoch in range(self.warm_restart_epochs):
            rng = np.random.RandomState(self.random_state + epoch)
            indices = rng.permutation(len(buf_X))
            for idx in indices:
                self.meta_learner.partial_fit(
                    buf_X[idx:idx+1], [buf_y[idx]], classes=[0, 1])

        self._drift_error_window.clear()
        self._drift_count += 1

    def _detect_drift(self, min_window=30):
        """ADWIN-inspired drift detector: compare recent vs old error means."""
        errors = self._drift_error_window
        if len(errors) < min_window:
            return False
        n = len(errors)
        mid = n // 2
        old_mean = np.mean(list(errors)[:mid])
        new_mean = np.mean(list(errors)[mid:])
        if new_mean - old_mean > self.drift_threshold:
            self._drift_detected = True
            return True
        self._drift_detected = False
        return False

    # ================================================================
    # State management
    # ================================================================

    def reset_online_state(self):
        """Reset online state, keep model weights and base models."""
        self._mini_batch_X.clear()
        self._mini_batch_y.clear()
        self._drift_error_window.clear()
        self._drift_detected = False
        self._drift_count = 0
        self._retrain_buffer_X = []
        self._retrain_buffer_y = []
        self._samples_seen = 0
        self._update_count = 0
        self._brier_history = []

    def get_diagnostics(self):
        """Return online learning diagnostics."""
        return {
            'samples_seen': self._samples_seen,
            'update_count': self._update_count,
            'drift_count': self._drift_count,
            'brier_history': list(self._brier_history),
            'buffer_size': len(self._retrain_buffer_X),
            'pending_mini_batch': len(self._mini_batch_X),
        }
