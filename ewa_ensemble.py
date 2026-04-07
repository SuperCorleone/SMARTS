"""
ewa_ensemble.py
Exponentially Weighted Aggregation (EWA) ensemble for online learning.
Uses base models from a pre-trained StackingEnsemble, replacing the
meta-learner with exponential weight updating.
"""

import numpy as np


class EWAEnsemble:
    """
    Exponentially Weighted Aggregation ensemble.
    Uses base models from a pre-trained StackingEnsemble, but maintains
    online-updated weights instead of a meta-learner.
    """

    def __init__(self, stacking_model, eta=0.5, use_log_odds=True):
        """
        Parameters
        ----------
        stacking_model : StackingEnsemble
            A fitted StackingEnsemble instance; its base_models are borrowed.
        eta : float
            Learning rate for exponential weight update.
        use_log_odds : bool
            If True, aggregate predictions in log-odds space.
        """
        self.base_models = list(stacking_model.base_models)
        self.eta = eta
        self.use_log_odds = use_log_odds
        self.n_models = len(self.base_models)
        self.weights = np.ones(self.n_models) / self.n_models

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip(p):
        return np.clip(p, 1e-7, 1 - 1e-7)

    @staticmethod
    def _prob_to_logodds(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    @staticmethod
    def _logodds_to_prob(lo):
        return 1 / (1 + np.exp(-lo))

    def _base_predictions(self, row_data_with_const):
        """Return an array of predicted probabilities from each base model."""
        preds = np.full(self.n_models, 0.5)
        for i, (idx_set, model) in enumerate(self.base_models):
            if model is not None:
                model_X = row_data_with_const.iloc[:, list(idx_set)]
                try:
                    pred = model.predict(model_X)
                    preds[i] = pred.values[0] if hasattr(pred, "values") else pred[0]
                except Exception:
                    pass  # keeps default 0.5
        return preds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, row_data_with_const):
        """
        Weighted combination of base model predictions for a single row.

        Returns
        -------
        float
            Predicted probability for the positive class.
        """
        preds = self._base_predictions(row_data_with_const)
        if self.use_log_odds:
            log_odds = self._prob_to_logodds(preds)
            combined = np.dot(self.weights, log_odds)
            return float(self._logodds_to_prob(combined))
        else:
            return float(np.dot(self.weights, preds))

    def online_update(self, row_data_with_const, y_true):
        """
        Update weights using cross-entropy loss per base model.

        For each base model i:
            loss_i = -[y * log(p_i) + (1 - y) * log(1 - p_i)]
            w_i   *= exp(-eta * loss_i)
        Weights are then normalised to sum to 1.
        """
        preds = self._clip(self._base_predictions(row_data_with_const))
        losses = -(y_true * np.log(preds) + (1 - y_true) * np.log(1 - preds))
        self.weights *= np.exp(-self.eta * losses)
        self.weights /= self.weights.sum()

    def predict_and_update(self, row_data_with_const, y_true):
        """
        Predict first, then update weights. Returns the prediction made
        *before* the weight update (prequential evaluation).
        """
        pred = self.predict_single(row_data_with_const)
        self.online_update(row_data_with_const, y_true)
        return pred

    def get_weights(self):
        """Return a copy of the current weight array."""
        return self.weights.copy()

    def reset_weights(self):
        """Reset all weights to uniform (1/K)."""
        self.weights = np.ones(self.n_models) / self.n_models
