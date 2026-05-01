"""
bma.py
Bayesian Model Averaging (BMA) implementation.

Core fit/predict/summary are a verbatim port of utils/BMA.py from the TAAS2024
replication package (Camilli et al., TAAS 2025). The only adaptations are:
  - `y` accepts numpy arrays (not just pandas Series)
  - `model_regr.params.iloc[i]` for pandas 2.x compatibility (base uses [i])
No intercept column is added — matching base library behavior where
`sm.add_constant` is intentionally omitted.

Online update (MAPE-K loop Analyze component, paper eq. 5):
    p(σ | D_{1:t}) = Σ_i p(σ | m_i, D_{1:t}) · p(m_i | D_{1:t})
    p(m_i | D_{1:t}) ∝ p(D_{1:t} | m_i) · p(m_i)               (eq. 5)

Strict alignment with the paper: every newly observed (X, y) updates the
posterior over the model space. With BIC-approximated marginal likelihood,
this is realized by refitting BMA from scratch on the enlarged dataset
(warmup + stream observed so far) — including the StandardScaler, which
the base library refits on every call to fit(). `retrain_every` defaults
to 1 to honor this; raise it only as an engineering shortcut and document
the deviation in the experiment log.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpmath import mp
from itertools import combinations
from sklearn.preprocessing import StandardScaler

mp.dps = 50


# noinspection PyPep8Naming
class BMA:
    def __init__(self, y, X, **kwargs):
        # Prepare X and y — StandardScaler normalization per base library
        self.scaler = StandardScaler()
        X_normalized = self.scaler.fit_transform(X)
        self.X = pd.DataFrame(X_normalized, columns=X.columns)
        # self.X = add_constant(self.X)  # base library: intercept intentionally omitted
        self.y = pd.DataFrame(np.asarray(y).ravel())

        self.names = list(X.columns)
        self.nRows, self.nCols = np.shape(X)
        self.likelihoods = mp.zeros(self.nCols, 1)
        self.likelihoods_all = {}
        self.coefficients_mp = mp.zeros(self.nCols, 1)
        self.coefficients = np.zeros(self.nCols)
        self.probabilities = np.zeros(self.nCols)

        if 'MaxVars' in kwargs.keys():
            self.MaxVars = kwargs['MaxVars']
        else:
            self.MaxVars = self.nCols

        if 'Priors' in kwargs.keys():
            if np.size(kwargs['Priors']) == self.nCols:
                self.Priors = kwargs['Priors']
            else:
                print("WARNING: Provided priors error.  Using equal priors instead.")
                print("The priors should be a numpy array of length equal tot he number of regressor variables.")
                self.Priors = np.ones(self.nCols)
        else:
            self.Priors = np.ones(self.nCols)

        if 'Verbose' in kwargs.keys():
            self.Verbose = kwargs['Verbose']
        else:
            self.Verbose = False

        if 'RegType' in kwargs.keys():
            self.RegType = kwargs['RegType']
        else:
            self.RegType = 'LS'

        # Online update state (MAPE-K loop extension, not in base library).
        # retrain_every=1 is paper-strict (eq. 5: posterior updated on every
        # new observation). Larger values are engineering shortcuts.
        self._orig_X = X.copy()
        self._orig_y = np.asarray(y).ravel()
        self._stream_X = []
        self._stream_y = []
        self._stream_count = 0
        self.retrain_every = kwargs.get('retrain_every', 1)

    def _normalize_data(self, data):
        data_normalized = self.scaler.transform(data)
        data = pd.DataFrame(data_normalized, columns=data.columns)
        return data

    def fit(self):
        # this will be the 'normalization' denominator in Bayes Theorem
        likelighood_sum = 0

        # forall number of elements in the model
        max_likelihood = 0
        for num_elements in range(1, self.MaxVars + 1):

            if self.Verbose:
                print("Computing BMA for models of size: ", num_elements)

            # make a list of all index sets of models of this size
            Models_next = list(combinations(list(range(self.nCols)), num_elements))

            # Occam's window
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
                        else:
                            pass
                Models_current = np.asarray(Models_next)[np.where(idx_keep == 1)].tolist()
                Models_previous = []

            # iterate through all possible models of the given size
            for model_index_set in Models_current:

                # compute the regression for this given model
                model_X = self.X.iloc[:, list(model_index_set)]
                if self.RegType == 'Logit':
                    model_regr = sm.Logit(self.y, model_X).fit(disp=0)
                else:
                    raise Exception("ONLY REGRESSION TYPE.")

                # compute the likelihood (times the prior) for the model
                model_likelihood = mp.exp(-model_regr.bic / 2) * np.prod(self.Priors[list(model_index_set)])

                if (model_likelihood > max_likelihood / 20):
                    if self.Verbose == True:
                        print("Model Variables:", model_index_set, "likelihood=", model_likelihood)
                    self.likelihoods_all[str(model_index_set)] = model_likelihood

                    # add this likelihood to the running tally of likelihoods
                    likelighood_sum = mp.fadd(likelighood_sum, model_likelihood)

                    # add this likelihood (times the priors) for each variable in the model
                    for idx, i in zip(model_index_set, range(num_elements)):
                        self.likelihoods[idx] = mp.fadd(self.likelihoods[idx], model_likelihood, prec=1000)
                        self.coefficients_mp[idx] = mp.fadd(self.coefficients_mp[idx],
                                                            model_regr.params.iloc[i] * model_likelihood, prec=1000)
                    Models_previous.append(model_index_set)
                    max_likelihood = np.max(
                        [max_likelihood, model_likelihood])
                else:
                    if self.Verbose == True:
                        print("Model Variables:", model_index_set, "rejected by Occam's window")

        # divide by the denominator in Bayes theorem to normalize the probabilities sum to one
        self.likelighood_sum = likelighood_sum
        for idx in range(self.nCols):
            self.probabilities[idx] = mp.fdiv(self.likelihoods[idx], likelighood_sum, prec=1000)
            self.coefficients[idx] = mp.fdiv(self.coefficients_mp[idx], likelighood_sum, prec=1000)

        return self

    def predict(self, data):
        data = self._normalize_data(data)
        # data = add_constant(data)  # base library: intercept intentionally omitted

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

    def summary(self):
        df = pd.DataFrame([self.names, list(self.probabilities), list(self.coefficients)],
                          ["Variable Name", "Probability", "Avg. Coefficient"]).T
        return df

    # ================================================================
    # Online update — MAPE-K loop Analyze component (paper eq. 5)
    # Extension on top of the base library; does not alter fit/predict.
    # ================================================================

    def online_update(self, X_new, y_new):
        """
        Append one observation to the stream buffer and refit BMA per
        TAAS2024 eq. (5).

        With retrain_every=1 (default), BMA is refit on
        D_{1:t} = warmup ∪ stream_{1:t} after every new observation —
        the strict realization of
            p(m_i | D_{1:t}) ∝ p(D_{1:t} | m_i) · p(m_i).
        With retrain_every>1 the refit is batched, which is an engineering
        approximation; report the chosen value when comparing to BMA.

        Call AFTER predict() in the MAPE-K loop (Knowledge phase).

        Parameters
        ----------
        X_new : DataFrame
            Raw features (no constant, no normalization) for one sample.
        y_new : int or float
            True label.
        """
        self._stream_X.append(X_new.reset_index(drop=True))
        self._stream_y.append(np.asarray(y_new).ravel())
        self._stream_count += 1
        if self._stream_count % self.retrain_every == 0:
            self._refit()

    def _refit(self):
        """Rebuild BMA from scratch on warmup + accumulated stream.

        Strictly mirrors what `BMA(y_combined, X_combined, ...).fit()` would
        produce on the enlarged dataset: a fresh StandardScaler is fit on the
        combined data (matching the base library's __init__ behavior), and
        all posterior accumulators are zeroed before refitting.
        """
        X_combined = pd.concat([self._orig_X] + self._stream_X, ignore_index=True)
        y_combined = np.concatenate([self._orig_y] + self._stream_y)

        # Refit scaler on enlarged data — TAAS2024 base library refits the
        # StandardScaler in __init__ each time BMA is constructed; preserving
        # the warmup-only scaler would silently misnormalize stream samples
        # whose distribution has drifted.
        self.scaler = StandardScaler()
        X_norm = self.scaler.fit_transform(X_combined)
        self.X = pd.DataFrame(X_norm, columns=X_combined.columns)
        self.y = pd.DataFrame(y_combined)
        self.nRows = len(y_combined)

        # Reset accumulators — fit() accumulates into these without resetting
        self.likelihoods = mp.zeros(self.nCols, 1)
        self.likelihoods_all = {}
        self.coefficients_mp = mp.zeros(self.nCols, 1)
        self.coefficients = np.zeros(self.nCols)
        self.probabilities = np.zeros(self.nCols)

        self.fit()
