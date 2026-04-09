"""
bma.py
Bayesian Model Averaging (BMA) implementation.
Shared module used by stacking experiment scripts as oracle/baseline.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mpmath import mp
from statsmodels.tools import add_constant
from itertools import combinations

mp.dps = 50


class BMA:
    """
    Bayesian Model Averaging for logistic regression.
    Enumerates variable subsets with Occam's window pruning,
    computes BIC-weighted model average coefficients.
    """

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

    def summary(self):
        return pd.DataFrame({
            'variable': self.names,
            'probability': self.probabilities,
            'coefficient': self.coefficients
        })
