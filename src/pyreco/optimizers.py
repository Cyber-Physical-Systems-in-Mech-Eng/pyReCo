import numpy as np
import random
from abc import ABC, abstractmethod

from sklearn.linear_model import Ridge


class Optimizer(ABC):

    def __init__(self, name: str = ''):
        self.name = name
        pass

    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        # expects input A = [(n_batch*n_time), n_nodes], b = [(n_batch*n_time), n_out]
        # returns W_out = [n_nodes, n_out]
        pass


class RidgeSK(Optimizer):
    # solves a linear regression model using sklearn's Ridge method,
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    def __init__(self, name: str = '', alpha=1.0):
        super().__init__(name)
        self.alpha = 1.0

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        clf = Ridge(self.alpha, fit_intercept=False).fit(A, b)
        W_out = clf.coef_.T
        return W_out


def assign_optimizer(optimizer: str):
    # maps names of optimizers to the correct implementation.
    if optimizer == 'ridge':
        return RidgeSK()

    # TODO: add more solvers (sparsity promoting, ...)
    else:
        raise (ValueError('{self.optimizer} not implemented! Check optimizers.py and assign_optimizers()'))
