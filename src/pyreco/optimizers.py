import numpy as np
import random
from abc import ABC, abstractmethod

from sklearn.linear_model import Ridge


class Optimizer(ABC):

    def __init__(self, name: str = ""):
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

    def __init__(self, name: str = "", alpha=1.0):
        super().__init__(name)
        self.alpha = 1.0

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        clf = Ridge(self.alpha, fit_intercept=False).fit(A, b)
        W_out = clf.coef_.T
        return W_out

    def set_alpha(self, value: float):
        if value < 0:
            raise ValueError("Alpha must be non-negative.")
        self.alpha = value

    def get_alpha(self) -> float:
        return self.alpha


def assign_optimizer(optimizer: str or Optimizer) -> Optimizer:
    """
    Maps names of optimizers to the correct implementation.

    Parameters
    ----------
    optimizer : str or Optimizer
        The name of the optimizer.

    Returns
    -------
    Optimizer
        An instance of the optimizer class corresponding to the given name.

    Raises
    ------
    ValueError
        If the given optimizer name is not implemented.
    """

    if not isinstance(optimizer, str):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"Optimizer must be a string or Optimizer object! Given optimizer is of type:{type(optimizer)}"
            )

    # maps names of optimizers to the correct implementation.
    if optimizer == "ridge" or optimizer == "Ridge":
        return RidgeSK()

    if isinstance(optimizer, Optimizer):
        return optimizer

    # TODO: add more solvers (sparsity promoting, ...)
    else:
        raise (
            ValueError(
                f"{optimizer} not implemented! Check optimizers.py and assign_optimizers()"
            )
        )
