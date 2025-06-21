import numpy as np
import random
from abc import ABC, abstractmethod

from sklearn.linear_model import Ridge
from typing import Optional, Union

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

    def __init__(self, name: str = "", optimizer_properties: dict = {}):
        super().__init__(name)
        self.alpha = optimizer_properties.get("lambda", 1.0)  # regularization strength

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        clf = Ridge(self.alpha, fit_intercept=False).fit(A, b)
        W_out = clf.coef_.T
        return W_out


def assign_optimizer(optimizer: Union[str, Optimizer], optimizer_properties: Optional[dict] = None) -> Optimizer:
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
        return RidgeSK(optimizer_properties = optimizer_properties)

    if isinstance(optimizer, Optimizer):
        return optimizer

    # TODO: add more solvers (sparsity promoting, ...)
    else:
        raise (
            ValueError(
                f"{optimizer} not implemented! Check optimizers.py and assign_optimizers()"
            )
        )
