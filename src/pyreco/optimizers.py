import numpy as np
import random
from abc import ABC, abstractmethod

from sklearn.linear_model import Ridge


class Optimizer(ABC):
    r"""
    Abstract base class of the solvers that find the readout weights.

    A solver gets the matrix :math:`R` of reservoir states and the targets :math:`Y` and returns the readout
    weights :math:`W_\mathrm{out}` with :math:`R\, W_\mathrm{out} \approx Y`.

    Parameters
    ----------
    name : str, default=""
        Name of the optimizer.
    """

    def __init__(self, name: str = ""):
        self.name = name
        pass

    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""
        Solve :math:`A\, W \approx b` for the weights :math:`W`.

        Parameters
        ----------
        A : ndarray of shape (n_batch * n_time, n_nodes)
            Reservoir states of the read-out nodes, one row per sample and time step.
        b : ndarray of shape (n_batch * n_time, n_out)
            Targets.

        Returns
        -------
        ndarray of shape (n_nodes, n_out)
            The readout weights :math:`W_\mathrm{out}`.
        """
        pass


class RidgeSK(Optimizer):
    r"""
    Ridge regression (L2-regularized least squares) with scikit-learn's ``Ridge``, without intercept.

    .. math::

        W_\mathrm{out} = \arg\min_W \left\| A W - b \right\|_F^2 + \alpha \left\| W \right\|_F^2
        = \left( A^\top A + \alpha I \right)^{-1} A^\top b

    The intercept of the readout is handled by the model, which centers states and targets before
    calling :meth:`solve`. See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    Parameters
    ----------
    name : str
        Name of the optimizer.
    alpha : float, default=1.0
        Strength :math:`\alpha \ge 0` of the L2 penalty on the readout weights. It can be changed later with
        :meth:`set_alpha` or ``model.set_hp(alpha=...)``.
    """

    def __init__(self, name: str = "", alpha: float = 1.0):
        super().__init__(name)
        self.set_alpha(alpha)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""Ridge solution :math:`W = (A^\top A + \alpha I)^{-1} A^\top b`, shape ``(n_nodes, n_out)``."""
        clf = Ridge(self.alpha, fit_intercept=False).fit(A, b)
        W_out = clf.coef_.T
        return W_out

    def set_alpha(self, value: float):
        r"""
        Set the ridge penalty :math:`\alpha`.

        Raises a TypeError for something else than a number and a ValueError for a negative or non-finite value.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("Alpha must be a number.")
        if not np.isfinite(value) or value < 0:
            raise ValueError("Alpha must be non-negative and finite.")
        self.alpha = float(value)

    def get_alpha(self) -> float:
        r"""The ridge penalty :math:`\alpha`."""
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
