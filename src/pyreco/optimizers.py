import numpy as np
import random
from abc import ABC, abstractmethod

from sklearn.linear_model import Ridge


class Optimizer(ABC):

    def __init__(self, name: str = ''):
        """
        Initialize the Optimizer with the specified name.

        Parameters
        ----------
        name : str, optional
            The name of the optimizer. Defaults to an empty string. This can be used to
            uniquely identify the optimizer if multiple optimizers are used in a system.

        Notes
        -----
        This method is intended to be overridden by subclasses to implement specific
        optimization algorithms.
        """
        self.name = name
        pass

    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve for the output weights (W_out) using a given system of equations.

        This method is meant to be implemented by subclasses to apply a specific
        optimization algorithm (e.g., ridge regression, least squares, etc.).

        Parameters
        ----------
        A : numpy.ndarray, shape [(n_batch * n_time), n_nodes]
            The matrix containing the reservoir states over time. Each row corresponds
            to a time step, and each column corresponds to a reservoir node.

        b : numpy.ndarray, shape [(n_batch * n_time), n_out]
            The target output values for the system. Each row corresponds to a time step,
            and each column corresponds to an output value.

        Notes
        -----
        This method is an abstract method and must be implemented in subclasses.
        The implementation will solve for the optimal weights based on the input data
        (A) and target data (b) and may update the internal state of the object.
        """

        # expects input A = [(n_batch*n_time), n_nodes], b = [(n_batch*n_time), n_out]
        # returns W_out = [n_nodes, n_out]
        pass


class RidgeSK(Optimizer):
    # solves a linear regression model using sklearn's Ridge method,
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    def __init__(self, name: str = '', alpha=1.0):
        """
        Initialize the RidgeSK optimizer with a given name and regularization strength.

        Parameters
        ----------
        name : str, optional
            The name of the optimizer (default is '').
        alpha : float, optional
            The regularization strength (default is 1.0). A higher value applies more regularization.

        """
        super().__init__(name)
        self.alpha = 1.0

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear regression problem using Ridge regularization.

        Parameters
        ----------
        A : numpy.ndarray
            The input matrix of shape (n_batch * n_time, n_nodes), representing the features.
        b : numpy.ndarray
            The target matrix of shape (n_batch * n_time, n_out), representing the outputs.

        Returns
        -------
        numpy.ndarray
            The learned weight matrix W_out of shape (n_nodes, n_out).
        """
        clf = Ridge(self.alpha, fit_intercept=False).fit(A, b)
        W_out = clf.coef_.T
        return W_out


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
    # maps names of optimizers to the correct implementation.
    if optimizer == 'ridge':
        return RidgeSK()
    
    if isinstance(optimizer, Optimizer):
        return optimizer
    
    # if isinstance(optimizer, Ridge):
    #     return optimizer

    # TODO: add more solvers (sparsity promoting, ...)
    else:
        raise (ValueError(f'{optimizer} not implemented! Check optimizers.py and assign_optimizers()'))
