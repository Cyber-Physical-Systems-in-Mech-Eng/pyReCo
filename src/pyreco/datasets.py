"""
Standard chaotic system datasets for reservoir computing.

This module provides time series data from chaotic dynamical systems,
formatted for direct use with PyReCo reservoir computers.
"""

import numpy as np
from reservoirpy import datasets


def _sliding_window(data, n_in, n_out=1):
    """
    Create sliding window views of time series data.

    Parameters
    ----------
    data : ndarray of shape (n_timesteps, n_features)
        Input time series
    n_in : int
        Number of time steps in input window
    n_out : int, default=1
        Number of time steps in output window

    Returns
    -------
    X : ndarray of shape (n_samples, n_in, n_features)
        Input windows
    y : ndarray of shape (n_samples, n_out, n_features)
        Output windows (targets)
    """
    n_timesteps, n_features = data.shape
    n_samples = n_timesteps - n_in - n_out + 1

    if n_samples < 1:
        raise ValueError(
            f"Not enough data: need at least {n_in + n_out} timesteps, got {n_timesteps}"
        )

    X = np.stack([data[i:i+n_in] for i in range(n_samples)], axis=0)
    y = np.stack([data[i+n_in:i+n_in+n_out] for i in range(n_samples)], axis=0)

    return X, y


def load(dataset_name, n_samples=5000, train_fraction=0.7, n_in=100, n_out=1, seed=None, **kwargs):
    """
    Load chaotic time series dataset with train/test split.

    This function generates data from chaotic dynamical systems using reservoirpy
    and formats it for reservoir computing tasks using sliding windows.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Supported options:
        - 'lorentz69' or 'lorenz' or 'lorenz63': Lorenz 1963 attractor (3D system)
        - 'mackey_glass' or 'mackeyglass' or 'mg': Mackey-Glass delay system (1D system)
    n_samples : int, default=5000
        Number of time steps to generate from the chaotic system
    train_fraction : float, default=0.7
        Fraction of data to use for training (between 0 and 1)
    n_in : int, default=100
        Number of time steps in each input window
    n_out : int, default=1
        Number of time steps in each output window (prediction horizon)
    seed : int, optional
        Random seed for reproducibility (used for Mackey-Glass)
    **kwargs : dict
        Additional parameters passed to the generator function:
        - For Lorenz: sigma, rho, beta, h (time step), x0 (initial condition)
        - For Mackey-Glass: tau, a, b, n, h (time step), x0 (initial value)

    Returns
    -------
    x_train : ndarray of shape (n_train_samples, n_in, n_features)
        Training input windows
    y_train : ndarray of shape (n_train_samples, n_out, n_features)
        Training output windows (targets)
    x_test : ndarray of shape (n_test_samples, n_in, n_features)
        Test input windows
    y_test : ndarray of shape (n_test_samples, n_out, n_features)
        Test output windows (targets)

    Examples
    --------
    >>> # Load Lorenz system data
    >>> x_train, y_train, x_test, y_test = load('lorentz69', n_samples=5000, seed=42)
    >>> print(x_train.shape)  # (3400, 100, 3)
    >>> print(y_train.shape)  # (3400, 1, 3)

    >>> # Load Mackey-Glass data
    >>> x_train, y_train, x_test, y_test = load('mackey_glass', n_samples=5000, seed=42)
    >>> print(x_train.shape)  # (3400, 100, 1)
    >>> print(y_train.shape)  # (3400, 1, 1)

    Notes
    -----
    - Input: Past n_in time steps
    - Output: Future n_out time steps (immediately following the input window)
    - Data is split chronologically (earlier data for training, later for testing)
    - Reproducible when seed is set (for Mackey-Glass)
    - Uses reservoirpy.datasets for data generation
    """
    # Normalize dataset name
    dataset_name = dataset_name.lower().replace('-', '_').replace(' ', '_')

    # Generate raw time series using reservoirpy.datasets
    if dataset_name in ['lorentz69', 'lorenz', 'lorenz63']:
        # Extract Lorenz-specific parameters with defaults
        sigma = kwargs.get('sigma', 10.0)
        rho = kwargs.get('rho', 28.0)
        beta = kwargs.get('beta', 8.0/3.0)
        h = kwargs.get('h', 0.01)
        x0 = kwargs.get('x0', [1.0, 1.0, 1.0])

        raw_data = datasets.lorenz(
            n_timesteps=n_samples,
            sigma=sigma,
            rho=rho,
            beta=beta,
            h=h,
            x0=x0
        )
        raw_data = np.asarray(raw_data, dtype=np.float64)

    elif dataset_name in ['mackey_glass', 'mackeyglass', 'mg']:
        # Extract Mackey-Glass-specific parameters with defaults
        tau = kwargs.get('tau', 17)
        a = kwargs.get('a', 0.2)
        b = kwargs.get('b', 0.1)
        n = kwargs.get('n', 10)
        h = kwargs.get('h', 1.0)
        x0 = kwargs.get('x0', 1.2)

        raw_data = datasets.mackey_glass(
            n_timesteps=n_samples,
            tau=tau,
            a=a,
            b=b,
            n=n,
            h=h,
            x0=x0,
            seed=seed
        )
        raw_data = np.asarray(raw_data, dtype=np.float64)
        # Ensure 2D shape (n_timesteps, 1)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1)

    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Supported: 'lorentz69', 'lorenz', 'lorenz63', 'mackey_glass', 'mackeyglass', 'mg'"
        )

    # Split time series first (to avoid data leakage)
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    n_timesteps = len(raw_data)
    n_train_timesteps = int(n_timesteps * train_fraction)

    train_data = raw_data[:n_train_timesteps]
    test_data = raw_data[n_train_timesteps:]

    # Create sliding windows separately for train and test
    x_train, y_train = _sliding_window(train_data, n_in=n_in, n_out=n_out)
    x_test, y_test = _sliding_window(test_data, n_in=n_in, n_out=n_out)

    return x_train, y_train, x_test, y_test
