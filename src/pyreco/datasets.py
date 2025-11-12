"""
Standard chaotic system datasets for reservoir computing.

This module provides time series data from chaotic dynamical systems,
formatted for direct use with PyReCo reservoir computers.

All data generators use numerical integration of the governing equations.
Lorenz system uses scipy.integrate.solve_ivp for stable ODE integration.
Mackey-Glass uses custom Euler integration for delay differential equations.
"""

import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp


def _generate_lorenz(n_timesteps, sigma=10.0, rho=28.0, beta=8.0/3.0, h=0.01, x0=None):
    """
    Generate Lorenz 63 attractor time series using scipy.integrate.solve_ivp.

    Lorenz system equations:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz

    Parameters
    ----------
    n_timesteps : int
        Number of time steps to generate
    sigma : float, default=10.0
        Prandtl number
    rho : float, default=28.0
        Rayleigh number
    beta : float, default=8/3
        Geometric factor
    h : float, default=0.01
        Integration time step
    x0 : array-like of shape (3,), optional
        Initial condition [x, y, z]. If None, defaults to [1.0, 1.0, 1.0]

    Returns
    -------
    trajectory : ndarray of shape (n_timesteps, 3)
        Lorenz attractor trajectory
    """
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    else:
        x0 = np.asarray(x0, dtype=np.float64)

    # Define Lorenz system for solve_ivp (requires t, y signature)
    def lorenz_deriv(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    # Define time points for solution
    t_span = (0, (n_timesteps - 1) * h)
    t_eval = np.linspace(0, (n_timesteps - 1) * h, n_timesteps)

    # Solve ODE using scipy.integrate.solve_ivp with default parameters
    sol = solve_ivp(lorenz_deriv, t_span, x0, t_eval=t_eval)

    # Check if integration was successful
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # Return trajectory as (n_timesteps, 3) array
    return sol.y.T


def _generate_mackey_glass(n_timesteps, tau=17, a=0.2, b=0.1, n=10, x0=1.2, h=1.0, seed=None):
    """
    Generate Mackey-Glass time series using delay differential equation integration.

    Mackey-Glass equation:
        dx/dt = a*x(t-τ)/(1 + x(t-τ)^n) - b*x(t)

    Parameters
    ----------
    n_timesteps : int
        Number of time steps to generate
    tau : int, default=17
        Time delay
    a : float, default=0.2
        Production rate
    b : float, default=0.1
        Degradation rate
    n : float, default=10
        Hill coefficient (nonlinearity)
    x0 : float, default=1.2
        Initial condition
    h : float, default=1.0
        Integration time step
    seed : int, optional
        Random seed for initialization noise. If None, uses deterministic initialization

    Returns
    -------
    timeseries : ndarray of shape (n_timesteps, 1)
        Mackey-Glass time series
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = None

    # Validate that tau/h is an integer to ensure correct delay alignment
    delay_steps = tau / h
    if abs(delay_steps - round(delay_steps)) > 1e-9:
        raise ValueError(
            f"tau/h must be an integer to correctly align the time delay. "
            f"Got tau={tau}, h={h}, tau/h={delay_steps:.6f}. "
            f"Please adjust tau or h so that tau/h is an integer."
        )

    # Initialize history buffer
    history_len = int(round(delay_steps)) + 1
    if rng is not None:
        # Add small random perturbations to initial history
        history = deque([x0 + rng.normal(0, 0.001) for _ in range(history_len)], maxlen=history_len)
    else:
        history = deque([x0] * history_len, maxlen=history_len)

    # Pre-allocate output
    timeseries = np.empty((n_timesteps,), dtype=np.float64)

    # Define Mackey-Glass derivative
    def mg_deriv(x_current, x_delayed):
        return a * x_delayed / (1 + x_delayed**n) - b * x_current

    # Euler integration (suitable for DDEs with moderate stiffness)
    for i in range(n_timesteps):
        x_current = history[-1]
        x_delayed = history[0]

        timeseries[i] = x_current

        # Euler step
        dx = mg_deriv(x_current, x_delayed)
        x_next = x_current + h * dx

        history.append(x_next)

    return timeseries.reshape(-1, 1)


def _check_sufficient_data(data, n_in, n_out, data_name):
    """
    Check if dataset has enough timesteps for sliding window.

    Parameters
    ----------
    data : ndarray
        Time series data
    n_in : int
        Input window size
    n_out : int
        Output window size
    data_name : str
        Name of dataset for error message

    Raises
    ------
    ValueError
        If data has insufficient timesteps
    """
    min_required = n_in + n_out
    if len(data) < min_required:
        raise ValueError(
            f"{data_name} has only {len(data)} timesteps but needs "
            f"at least {min_required} (n_in={n_in} + n_out={n_out}). "
            f"Increase n_samples or decrease train_fraction/val_fraction."
        )


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


def load(dataset_name, n_samples=5000, train_fraction=0.7, n_in=100, n_out=1, seed=None, val_fraction=None, standardize=False, **kwargs):
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
    val_fraction : float, optional
        Fraction of training data to use for validation (between 0 and 1).
        If None (default), no validation split is created.
        If set, the function will:
        - Split validation set BEFORE sliding window (prevents data leakage)
        - Automatically enable standardization (standardize=True)
        - Fit StandardScaler only on the final training set
        - Return 7 items: x_train, y_train, x_val, y_val, x_test, y_test, scaler
    standardize : bool, default=False
        Whether to standardize the data using StandardScaler.
        If True (and val_fraction=None), returns 5 items including the scaler.
        If val_fraction is set, standardization is automatically enabled.
        Standardization is recommended for reservoir computing as it:
        - Normalizes features to similar scales
        - Improves Ridge regression performance
        - Accelerates neural network training
    **kwargs : dict
        Additional parameters passed to the generator function:
        - For Lorenz: sigma, rho, beta, h (time step), x0 (initial condition)
        - For Mackey-Glass: tau, a, b, n, h (time step), x0 (initial value)

    Returns
    -------
    When standardize=False and val_fraction=None (default, backward compatible):
        x_train : ndarray of shape (n_train_samples, n_in, n_features)
            Training input windows
        y_train : ndarray of shape (n_train_samples, n_out, n_features)
            Training output windows (targets)
        x_test : ndarray of shape (n_test_samples, n_in, n_features)
            Test input windows
        y_test : ndarray of shape (n_test_samples, n_out, n_features)
            Test output windows (targets)

    When standardize=True and val_fraction=None:
        x_train : ndarray of shape (n_train_samples, n_in, n_features)
            Training input windows (standardized)
        y_train : ndarray of shape (n_train_samples, n_out, n_features)
            Training output windows (standardized)
        x_test : ndarray of shape (n_test_samples, n_in, n_features)
            Test input windows (standardized)
        y_test : ndarray of shape (n_test_samples, n_out, n_features)
            Test output windows (standardized)
        scaler : StandardScaler
            Fitted scaler object (fitted only on training data)

    When val_fraction is set (standardize automatically enabled):
        x_train : ndarray of shape (n_train_final_samples, n_in, n_features)
            Training input windows (standardized)
        y_train : ndarray of shape (n_train_final_samples, n_out, n_features)
            Training output windows (standardized)
        x_val : ndarray of shape (n_val_samples, n_in, n_features)
            Validation input windows (standardized)
        y_val : ndarray of shape (n_val_samples, n_out, n_features)
            Validation output windows (standardized)
        x_test : ndarray of shape (n_test_samples, n_in, n_features)
            Test input windows (standardized)
        y_test : ndarray of shape (n_test_samples, n_out, n_features)
            Test output windows (standardized)
        scaler : StandardScaler
            Fitted scaler object (fitted only on training data)

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
    - Data generation uses numerical integration:
      * Lorenz: scipy.integrate.solve_ivp for stable ODE integration
      * Mackey-Glass: Euler integration for delay differential equation
    - When val_fraction is used:
      * Validation split happens BEFORE sliding window to prevent data leakage
      * StandardScaler is fitted ONLY on final training data (not on validation)
      * This ensures validation set statistics don't leak into training preprocessing
      * All datasets (train/val/test) are standardized using the same scaler
    """
    # Normalize dataset name
    dataset_name = dataset_name.lower().replace('-', '_').replace(' ', '_')

    # Generate raw time series using custom integrators
    if dataset_name in ['lorentz69', 'lorenz', 'lorenz63']:
        # Extract Lorenz-specific parameters with defaults
        sigma = kwargs.get('sigma', 10.0)
        rho = kwargs.get('rho', 28.0)
        beta = kwargs.get('beta', 8.0/3.0)
        h = kwargs.get('h', 0.01)
        x0 = kwargs.get('x0', [1.0, 1.0, 1.0])

        raw_data = _generate_lorenz(
            n_timesteps=n_samples,
            sigma=sigma,
            rho=rho,
            beta=beta,
            h=h,
            x0=x0
        )

    elif dataset_name in ['mackey_glass', 'mackeyglass', 'mg']:
        # Extract Mackey-Glass-specific parameters with defaults
        tau = kwargs.get('tau', 17)
        a = kwargs.get('a', 0.2)
        b = kwargs.get('b', 0.1)
        n = kwargs.get('n', 10)
        h = kwargs.get('h', 1.0)
        x0 = kwargs.get('x0', 1.2)

        raw_data = _generate_mackey_glass(
            n_timesteps=n_samples,
            tau=tau,
            a=a,
            b=b,
            n=n,
            h=h,
            x0=x0,
            seed=seed
        )

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

    # Automatically enable standardization when val_fraction is set
    if val_fraction is not None:
        standardize = True

    # Mode 1: No standardization, no validation (backward compatible)
    if not standardize and val_fraction is None:
        # Check data sufficiency before sliding window
        _check_sufficient_data(train_data, n_in, n_out, "Training data")
        _check_sufficient_data(test_data, n_in, n_out, "Test data")

        x_train, y_train = _sliding_window(train_data, n_in=n_in, n_out=n_out)
        x_test, y_test = _sliding_window(test_data, n_in=n_in, n_out=n_out)
        return x_train, y_train, x_test, y_test

    # Mode 2: Standardization without validation split
    elif standardize and val_fraction is None:
        # Check data sufficiency before processing
        _check_sufficient_data(train_data, n_in, n_out, "Training data")
        _check_sufficient_data(test_data, n_in, n_out, "Test data")

        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(train_data)

        # Transform both datasets
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Create sliding windows
        x_train, y_train = _sliding_window(train_scaled, n_in=n_in, n_out=n_out)
        x_test, y_test = _sliding_window(test_scaled, n_in=n_in, n_out=n_out)

        return x_train, y_train, x_test, y_test, scaler

    # Mode 3: With validation set and standardization (strict no-leakage)
    else:
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

        # Split validation from training BEFORE sliding window
        n_train_final_timesteps = int(len(train_data) * (1 - val_fraction))
        train_final_data = train_data[:n_train_final_timesteps]
        val_data = train_data[n_train_final_timesteps:]

        # Check data sufficiency for all three splits
        _check_sufficient_data(train_final_data, n_in, n_out, "Final training data")
        _check_sufficient_data(val_data, n_in, n_out, "Validation data")
        _check_sufficient_data(test_data, n_in, n_out, "Test data")

        # Fit scaler ONLY on final training data (no leakage)
        scaler = StandardScaler()
        scaler.fit(train_final_data)

        # Transform all three datasets
        train_final_scaled = scaler.transform(train_final_data)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)

        # Create sliding windows on scaled data
        x_train, y_train = _sliding_window(train_final_scaled, n_in=n_in, n_out=n_out)
        x_val, y_val = _sliding_window(val_scaled, n_in=n_in, n_out=n_out)
        x_test, y_test = _sliding_window(test_scaled, n_in=n_in, n_out=n_out)

        return x_train, y_train, x_val, y_val, x_test, y_test, scaler
