"""
Standard chaotic system datasets for reservoir computing.

This module provides time series data from chaotic dynamical systems,
formatted for direct use with PyReCo reservoir computers.

Every system has a generator with the same signature,

    generate_<system>(n_timesteps, *, <system parameters>, washout=0, seed=None)
        -> ndarray of shape (n_timesteps, n_features)

(`generate_lorenz`, `generate_mackey_glass`, `generate_narma`), and `generate(name, n_timesteps,
...)` calls them by name. They return the raw series, `load` cuts it into training and test windows.
Continuous systems take their sampling interval as `dt`.

All data generators use numerical integration of the governing equations.
Lorenz system uses scipy.integrate.solve_ivp for stable ODE integration.
Mackey-Glass uses a second-order (Heun) scheme with a fine internal step for the delay
differential equation, sampled at the requested interval.
NARMA5 and NARMA10 are nonlinear system-identification benchmarks (random input, the target is
the output of a nonlinear autoregressive moving-average system); they test memory and
nonlinearity of a reservoir rather than forecasting.

Note: the generated series start from the initial condition of the system and include
the start-up transient, i.e. the first samples are not yet on the attractor. Use `washout` to
drop them (the reservoir's `discard_transients` is the warm-up of the reservoir state, a different
thing).
"""

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

from pyreco.utils_random import make_rng

__all__ = ["generate", "generate_lorenz", "generate_mackey_glass", "generate_narma", "load"]


#: standard deviation of the perturbation of the initial condition that a `seed` draws for the
#: deterministic systems (Lorenz, Mackey-Glass)
INITIAL_PERTURBATION = 1e-3


def _check_count(name, value, minimum):
    """An integer (not a bool) of at least `minimum`."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _check_positive(name, value):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def generate_lorenz(
    n_timesteps, *, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, x0=None, washout=0, seed=None
):
    """
    Generate a trajectory of the Lorenz-63 system.

        dx/dt = sigma (y - x)
        dy/dt = x (rho - z) - y
        dz/dt = x y - beta z

    The system is integrated with `scipy.integrate.solve_ivp` (adaptive Runge-Kutta, default
    tolerances) and sampled every `dt` time units.

    Parameters
    ----------
    n_timesteps : int
        Number of samples to return.
    sigma, rho, beta : float
        Prandtl number, Rayleigh number and geometric factor of the system.
    dt : float, default=0.01
        Sampling interval of the returned series, in time units.
    x0 : array-like of shape (3,), optional
        Initial condition (x, y, z), default (1, 1, 1).
    washout : int, default=0
        Number of leading samples that are generated and dropped: the series starts at the
        initial condition, which is not on the attractor, so the first samples are a transient.
    seed : int, optional
        If None, the series is deterministic. If given, the initial condition is perturbed by
        normal noise (std `INITIAL_PERTURBATION`), so different seeds give different
        trajectories on the same attractor.

    Returns
    -------
    trajectory : ndarray of shape (n_timesteps, 3)
        Columns x, y, z.
    """
    n_timesteps = _check_count("n_timesteps", n_timesteps, 1)
    washout = _check_count("washout", washout, 0)
    dt = _check_positive("dt", dt)
    x0 = np.array([1.0, 1.0, 1.0] if x0 is None else x0, dtype=np.float64)
    if x0.shape != (3,) or not np.isfinite(x0).all():
        raise ValueError(f"x0 must be three finite numbers (x, y, z), got {x0!r}")
    if seed is not None:
        x0 = x0 + make_rng(seed).normal(0.0, INITIAL_PERTURBATION, size=3)

    def lorenz_deriv(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    n_total = n_timesteps + washout
    t_eval = np.arange(max(n_total, 2)) * dt  # solve_ivp needs a non-empty time span
    sol = solve_ivp(lorenz_deriv, (0.0, t_eval[-1]), x0, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return np.ascontiguousarray(sol.y.T[washout:n_total])


def generate_mackey_glass(
    n_timesteps, *, tau=17, a=0.2, b=0.1, n=10, dt=1.0, dt_int=0.1, x0=1.2, washout=0, seed=None
):
    """
    Generate a Mackey-Glass time series (delay differential equation).

        dx/dt = a x(t - tau) / (1 + x(t - tau)^n) - b x(t)

    The equation is integrated with Heun's method (explicit trapezoidal rule) using the internal
    step `dt_int` and the series is sampled every `dt` time units. Integrating with a coarse step
    (e.g. an Euler step of 1.0) gives a visibly different trajectory (deviations of up to 0.8 on a
    series with range 0.4 to 1.3 for tau=17).

    Parameters
    ----------
    n_timesteps : int
        Number of samples to return.
    tau : float, default=17
        Time delay. The system is chaotic for tau >= 17 (weakly at 17).
    a, b : float
        Production and degradation rate.
    n : float, default=10
        Hill coefficient (nonlinearity).
    dt : float, default=1.0
        Sampling interval of the returned series, in time units.
    dt_int : float, default=0.1
        Internal integration step. `tau / dt_int` and `dt / dt_int` must be integers.
    x0 : float, default=1.2
        Initial condition (constant history on [-tau, 0]).
    washout : int, default=0
        Number of leading samples that are generated and dropped (the series starts from the
        constant history, which is not on the attractor).
    seed : int, optional
        If None, the series is deterministic. If given, the initial history is perturbed by
        normal noise (std `INITIAL_PERTURBATION`), so different seeds give different
        trajectories on the same attractor.

    Returns
    -------
    timeseries : ndarray of shape (n_timesteps, 1)
    """
    n_timesteps = _check_count("n_timesteps", n_timesteps, 1)
    washout = _check_count("washout", washout, 0)
    tau, dt, dt_int = (_check_positive(k, v) for k, v in (("tau", tau), ("dt", dt), ("dt_int", dt_int)))
    if not np.isscalar(x0) or not np.isfinite(x0):
        raise ValueError(f"x0 must be a finite number, got {x0!r}")

    # the delay and the sampling interval have to be multiples of the internal step
    delay_steps = tau / dt_int
    sub_steps = dt / dt_int
    for name, quotient in (("tau/dt_int", delay_steps), ("dt/dt_int", sub_steps)):
        if abs(quotient - round(quotient)) > 1e-9:
            raise ValueError(
                f"{name} must be an integer to correctly align the time delay. "
                f"Got tau={tau}, dt={dt}, dt_int={dt_int}, {name}={quotient:.6f}."
            )
    delay_steps, sub_steps = int(round(delay_steps)), int(round(sub_steps))

    # History ring buffer holding x(t - tau), ..., x(t); the slot at `oldest` is x(t - tau)
    history = np.full(delay_steps + 1, float(x0))
    if seed is not None:
        history = history + make_rng(seed).normal(0, INITIAL_PERTURBATION, size=delay_steps + 1)
    oldest = 0
    x_current = history[-1]

    def mg_deriv(x_now, x_delayed):
        return a * x_delayed / (1 + x_delayed**n) - b * x_now

    n_total = n_timesteps + washout
    timeseries = np.empty((n_total,), dtype=np.float64)
    for i in range(n_total * sub_steps):
        if i % sub_steps == 0:
            timeseries[i // sub_steps] = x_current

        # Heun step; the delayed value at the end of the step is the next slot
        x_delayed = history[oldest]
        x_delayed_next = history[(oldest + 1) % (delay_steps + 1)]
        k1 = mg_deriv(x_current, x_delayed)
        k2 = mg_deriv(x_current + dt_int * k1, x_delayed_next)
        x_current = x_current + 0.5 * dt_int * (k1 + k2)

        # the oldest slot is replaced by the new value
        history[oldest] = x_current
        oldest = (oldest + 1) % (delay_steps + 1)

    return timeseries[washout:].reshape(-1, 1)


def generate_narma(
    n_timesteps,
    *,
    order=10,
    alpha=0.3,
    beta=0.05,
    gamma=1.5,
    delta=0.1,
    u_max=0.5,
    max_attempts=100,
    washout=0,
    seed=None,
):
    """
    Generate the NARMA-`order` system identification benchmark.

    The input u(t) is i.i.d. uniform on [0, u_max] and the system output follows

        y(t+1) = alpha*y(t) + beta*y(t)*sum_{i=0}^{order-1} y(t-i) + gamma*u(t-order+1)*u(t) + delta

    with y = 0 as initial condition. The standard benchmarks are NARMA10 (Jaeger) and NARMA5
    (order 5) with alpha=0.3, beta=0.05, gamma=1.5, delta=0.1 and u_max=0.5. The output depends
    on the last `order` inputs and on its own history, so the task needs memory and
    nonlinearity.

    The output can blow up for some input sequences (NARMA10 for a few percent of the
    sequences of 5000 steps, NARMA5 practically never). Such sequences are redrawn, up to
    `max_attempts` times.

    Parameters
    ----------
    n_timesteps : int
        Number of samples to return.
    order : int, default=10
        Order of the system (memory length of the input term and of the output sum).
    alpha, beta, gamma, delta, u_max : float
        Coefficients of the system and the upper bound of the input.
    max_attempts : int, default=100
        Number of redraws of the input sequence if the output diverges.
    washout : int, default=0
        Number of leading samples that are generated and dropped, because the system starts
        from the artificial initial condition y = 0 (the first `order` outputs are exactly 0).
        The benchmark loader `load` drops 100 by default.
    seed : int, optional
        Seed of the random input sequence. If None, the input is different on every call (the
        system itself has no free initial condition).

    Returns
    -------
    data : ndarray of shape (n_timesteps, 2)
        Column 0 is the input u(t), column 1 the target y(t+1): the system output right
        after the input u(t) has been applied. A reservoir state r(t+1) has seen the inputs
        u(0), ..., u(t), so it is the state that is paired with the target y(t+1).
    """
    n_timesteps = _check_count("n_timesteps", n_timesteps, 1)
    washout = _check_count("washout", washout, 0)
    if isinstance(order, bool) or int(order) != order or order < 2:
        raise ValueError(f"order must be an integer >= 2, got {order}")
    order = int(order)
    rng = make_rng(seed)
    n_total = n_timesteps + washout

    for _ in range(max_attempts):
        u = rng.uniform(0.0, u_max, size=n_total)
        y = np.zeros(n_total + 1)  # y[t] is the system output at time t
        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(order - 1, n_total):
                y[t + 1] = (
                    alpha * y[t]
                    + beta * y[t] * y[t - order + 1 : t + 1].sum()
                    + gamma * u[t - order + 1] * u[t]
                    + delta
                )
        if np.isfinite(y).all() and np.abs(y).max() < 10.0:
            return np.stack([u[washout:], y[1:][washout:]], axis=1)

    raise RuntimeError(
        f"NARMA{order} diverged for all {max_attempts} random input sequences. "
        "Use a smaller u_max or fewer samples."
    )


# name -> (generator, parameters that define the benchmark, samples that `load` drops by default)
_SYSTEMS = {
    "lorenz": (generate_lorenz, {}, 0),
    "mackey_glass": (generate_mackey_glass, {}, 0),
    "narma5": (generate_narma, {"order": 5}, 100),
    "narma10": (generate_narma, {"order": 10}, 100),
}
_ALIASES = {
    "lorentz69": "lorenz",
    "lorenz63": "lorenz",
    "mackeyglass": "mackey_glass",
    "mg": "mackey_glass",
    "narma_5": "narma5",
    "narma_10": "narma10",
}
# names of the parameters before the generators shared one convention
_RENAMED = {"h": "dt", "h_int": "dt_int"}


def _system(dataset_name):
    key = str(dataset_name).lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in _SYSTEMS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Supported: 'lorenz' (alias 'lorentz69', "
            f"'lorenz63'), 'mackey_glass' (alias 'mackeyglass', 'mg'), 'narma5', 'narma10'"
        )
    return key


def generate(dataset_name, n_timesteps, *, washout=None, seed=None, **parameters):
    """
    Generate a benchmark time series by name; the same call for every system.

    Parameters
    ----------
    dataset_name : str
        'lorenz', 'mackey_glass' or 'narma5' / 'narma10' (case and separators do not matter,
        aliases as in `load`).
    n_timesteps : int
        Number of samples to return.
    washout : int, optional
        Leading samples that are generated and dropped. Default: the recommended value of the
        system (100 for NARMA, whose output starts from the artificial value 0, otherwise 0).
    seed : int, optional
        Seed of the random input (NARMA) or of the perturbation of the initial condition (Lorenz,
        Mackey-Glass). Without a seed the Lorenz and Mackey-Glass series are deterministic and
        the NARMA input is different on every call.
    **parameters
        Parameters of the system, see `generate_lorenz`, `generate_mackey_glass` and
        `generate_narma` (e.g. `dt`, `x0`, `tau`). An unknown parameter raises a `TypeError`.

    Returns
    -------
    data : ndarray of shape (n_timesteps, n_features)
        1 feature for Mackey-Glass, 3 (x, y, z) for Lorenz, 2 (input u, target y) for NARMA.
    """
    function, fixed, default_washout = _SYSTEMS[_system(dataset_name)]
    for old, new in _RENAMED.items():
        if old in parameters:
            warnings.warn(f"'{old}' was renamed to '{new}'", DeprecationWarning, stacklevel=2)
            parameters.setdefault(new, parameters.pop(old))
    clash = sorted(set(parameters) & set(fixed))
    if clash:
        raise ValueError(f"{clash} is fixed by the dataset '{dataset_name}'")
    return function(
        n_timesteps,
        washout=default_washout if washout is None else washout,
        seed=seed,
        **fixed,
        **parameters,
    )


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


def _sliding_window_io(inputs, targets, n_in, n_out=1):
    """
    Create windows for input/output (system identification) tasks.

    Parameters
    ----------
    inputs : ndarray of shape (n_timesteps, n_in_features)
    targets : ndarray of shape (n_timesteps, n_out_features)
        targets[t] is the target that belongs to the state after inputs[t] was applied
    n_in : int
        Number of time steps in the input window
    n_out : int, default=1
        Number of target time steps. They are the targets of the last n_out input steps,
        i.e. the target windows end together with the input windows (n_out <= n_in).

    Returns
    -------
    X : ndarray of shape (n_samples, n_in, n_in_features)
    y : ndarray of shape (n_samples, n_out, n_out_features)
    """
    n_timesteps = len(inputs)
    n_samples = n_timesteps - n_in + 1
    if n_samples < 1:
        raise ValueError(
            f"Not enough data: need at least {n_in} timesteps, got {n_timesteps}"
        )

    X = np.stack([inputs[i : i + n_in] for i in range(n_samples)], axis=0)
    y = np.stack([targets[i + n_in - n_out : i + n_in] for i in range(n_samples)], axis=0)
    return X, y


def load(
    dataset_name,
    n_samples=5000,
    train_fraction=0.7,
    n_in=100,
    n_out=1,
    seed=None,
    val_fraction=None,
    standardize=False,
    washout=None,
    **kwargs,
):
    """
    Load chaotic time series dataset with train/test split.

    This function generates data from chaotic dynamical systems by numerical integration
    and formats it for reservoir computing tasks using sliding windows.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Supported options:
        - 'lorentz69' or 'lorenz' or 'lorenz63': Lorenz 1963 attractor (3D system)
        - 'mackey_glass' or 'mackeyglass' or 'mg': Mackey-Glass delay system (1D system)
        - 'narma5' or 'narma10': NARMA system identification benchmarks (1D input, 1D
          target). The inputs are random, the target is the output of the NARMA system.
    n_samples : int, default=5000
        Number of time steps to generate from the chaotic system
    train_fraction : float, default=0.7
        Fraction of data to use for training (between 0 and 1)
    n_in : int, default=100
        Number of time steps in each input window
    n_out : int, default=1
        Number of time steps in each output window (prediction horizon)
    seed : int, optional
        Random seed for reproducibility. For Lorenz and Mackey-Glass it perturbs the initial
        condition slightly (seed=None: the deterministic series), for NARMA it draws the random
        input sequence (seed=None: a different sequence on every call).
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
    washout : int, optional
        Number of leading samples that are generated and dropped before the split. Default:
        100 for NARMA (the output starts from the artificial value 0), 0 otherwise.
    **kwargs : dict
        Parameters of the system, passed to `generate` (an unknown name raises a `TypeError`):
        - For Lorenz: sigma, rho, beta, dt (sampling interval), x0 (initial condition)
        - For Mackey-Glass: tau, a, b, n, dt (sampling interval), dt_int (internal
          integration step), x0 (initial value)
        - For NARMA: alpha, beta, gamma, delta, u_max, max_attempts

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
    - NARMA is an input/output task instead of a forecasting task: the input window is a
      window of the random input u, and the output window holds the n_out targets that
      belong to the last n_out input steps (n_out <= n_in). With n_out=1 the target is the
      system output right after the last input of the window. If the data is standardized,
      the returned scaler is fitted on the two columns [input, target].
    - Data is split chronologically (earlier data for training, later for testing)
    - Reproducible when seed is set (for Mackey-Glass)
    - Data generation uses numerical integration:
      * Lorenz: scipy.integrate.solve_ivp for stable ODE integration
      * Mackey-Glass: Heun integration with an internal step of 0.1 for the delay
        differential equation
    - The series include the start-up transient (no washout is discarded)
    - The return value depends on the options: 4 items (default), 5 items with
      standardize=True, 7 items with val_fraction (see Returns)
    - When val_fraction is used:
      * Validation split happens BEFORE sliding window to prevent data leakage
      * StandardScaler is fitted ONLY on final training data (not on validation)
      * This ensures validation set statistics don't leak into training preprocessing
      * All datasets (train/val/test) are standardized using the same scaler
    """
    name = _system(dataset_name)
    io_task = name.startswith("narma")
    if io_task and n_out > n_in:
        raise ValueError(
            f"For NARMA the number of output steps must not exceed the input steps, "
            f"got n_in={n_in}, n_out={n_out}."
        )
    raw_data = generate(name, n_samples, washout=washout, seed=seed, **kwargs)

    def _windows(data):
        # forecasting: the next n_out values of the series follow the input window;
        # input/output task: the targets belong to the last n_out steps of the window
        if io_task:
            return _sliding_window_io(data[:, :1], data[:, 1:], n_in=n_in, n_out=n_out)
        return _sliding_window(data, n_in=n_in, n_out=n_out)

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

        x_train, y_train = _windows(train_data)
        x_test, y_test = _windows(test_data)
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
        x_train, y_train = _windows(train_scaled)
        x_test, y_test = _windows(test_scaled)

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
        x_train, y_train = _windows(train_final_scaled)
        x_val, y_val = _windows(val_scaled)
        x_test, y_test = _windows(test_scaled)

        return x_train, y_train, x_val, y_val, x_test, y_test, scaler
