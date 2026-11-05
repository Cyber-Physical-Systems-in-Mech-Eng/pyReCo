r"""
Toy problems for quick tests of reservoir computers (periodic signals).

The shapes are always inputs ``X`` of shape ``(n_batch, n_timesteps, n_states)`` and outputs ``y`` of shape
``(n_batch, n_timesteps, n_states)``. The task functions return ``X_train, X_test, y_train, y_test`` from a
random 80/20 split of overlapping windows of one signal. The generators of
:mod:`pyreco.datasets` (``generate_sine``, ``generate_sine_to_cosine``, ``generate_sincos2``) are the
successors with the same interface as the other benchmark systems.

The signals are sines and cosines of the form :math:`\sin(\omega t_i)` on the grid
:math:`t_i = \frac{i}{n-1}\, \frac{n \omega}{50}`, so the phase grows like :math:`\omega^2` (the step between
samples is :math:`\omega^2 / 50` for large :math:`n`).
"""

import numbers

import numpy as np

from pyreco.utils_random import make_rng

# Chaotic benchmark systems (Lorenz, Mackey-Glass) are in pyreco.datasets, as are the successors of the
# toy problems below (generate_sine, generate_sine_to_cosine, generate_sincos2 / load('sincos2')): the same
# call signature as the other benchmarks, chronological split. The functions here keep their interface
# (random split, X_train, X_test, y_train, y_test).


def gen_sine(n=10, omega=np.pi):
    r"""
    Sine :math:`\sin(\omega t_i)` on ``n`` points :math:`t_i \in [0, n \omega / 50]`.

    Parameters
    ----------
    n : int, default=10
        Number of samples.
    omega : float, default=pi
        Frequency parameter :math:`\omega` (it scales both the grid and the argument).

    Returns
    -------
    ndarray of shape (n,)
    """
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.sin(omega * t)


def gen_cos(n=10, omega=np.pi):
    r"""Cosine :math:`\cos(\omega t_i)` on the grid of :func:`gen_sine`; returns an array of shape ``(n,)``."""
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.cos(omega * t)


def gen_sincos(n=10, omega=np.pi, a_sc=1, b_sc=0.25, P_sc=3):
    r"""
    Nonlinear combination of sine and cosine on the grid of :func:`gen_sine`.

    .. math::

        y_i = a \sin^P(\omega t_i) + b \cos^P(\omega t_i)

    Parameters
    ----------
    n : int, default=10
        Number of samples.
    omega : float, default=pi
        Frequency parameter.
    a_sc, b_sc : float
        Coefficients :math:`a` and :math:`b`.
    P_sc : int
        Exponent :math:`P`.

    Returns
    -------
    ndarray of shape (n,)
    """
    sine_wave = gen_sine(n, omega)
    cos_wave = gen_cos(n, omega)
    return a_sc * sine_wave**P_sc + b_sc * cos_wave**P_sc


def split_sequence(signal, n_batch, n_time_in, n_time_out):
    r"""
    Cut a signal into overlapping windows: the input of window :math:`i` is :math:`s_{i:i+T_\mathrm{in}}` and
    the target the :math:`T_\mathrm{out}` values that follow.

    Parameters
    ----------
    signal : ndarray of shape (n_timesteps, n_states)
        The signal.
    n_batch : int
        Number of windows (consecutive start indices).
    n_time_in, n_time_out : int
        Length of the input and the target window.

    Returns
    -------
    x : ndarray of shape (n_batch, n_time_in, n_states)
    y : ndarray of shape (n_batch, n_time_out, n_states)
    """

    # convert into inputs (function at last n_time_in points) and outputs (function at next n_time_out points)
    x, y = [], []
    for i in range(n_batch):
        idx_in_end = i + n_time_in
        idx_out_end = idx_in_end + n_time_out
        x.append(signal[i:idx_in_end, :])  # last n_timestep_in points
        y.append(signal[idx_in_end:idx_out_end, :])  # next n_timestep_out points
    x, y = np.array(x), np.array(y)

    return x, y


def train_test_split(x, y, seed=None):
    """
    Random train/test split of the first axis, 80 % of the samples for training.

    The samples are overlapping windows of one signal, so the split is not leakage-free (test windows share
    values with training windows); ``pyreco.datasets.load`` splits chronologically.

    Parameters
    ----------
    x, y : ndarray
        Inputs and targets with the same first dimension.
    seed : int, Generator or None
        Seed of the split; the global random state is not used.

    Returns
    -------
    x_train, x_test, y_train, y_test : ndarray
    """
    # train-test split: 80% randomly chosen samples for training, the rest for testing.
    # Note: the samples are overlapping windows of one signal, so the random split is not
    # leakage-free (test windows share values with training windows). Fine for toy data;
    # pyreco.datasets.load splits chronologically.
    # seed (int, Generator or None): seed of the random split, the global state is not used
    n = x.shape[0]

    ratio = 0.8

    split_idx = np.max([1, int(n * ratio)])

    shuffle_idx = make_rng(seed).choice(n, size=n, replace=False)
    train_idx, test_idx = shuffle_idx[: int(n * ratio)], shuffle_idx[int(n * ratio) :]

    # split data according to train/test split and return.
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def sine_pred(n_batch, n_time_in, n_time_out, n_states, seed=None):
    r"""
    Forecasting task: predict sines, state :math:`k` with the frequency parameter :math:`\pi + 0.314 k`.

    Parameters
    ----------
    n_batch : int
        Number of windows.
    n_time_in, n_time_out : int
        Length of the input and target window (single- and multi-step prediction).
    n_states : int
        Number of sines.
    seed : int, optional
        Seed of the train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Of shape ``(n_samples, n_time_in or n_time_out, n_states)``.
    """

    # we will create different signal frequencies for the different states
    signal, omega = [], np.pi
    for _ in range(n_states):
        signal.append(gen_sine(n=n_batch + (n_time_in + n_time_out) + 1, omega=omega))
        omega += 0.314

    # 2D array of shape [n_timesteps, n_states]
    signal = np.array(signal).transpose()

    # split into inputs and outputs
    x, y = split_sequence(signal, n_batch, n_time_in, n_time_out)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, seed=seed)

    return X_train, X_test, y_train, y_test


def sine_to_cosine(n_batch, n_time_in, n_time_out, n_states, seed=None):
    r"""
    Input/output task: sines as inputs, the cosines of the same frequency parameter as targets,
    :math:`y_k = \cos(\omega_k t) = \sin(\omega_k t + \pi/2)` with :math:`\omega_k = \pi + 0.314 k`.

    As in :func:`sine_pred`, the target window is the signal on the time steps that follow the input window.
    Arguments and return values as in :func:`sine_pred`.
    """
    total_time = n_batch + n_time_in + n_time_out
    x, y = [], []
    omega = np.pi
    for _ in range(n_states):
        x.append(gen_sine(n=total_time, omega=omega))
        y.append(gen_cos(n=total_time, omega=omega))
        omega += 0.314

    # Convert to 2D arrays of shape [n_timesteps, n_states]
    x = np.array(x).T
    y = np.array(y).T

    # Split into sequences
    x = split_sequence(x, n_batch, n_time_in, n_time_out)
    y = split_sequence(y, n_batch, n_time_in, n_time_out)

    # Unpack the tuples returned by split_sequence
    x_input, _ = x
    _, y_output = y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x_input, y_output, seed=seed)

    return X_train, X_test, y_train, y_test


def sincos2(n_batch, n_time_in, n_time_out, n_states, seed=None):
    r"""
    Input/output task: sines as inputs and :math:`y_k = \sin^3(\omega_k t) + 0.25 \cos^3(\omega_k t)` as
    targets, with :math:`\omega_k = 1 + 0.314 k`; the target window follows the input window as in
    :func:`sine_pred`. Arguments and return values as in :func:`sine_pred`.
    """
    total_time = n_batch + n_time_in + n_time_out
    x, y = [], []
    omega = 1  # As specified in the document

    for _ in range(n_states):
        x.append(gen_sine(n=total_time, omega=omega))
        y.append(gen_sincos(n=total_time, omega=omega))
        omega += 0.314  # Increment omega for each state, as in the original function

    # Convert to 2D arrays of shape [n_timesteps, n_states]
    x = np.array(x).T
    y = np.array(y).T

    # Split into sequences
    x = split_sequence(x, n_batch, n_time_in, n_time_out)
    y = split_sequence(y, n_batch, n_time_in, n_time_out)

    # Unpack the tuples returned by split_sequence
    x_input, _ = x
    _, y_output = y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x_input, y_output, seed=seed)

    return X_train, X_test, y_train, y_test


# Available toy problems. The same names work in all functions below; the alternative names
# are accepted for backward compatibility.
_PROBLEMS = {
    "sine_pred": sine_pred,  # predict a sine signal (multi-state: different frequencies)
    "sine_to_cosine": sine_to_cosine,  # map sines to cosines
    "sincos2": sincos2,  # map a sine to a_sc*sin^3 + b_sc*cos^3
}
_ALIASES = {"sine_prediction": "sine_pred", "sin_to_cos2": "sincos2"}


def _check_int(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def available_problems() -> list:
    """Names of the toy problems accepted by the data functions in this module."""
    return sorted(_PROBLEMS) + sorted(_ALIASES)


def _generate(name, n_batch, n_states, n_time_in, n_time_out, seed):
    key = _ALIASES.get(name, name)
    if key not in _PROBLEMS:
        raise ValueError(f"Unknown problem '{name}'. Available: {available_problems()}")

    # make sure to have at least 1 testing sample
    n_batch = int(np.max([n_batch, 2]))

    return _PROBLEMS[key](
        n_batch=n_batch,
        n_time_in=n_time_in,
        n_time_out=n_time_out,
        n_states=n_states,
        seed=seed,
    )


"""
CASE 1: Vector to Vector
"""


def vector_to_vector(name, n_batch: int = 50, n_states=1, seed=None):
    """
    Case 1, vector to vector: one input time step, one target time step.

    Parameters
    ----------
    name : str
        Problem name, see :func:`available_problems`.
    n_batch : int, default=50
        Number of samples (at least 2).
    n_states : int, default=1
        Number of states.
    seed : int, optional
        Seed of the train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Of shape ``(n_samples, 1, n_states)``.
    """
    _check_int("n_states", n_states)
    return _generate(name, n_batch, n_states, n_time_in=1, n_time_out=1, seed=seed)


"""
CASE 2: Sequence to scalar
"""


def sequence_to_scalar(name, n_batch: int = 50, n_states=1, n_time_in=2, seed=None):
    """
    Case 2, sequence to scalar: ``n_time_in`` input time steps, one target time step.

    Arguments as in :func:`vector_to_vector`, plus ``n_time_in``. Returns arrays of shape
    ``(n_samples, n_time_in, n_states)`` (inputs) and ``(n_samples, 1, n_states)`` (targets).
    """
    _check_int("n_states", n_states)
    _check_int("n_time_in", n_time_in)
    return _generate(name, n_batch, n_states, n_time_in=n_time_in, n_time_out=1, seed=seed)


"""
CASE 3: Sequence to sequence
"""


def sequence_to_sequence(name, n_batch: int = 50, n_states: int = 2, n_time: int = 3, seed=None):
    """
    Case 3, sequence to sequence: ``n_time`` input and ``n_time`` target time steps.

    Arguments as in :func:`vector_to_vector`, plus ``n_time``. Returns arrays of shape
    ``(n_samples, n_time, n_states)``.
    """
    return _generate(name, n_batch, n_states, n_time_in=n_time, n_time_out=n_time, seed=seed)


"""
CASE 4: free choice of the number of input and output states and time steps
"""


def x_to_x(
    name,
    n_batch: int = 50,
    n_states_in: int = 2,
    n_states_out: int = 2,
    n_time_in: int = 1,
    n_time_out: int = 1,
    seed=None,
):
    """
    Case 4, free choice of the number of input and output states and time steps.

    The problem is generated with ``max(n_states_in, n_states_out)`` states and the first
    ``n_states_in`` / ``n_states_out`` states are used as inputs / targets.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Inputs of shape ``(n_samples, n_time_in, n_states_in)``, targets of shape
        ``(n_samples, n_time_out, n_states_out)``.
    """
    # generate with as many states as needed, then cut the input and output states
    n_states = int(np.max([n_states_in, n_states_out]))
    X_train, X_test, y_train, y_test = _generate(
        name, n_batch, n_states, n_time_in=n_time_in, n_time_out=n_time_out, seed=seed
    )
    X_train, X_test = X_train[:, :, :n_states_in], X_test[:, :, :n_states_in]
    y_train, y_test = y_train[:, :, :n_states_out], y_test[:, :, :n_states_out]
    return X_train, X_test, y_train, y_test
