"""
Provides testing data sets.

SHAPES will always be:
inputs X: [n_batch, n_timesteps, n_states]
outputs y: [n_batch, n_timesteps, n_states]

"""

import numbers

import numpy as np

from pyreco.utils_random import make_rng

# Chaotic benchmark systems (Lorenz, Mackey-Glass) are in pyreco.datasets, as are the successors of the
# toy problems below (generate_sine, generate_sine_to_cosine, generate_sincos2 / load('sincos2')): the same
# call signature as the other benchmarks, chronological split. The functions here keep their interface
# (random split, X_train, X_test, y_train, y_test).


def gen_sine(n=10, omega=np.pi):
    # generates a sine with a given frequency: n points on t in [0, n/50 * omega]
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.sin(omega * t)


def gen_cos(n=10, omega=np.pi):
    # generates a cosine with a given frequency: n points on t in [0, n/50 * omega]
    t = np.linspace(start=0, stop=n / 50 * omega, num=n, endpoint=True)
    return np.cos(omega * t)


def gen_sincos(n=10, omega=np.pi, a_sc=1, b_sc=0.25, P_sc=3):
    # generates a sequence of a_sc*sin(omega*t)^P_sc + b_sc*cos(omega*t)^P_sc
    # using gen_sine and gen_cos functions
    sine_wave = gen_sine(n, omega)
    cos_wave = gen_cos(n, omega)
    return a_sc * sine_wave**P_sc + b_sc * cos_wave**P_sc


def split_sequence(signal, n_batch, n_time_in, n_time_out):
    # expects [n_timesteps, n_states] sequence

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
    # predict a sine signal. Single- and multi-step prediction supported

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
    # Generate sine input and cosine output signals
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
    # Generate sine input and SinCos-2 output signals
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
    _check_int("n_states", n_states)
    return _generate(name, n_batch, n_states, n_time_in=1, n_time_out=1, seed=seed)


"""
CASE 2: Sequence to scalar
"""


def sequence_to_scalar(name, n_batch: int = 50, n_states=1, n_time_in=2, seed=None):
    _check_int("n_states", n_states)
    _check_int("n_time_in", n_time_in)
    return _generate(name, n_batch, n_states, n_time_in=n_time_in, n_time_out=1, seed=seed)


"""
CASE 3: Sequence to sequence
"""


def sequence_to_sequence(name, n_batch: int = 50, n_states: int = 2, n_time: int = 3, seed=None):
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
    # generate with as many states as needed, then cut the input and output states
    n_states = int(np.max([n_states_in, n_states_out]))
    X_train, X_test, y_train, y_test = _generate(
        name, n_batch, n_states, n_time_in=n_time_in, n_time_out=n_time_out, seed=seed
    )
    X_train, X_test = X_train[:, :, :n_states_in], X_test[:, :, :n_states_in]
    y_train, y_test = y_train[:, :, :n_states_out], y_test[:, :, :n_states_out]
    return X_train, X_test, y_train, y_test
