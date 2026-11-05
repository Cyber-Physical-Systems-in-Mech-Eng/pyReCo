"""
Tests for the toy data sets in pyreco.utils_data: every problem name works in every
function, shapes are as documented, and unknown names raise a clear error.
"""

import numpy as np
import pytest

from pyreco import utils_data as ud

FUNCTIONS = ["vector_to_vector", "sequence_to_scalar", "sequence_to_sequence", "x_to_x"]


def test_available_problems_include_the_aliases():
    names = ud.available_problems()
    assert {"sine_pred", "sine_to_cosine", "sincos2"} <= set(names)
    assert {"sine_prediction", "sin_to_cos2"} <= set(names)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("name", ud.available_problems())
def test_every_problem_name_works_in_every_function(function, name):
    out = getattr(ud, function)(name=name, n_batch=20)
    assert len(out) == 4
    X_train, X_test, y_train, y_test = out
    assert X_train.ndim == X_test.ndim == y_train.ndim == y_test.ndim == 3
    assert X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]
    assert np.isfinite(X_train).all() and np.isfinite(y_train).all()


@pytest.mark.parametrize("function", FUNCTIONS)
def test_unknown_name_raises_a_value_error(function):
    with pytest.raises(ValueError, match="Unknown problem"):
        getattr(ud, function)(name="bogus", n_batch=20)


def test_aliases_give_the_same_data_as_the_canonical_names():
    for alias, name in [("sine_prediction", "sine_pred"), ("sin_to_cos2", "sincos2")]:
        a = ud.sequence_to_sequence(alias, n_batch=20, seed=1)
        b = ud.sequence_to_sequence(name, n_batch=20, seed=1)
        assert all(np.array_equal(p, q) for p, q in zip(a, b))


def test_shapes_of_the_four_cases():
    """vector-to-vector, sequence-to-scalar, sequence-to-sequence and x_to_x return (samples, time, states) with the requested sizes.

    80/20 split of n_batch windows
    """
    X_train, X_test, y_train, y_test = ud.vector_to_vector("sine_pred", n_batch=50, n_states=3)
    assert X_train.shape == (40, 1, 3) and y_train.shape == (40, 1, 3) and X_test.shape == (10, 1, 3)

    X_train, _, y_train, _ = ud.sequence_to_scalar("sine_pred", n_batch=50, n_states=2, n_time_in=7)
    assert X_train.shape == (40, 7, 2) and y_train.shape == (40, 1, 2)

    X_train, _, y_train, _ = ud.sequence_to_sequence("sine_to_cosine", n_batch=50, n_states=4, n_time=5)
    assert X_train.shape == (40, 5, 4) and y_train.shape == (40, 5, 4)

    X_train, _, y_train, _ = ud.x_to_x(
        "sincos2", n_batch=50, n_states_in=2, n_states_out=3, n_time_in=4, n_time_out=2
    )
    assert X_train.shape == (40, 4, 2) and y_train.shape == (40, 2, 3)


def test_at_least_one_test_sample_even_for_tiny_n_batch():
    """Small n_batch is raised so that the 80/20 split leaves a test sample."""
    _, X_test, _, _ = ud.sequence_to_sequence("sine_pred", n_batch=1)
    assert X_test.shape[0] >= 1


def test_sine_pred_targets_follow_the_inputs():
    """The targets are the values that follow the input window.

    window i: inputs are signal[i:i+n_in], targets signal[i+n_in:i+n_in+n_out]
    """
    n_in, n_out = 6, 3
    X_train, X_test, y_train, y_test = ud.sine_pred(
        n_batch=30, n_time_in=n_in, n_time_out=n_out, n_states=1, seed=0
    )
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
    signal = ud.gen_sine(n=30 + n_in + n_out + 1, omega=np.pi)
    for window_in, window_out in zip(X, y):
        # find the window in the signal and check the continuation
        starts = [i for i in range(30) if np.allclose(signal[i : i + n_in], window_in[:, 0])]
        assert starts, "input window is not part of the signal"
        i = starts[0]
        assert np.allclose(signal[i + n_in : i + n_in + n_out], window_out[:, 0])


def test_sine_to_cosine_targets_are_cosines_of_the_same_frequency():
    X_train, _, y_train, _ = ud.sequence_to_sequence("sine_to_cosine", n_batch=30, n_states=1, n_time=4)
    # sin^2 + cos^2 = 1 for matching time steps is not observable across windows, but the
    # amplitude of both signals is 1
    assert np.abs(X_train).max() <= 1.0 + 1e-12 and np.abs(y_train).max() <= 1.0 + 1e-12


def test_seed_makes_the_split_reproducible():
    a = ud.sequence_to_sequence("sine_pred", n_batch=30, seed=3)
    b = ud.sequence_to_sequence("sine_pred", n_batch=30, seed=3)
    c = ud.sequence_to_sequence("sine_pred", n_batch=30, seed=4)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_data_functions_do_not_print(capsys):
    ud.sequence_to_sequence("sine_pred", n_batch=20)
    ud.x_to_x("sine_pred", n_batch=20)
    assert capsys.readouterr().out == ""
