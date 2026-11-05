"""
Data checks: fit, predict and evaluate reject data that does not fit the model with a message that
says what is wrong (they used to fail with a broadcast or index error, return NaN or the wrong
shape, or train a model with other outputs than the layer declares).
"""

import numpy as np
import pytest

from helpers import build_rc
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

T, WARMUP = 50, 5  # inputs have T time steps, the targets T - WARMUP


def _model():
    return build_rc(T=T, K=T - WARMUP, nodes=10, density=0.3, seed=0, layer_seed=0, discard_transients=WARMUP)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(2, T, 2)), rng.normal(size=(2, T - WARMUP, 1))


@pytest.fixture
def fitted(data):
    model = _model()
    model.fit(*data)
    return model


# ------------------------------------------------------------------ type, dimension, length


@pytest.mark.parametrize(
    "x, y, error, match",
    [
        ([[1.0]], np.zeros((8, 30, 1)), TypeError, "numpy arrays"),
        (np.zeros((8, 30, 2)), [[1.0]], TypeError, "numpy arrays"),
        (np.zeros((8, 30)), np.zeros((8, 30, 1)), ValueError, "3 dimensions"),
        (np.zeros((8, 30, 2)), np.zeros((8, 30)), ValueError, "3 dimensions"),
        (np.zeros((8, 30, 2)), np.zeros((7, 30, 1)), ValueError, "same number of samples"),
        (np.zeros((8, 10, 2)), np.zeros((8, 30, 1)), ValueError, "at least as many timesteps"),
    ],
)
def test_fit_rejects_data_of_the_wrong_type_or_shape(x, y, error, match):
    """x and y must be 3D numpy arrays (samples, time, channels) with matching sample counts."""
    with pytest.raises(error, match=match):
        build_rc(T=30, nodes=30, seed=1, layer_seed=2).fit(x, y)


def test_fit_rejects_a_non_boolean_store_states_flag(data):
    with pytest.raises(TypeError, match="store_states"):
        _model().fit(*data, store_states="yes")


def test_sequences_not_longer_than_the_warm_up_are_rejected(fitted, data):
    """After the warm-up of 5 steps at least one state has to be left to predict from."""
    x = data[0]
    with pytest.raises(ValueError, match="warm-up is 5"):
        fitted.predict(x[:, :WARMUP])
    assert fitted.predict(x[:, : WARMUP + 1]).shape == (2, 1, 1)


# -------------------------------------------------------------- finite values and channels


@pytest.mark.parametrize("which", ["x", "y"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_non_finite_values_with_their_position(data, which, bad):
    x, y = (a.copy() for a in data)
    {"x": x, "y": y}[which][1, 3, 0] = bad
    with pytest.raises(ValueError, match=rf"{which} contains 1 NaN or infinite.*\(1, 3, 0\)"):
        _model().fit(x, y)


def test_predict_and_evaluate_reject_non_finite_inputs(fitted, data):
    x, y = data
    x = x.copy()
    x[0, 7, 1] = np.nan
    with pytest.raises(ValueError, match="x contains 1 NaN"):
        fitted.predict(x)
    with pytest.raises(ValueError, match="x contains 1 NaN"):
        fitted.evaluate(x, y)


def test_wrong_number_of_input_channels_is_named(fitted, data):
    x3 = np.random.default_rng(1).normal(size=(2, T, 3))
    with pytest.raises(ValueError, match="x has 3 input channels.*the model has 2"):
        fitted.predict(x3)
    with pytest.raises(ValueError, match="x has 3 input channels.*the model has 2"):
        fitted.fit(x3, data[1])


def test_wrong_number_of_target_channels_is_named(data):
    y2 = np.random.default_rng(1).normal(size=(2, T - WARMUP, 2))
    with pytest.raises(ValueError, match="y has 2 target channels.*the model has 1"):
        _model().fit(data[0], y2)


# --------------------------------------------------------------------- order of the calls


def test_predict_before_fit_says_so(data):
    with pytest.raises(RuntimeError, match="not fitted"):
        _model().predict(data[0])


def test_fit_before_compile_asks_for_compile(data):
    model = build_rc(T=T, K=T - WARMUP, nodes=10, compiled=False)
    with pytest.raises(RuntimeError, match="compile"):
        model.fit(*data)
    with pytest.raises(RuntimeError, match="compile"):
        RC().fit(*data)  # not even layers


def test_valid_data_is_unaffected(fitted, data):
    """The checks do not change the result for correct data."""
    assert fitted.predict(data[0]).shape == (2, T - WARMUP, 1)
    assert np.isfinite(fitted.evaluate(data[0], data[1])).all()
