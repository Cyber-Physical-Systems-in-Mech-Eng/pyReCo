"""
fit, predict and evaluate reject data that does not fit the model with a message that says what is
wrong (they used to fail with a broadcast or index error, or to return NaN or the wrong shape).
"""

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

T, WARMUP = 50, 5


def _model():
    model = RC(seed=0)
    model.add(InputLayer(input_shape=(T, 2)))
    model.add(RandomReservoirLayer(nodes=10, density=0.3, seed=0))
    model.add(ReadoutLayer((T - WARMUP, 1)))
    model.compile(discard_transients=WARMUP)
    return model


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(2, T, 2)), rng.normal(size=(2, T - WARMUP, 1))


@pytest.fixture
def fitted(data):
    model = _model()
    model.fit(*data)
    return model


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


def test_sequences_not_longer_than_the_warm_up_are_rejected(fitted, data):
    x = data[0]
    with pytest.raises(ValueError, match="warm-up is 5"):
        fitted.predict(x[:, :WARMUP])
    assert fitted.predict(x[:, : WARMUP + 1]).shape == (2, 1, 1)


def test_prediction_before_fit_says_so(data):
    with pytest.raises(RuntimeError, match="not fitted"):
        _model().predict(data[0])


def test_valid_data_is_unaffected(fitted, data):
    assert fitted.predict(data[0]).shape == (2, T - WARMUP, 1)
    assert np.isfinite(fitted.evaluate(*data[:1], data[1])).all()
