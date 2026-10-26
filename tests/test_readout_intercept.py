"""
Tests for the readout intercept: the readout is y = R W + b, with b fitted by centering
the reservoir states and the targets (the intercept is not penalized by the ridge term).
"""

import copy

import numpy as np

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer


def _data(n_out=1, n_batch=6, n_time=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_batch, n_time, 2))
    y = np.concatenate([np.tanh(x[:, :, :1]) + 0.3 * x[:, :, 1:2]] * n_out, axis=-1)
    y = y * np.linspace(1.0, 2.0, n_out)
    return x, y


def _model(n_out=1) -> RC:
    model = RC()
    model.add(InputLayer(input_shape=(40, 2)))
    model.add(RandomReservoirLayer(nodes=40, density=0.3))
    model.add(ReadoutLayer(output_shape=(40, n_out)))
    model.compile()
    return model


def test_bias_is_fitted_with_one_entry_per_output():
    for n_out in (1, 3):
        x, y = _data(n_out)
        model = _model(n_out)
        model.fit(x, y)
        assert model.readout_layer.bias.shape == (n_out,)
        assert model.readout_layer.weights.shape == (len(model.readout_layer.readout_nodes), n_out)


def test_shifting_the_targets_shifts_the_predictions_exactly():
    # an offset in the targets is absorbed by the intercept and not by the weights
    x, y = _data()
    m1 = _model()
    m2 = copy.deepcopy(m1)
    m1.fit(x, y)
    m2.fit(x, y + 100.0)
    assert np.allclose(m2.predict(x), m1.predict(x) + 100.0)
    assert np.allclose(m2.readout_layer.weights, m1.readout_layer.weights)
    assert np.allclose(m2.readout_layer.bias, m1.readout_layer.bias + 100.0)


def test_large_offset_is_fitted_well():
    x, y = _data()
    model = _model()
    model.fit(x, y + 1000.0)
    resid = model.predict(x) - (y + 1000.0)
    assert np.mean(resid**2) < np.var(y)


def test_training_predictions_reproduce_the_target_mean():
    # property of least squares with an intercept (no transients discarded)
    x, y = _data(n_out=2)
    model = _model(n_out=2)
    model.fit(x, y)
    assert np.allclose(model.predict(x).mean(axis=(0, 1)), y.mean(axis=(0, 1)))
