"""
Tests for the input preprocessing: per-channel z-scoring followed by input scaling.
The same transform must be applied in fit() and predict().
"""

import copy

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer


def _data(n_batch=6, n_time=40, n_in=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=[3.0, -2.0][:n_in], scale=[2.0, 0.5][:n_in], size=(n_batch, n_time, n_in))
    y = np.tanh(x[:, :, :1]) + 0.3 * x[:, :, 1:2] if n_in > 1 else np.tanh(x)
    return x, y


def _model(n_in=2, **compile_kwargs) -> RC:
    model = RC()
    model.add(InputLayer(input_shape=(40, n_in)))
    model.add(RandomReservoirLayer(nodes=40, density=0.3))
    model.add(ReadoutLayer(output_shape=(40, 1)))
    model.compile(**compile_kwargs)
    return model


def test_defaults_zscoring_on_unit_scaling():
    model = _model()
    assert model.normalize_inputs is True
    assert model.input_scaling == 1.0


def test_compile_arguments_set_preprocessing():
    model = _model(normalize_inputs=False, input_scaling=[0.5, 2.0])
    assert model.normalize_inputs is False
    assert np.allclose(model.input_scaling, [0.5, 2.0])


def test_compile_keeps_settings_made_before_compile():
    model = RC()
    model.add(InputLayer(input_shape=(40, 2)))
    model.add(RandomReservoirLayer(nodes=40, density=0.3))
    model.add(ReadoutLayer(output_shape=(40, 1)))
    model.set_input_scaling(2.5)
    model.set_normalization(normalize_inputs=False)
    model.compile()
    assert model.input_scaling == 2.5
    assert model.normalize_inputs is False


def test_zscore_statistics_over_batch_and_time():
    x, y = _data()
    model = _model()
    model.fit(x, y)
    assert np.allclose(model.input_mean, x.mean(axis=(0, 1)))
    assert np.allclose(model.input_std, x.std(axis=(0, 1)))
    xt = model._transform_x(x)
    assert np.allclose(xt.mean(axis=(0, 1)), 0.0, atol=1e-12)
    assert np.allclose(xt.std(axis=(0, 1)), 1.0)


@pytest.mark.parametrize("scaling", [5.0, np.array([0.3, 4.0])])
def test_input_scaling_equals_scaling_the_readin_matrix(scaling):
    # input_scaling s on z-scored inputs == scaling 1 with W_in * diag(s)
    x, y = _data()
    m_scaled = _model()
    m_weights = copy.deepcopy(m_scaled)

    m_scaled.set_input_scaling(scaling)
    m_weights.input_layer.weights = m_weights.input_layer.weights * np.atleast_1d(scaling)[None, :]

    m_scaled.fit(x, y)
    m_weights.fit(x, y)

    assert np.allclose(m_scaled.predict(x), m_weights.predict(x))
    assert np.allclose(m_scaled.predict(x[:, :15]), m_weights.predict(x[:, :15]))


def test_training_error_does_not_blow_up_with_input_scaling():
    x, y = _data()
    base, scaled = _model(), _model()
    scaled.reservoir_layer.weights = base.reservoir_layer.weights.copy()
    scaled.input_layer.weights = base.input_layer.weights.copy()
    scaled.reservoir_layer.initial_res_states = base.reservoir_layer.initial_res_states.copy()
    scaled.readout_layer.readout_nodes = list(base.readout_layer.readout_nodes)
    scaled.set_input_scaling(5.0)

    base.fit(x, y)
    scaled.fit(x, y)
    mse = lambda m: float(np.mean((m.predict(x) - y) ** 2))
    assert mse(scaled) < 10 * max(mse(base), 1e-3)


def test_changing_scaling_after_fit_does_not_change_predictions_until_refit():
    x, y = _data()
    model = _model()
    model.fit(x, y)
    before = model.predict(x)
    model.set_input_scaling(3.0)
    assert np.array_equal(model.predict(x), before)
    model.fit(x, y)
    assert not np.allclose(model.predict(x), before)


def test_normalization_off_leaves_inputs_unchanged():
    x, y = _data()
    model = _model(normalize_inputs=False)
    model.fit(x, y)
    assert model.input_mean is None and model.input_std is None
    assert np.array_equal(model._transform_x(x), x)


def test_transform_for_selected_channels():
    # used to map fed-back predictions into the coordinates of the input channels
    x, y = _data()
    model = _model(input_scaling=[0.5, 2.0])
    model.fit(x, y)
    v = np.array([[1.0], [2.0]])
    out = model._transform_x(v, channels=[1])
    expected = 2.0 * (v - model.input_mean[1]) / model.input_std[1]
    assert np.allclose(out, expected)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, [1.0, -1.0], np.ones((2, 2))])
def test_invalid_input_scaling_raises(bad):
    model = _model()
    with pytest.raises(ValueError):
        model.set_input_scaling(bad)


def test_wrong_number_of_scaling_entries_raises_at_fit():
    x, y = _data()
    model = _model(input_scaling=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="channels"):
        model.fit(x, y)


def test_predict_before_fit_raises():
    x, _ = _data()
    model = _model()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(x)


def test_output_normalization_and_scaling_are_removed():
    model = _model()
    assert not hasattr(model, "set_output_scaling")
    assert not hasattr(model, "normalize_outputs")
    with pytest.raises(ValueError):
        model.set_hp(output_scaling=0.5)
    with pytest.raises(TypeError):
        model.set_normalization(normalize_outputs=True)
    assert "output_scaling" not in model.get_hp()


def test_predictions_are_in_physical_units_of_y():
    # targets are used as given and ridge is linear in them: fitting 1000*y gives
    # exactly 1000 times the predictions
    x, y = _data()
    m1 = _model()
    m2 = copy.deepcopy(m1)
    m1.fit(x, y)
    m2.fit(x, 1000.0 * y)
    assert np.allclose(m2.predict(x), 1000.0 * m1.predict(x))
