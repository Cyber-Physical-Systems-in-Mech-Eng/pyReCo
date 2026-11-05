"""
Input bias (a constant input to every reservoir node) and the distribution of the read-in weights.
"""

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import INPUT_WEIGHT_DISTRIBUTIONS, InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.metrics import nrmse


def _model(nodes=20, seed=1, weight_distribution="normal", fraction_input=1.0, shape=(40, 1), **compile_kwargs):
    model = RC(seed=seed)
    model.add(InputLayer(input_shape=shape, weight_distribution=weight_distribution))
    model.add(RandomReservoirLayer(nodes=nodes, density=0.2, seed=seed, fraction_input=fraction_input))
    model.add(ReadoutLayer(output_shape=shape))
    model.compile(**compile_kwargs)
    return model


def _data(n=10, t=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, t, 1))
    return x, 0.5 * np.tanh(x) + 0.1


# --------------------------------------------------------------------- input bias


def test_default_is_no_bias_and_nothing_changes():
    """input_bias = 0 reproduces the model without a bias exactly."""
    x, y = _data()
    a, b = _model(), _model(input_bias=0)
    assert a.input_bias == 0.0 and a.get_hp("input_bias") == {"input_bias": 0.0}
    a.fit(x, y)
    b.fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))


def test_bias_signs_are_plus_minus_one_for_every_node_and_do_not_depend_on_the_gain():
    """The bias vector b has entries +-1; the gain scales it (b_eff = gain * b)."""
    off, on = _model(input_bias=0.0), _model(input_bias=2.0)
    assert off.input_layer.bias.shape == (20,)
    assert set(np.unique(off.input_layer.bias)) == {-1.0, 1.0}
    assert np.array_equal(off.input_layer.bias, on.input_layer.bias)
    # the other random choices are not touched by the bias stream
    assert np.array_equal(off.input_layer.weights, on.input_layer.weights)
    assert np.array_equal(off.reservoir_layer.input_receiving_nodes, on.reservoir_layer.input_receiving_nodes)
    assert not np.array_equal(_model(seed=2).input_layer.bias, off.input_layer.bias)


def test_bias_reaches_every_node_even_if_only_some_receive_the_input():
    """The constant bias goes to all nodes, the input only to the fraction_input nodes."""
    model = _model(fraction_input=0.3, input_bias=1.0, normalize_inputs=False)
    x = np.zeros((2, 15, 1))
    states = model.compute_reservoir_state(x)
    assert np.all(states[:, -1] != 0)  # every node moved although x = 0 everywhere
    model.set_input_bias(0.0)
    model._input_bias_fit = None
    assert np.all(model.compute_reservoir_state(x) == 0)


def test_bias_moves_the_states_by_the_gain_times_the_signs_for_one_step():
    """For zero input and r(0) = 0 the first state is a tanh(gain * b), with the leakage rate a."""
    model = _model(input_bias=0.7, normalize_inputs=False)
    alpha = model.reservoir_layer.leakage_rate
    states = model.compute_reservoir_state(np.zeros((1, 1, 1)))
    expected = alpha * np.tanh(0.7 * model.input_layer.bias)  # initial state is zero
    assert np.allclose(states[0, 1], expected)


def test_bias_is_frozen_at_fit_like_the_input_scaling():
    """Changing the gain after fit does not change predictions until the next fit."""
    x, y = _data()
    model = _model(input_bias=1.0)
    model.fit(x, y)
    before = model.predict(x)
    model.set_input_bias(3.0)
    assert np.array_equal(model.predict(x), before)
    model.fit(x, y)
    assert not np.array_equal(model.predict(x), before)


def test_a_bias_lets_the_reservoir_learn_an_even_function():
    """tanh is odd, so without a bias (and with an input without mean) the states are odd functions of the
    input and cannot represent y = x^2 (NRMSE about 1); with a bias they can."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(20, 60, 1))
    y = x**2
    x_test = rng.uniform(-1, 1, size=(20, 60, 1))
    errors = {}
    for bias in (0.0, 1.0):
        model = _model(nodes=30, shape=(60, 1), normalize_inputs=False, input_scaling=1.0, input_bias=bias)
        model.fit(x, y)
        errors[bias] = float(nrmse(x_test**2, model.predict(x_test)))
    assert errors[0.0] > 0.95 and errors[1.0] < 0.7 * errors[0.0]  # without bias: no better than the mean


def test_input_bias_is_a_hyperparameter():
    """input_bias is set and read with set_hp / get_hp."""
    model = _model()
    model.set_hp(input_bias=0.5)
    assert model.get_hp("input_bias") == {"input_bias": 0.5}
    assert model.get_hp()["input_bias"] == 0.5


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
def test_invalid_bias_values_are_rejected(bad):
    with pytest.raises(ValueError, match="input_bias"):
        _model(input_bias=bad)


@pytest.mark.parametrize("bad", ["1", True, None.__class__])
def test_non_numeric_bias_is_rejected(bad):
    with pytest.raises(TypeError, match="input_bias"):
        _model().set_input_bias(bad)


def test_removing_nodes_removes_their_bias_and_the_model_still_trains():
    """The bias vector shrinks together with the reservoir."""
    x, y = _data()
    model = _model(input_bias=1.0)
    model.fit(x, y)
    signs = model.input_layer.bias.copy()
    model.remove_reservoir_nodes([0, 5])
    assert np.array_equal(model.input_layer.bias, np.delete(signs, [0, 5]))
    model.fit(x, y)
    assert model.predict(x).shape == y.shape


def test_bias_is_seeded_by_the_model_seed():
    """The signs are drawn from the model seed."""
    x, y = _data()
    runs = []
    for _ in range(2):
        model = _model(input_bias=1.0)
        model.fit(x, y)
        runs.append(model.predict(x))
    assert np.array_equal(*runs)


# ------------------------------------------------------------ read-in weight distribution


def test_default_read_in_weights_are_normal_scaled_to_one():
    w = _model().input_layer.weights
    assert np.isclose(np.max(np.abs(w)), 1.0)
    assert len(np.unique(w)) > 5


def test_sign_read_in_weights_have_equal_magnitude_and_both_signs():
    w = _model(nodes=50, weight_distribution="sign", fraction_input=0.6).input_layer.weights
    receiving = w[w != 0]
    assert np.all(np.abs(receiving) == 1.0)
    assert (receiving > 0).any() and (receiving < 0).any()
    assert np.count_nonzero(w) == 30  # only the nodes that receive the input


def test_uniform_read_in_weights_are_in_range():
    w = _model(nodes=50, weight_distribution="uniform").input_layer.weights
    assert np.all(np.abs(w) <= 1.0)
    assert (w > 0).any() and (w < 0).any()


def test_unknown_read_in_distribution_is_rejected():
    with pytest.raises(ValueError, match="weight_distribution"):
        InputLayer(input_shape=(10, 1), weight_distribution="cauchy")
    assert set(INPUT_WEIGHT_DISTRIBUTIONS) == {"normal", "uniform", "sign"}


def test_read_in_distribution_is_seeded():
    a = _model(weight_distribution="sign").input_layer.weights
    b = _model(weight_distribution="sign").input_layer.weights
    assert np.array_equal(a, b)
