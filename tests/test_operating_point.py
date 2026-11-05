"""
operating_point_statistics: the state-dependent node and edge quantifiers of a fitted model.
"""

import numpy as np
import pytest

from helpers import build_rc, make_xy
from pyreco.edge_analyzer import operating_point_statistics

N = 30


def _fitted(activation="tanh", input_bias=0.0, leakage_rate=0.5, K=None, **compile_kwargs):
    x, y = make_xy(n=8, T=40, n_in=2, K=K, seed=0)
    model = build_rc(
        K=K, nodes=N, seed=0, layer_seed=1, layer=dict(activation=activation, leakage_rate=leakage_rate),
        input_bias=input_bias, **compile_kwargs,
    )
    model.fit(x, y)
    return model, x


def test_shapes_and_edges_match_the_reservoir_matrix():
    model, x = _fitted()
    out = operating_point_statistics(model, x)
    A = model.reservoir_layer.weights
    for key, values in out["node"].items():
        assert values.shape == (N,), key
    edges = out["edge"]["edges"]
    assert edges.shape == (np.count_nonzero(A), 2)
    assert np.array_equal(A[edges[:, 0], edges[:, 1]] != 0, np.ones(len(edges), dtype=bool))
    assert out["edge"]["edge_signal_variance"].shape == (len(edges),)
    assert out["edge"]["edge_gain_weighted_variance"].shape == (len(edges),)


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_statistics_agree_with_a_plain_loop(activation):
    """Recompute the pre-activations step by step, as in the update equation of the model."""
    model, x = _fitted(activation=activation, input_bias=1.0)
    d = 5
    out = operating_point_statistics(model, x, discard=d)

    layer, W_in, a = model.reservoir_layer, model.input_layer.weights, model.reservoir_layer.leakage_rate
    A, bias = layer.weights, model._bias_contribution()
    x_tilde = model._transform_x(x)
    z_all, r_all = [], []
    for sample in x_tilde:
        r = np.array(layer.initial_res_states, dtype=float)
        for t in range(len(sample)):
            z = r @ A + W_in @ sample[t] + bias
            if t >= d:
                z_all.append(z)
                r_all.append(r.copy())
            r = (1 - a) * r + a * layer.activation_fun(z)
    z_all, r_all = np.array(z_all), np.array(r_all)

    g = layer.activation_fun(z_all)
    slope = 1 - g**2 if activation == "tanh" else g * (1 - g)
    assert np.allclose(out["node"]["mean_preactivation"], z_all.mean(axis=0))
    assert np.allclose(out["node"]["std_preactivation"], z_all.std(axis=0))
    assert np.allclose(out["node"]["mean_gain"], slope.mean(axis=0))

    u, v = out["edge"]["edges"].T
    expected = A[u, v] ** 2 * r_all.var(axis=0)[u]
    assert np.allclose(out["edge"]["edge_signal_variance"], expected)
    assert np.allclose(out["edge"]["edge_gain_weighted_variance"], slope.mean(axis=0)[v] ** 2 * expected)


def test_a_strong_input_bias_saturates_the_nodes_and_follows_the_signs():
    """The bias holds the nodes on the side of its sign, in the flat part of the activation."""
    x, y = make_xy(n=8, T=40, n_in=2, seed=0)
    model = build_rc(nodes=N, seed=0, layer_seed=1, input_bias=4.0)
    model.fit(x, y)
    out = operating_point_statistics(model, x)["node"]
    b = model.input_layer.bias
    assert np.array_equal(out["bias"], 4.0 * b)
    assert np.all(np.sign(out["mean_preactivation"]) == np.sign(b))
    assert out["saturation_fraction"].mean() > 0.5

    # without the bias the nodes sit near the middle of the activation curve
    plain, _ = _fitted(input_bias=0.0)
    out0 = operating_point_statistics(plain, x)["node"]
    assert np.all(out0["bias"] == 0)
    assert out0["mean_gain"].mean() > out["mean_gain"].mean()


def test_the_model_is_not_modified():
    model, x = _fitted(input_bias=1.0)
    before = model.predict(x)
    weights = model.reservoir_layer.weights.copy()
    operating_point_statistics(model, x)
    assert np.array_equal(before, model.predict(x))
    assert np.array_equal(weights, model.reservoir_layer.weights)


def test_the_default_warm_up_is_the_one_of_the_model():
    model, x = _fitted(K=33, discard_transients=7)
    default = operating_point_statistics(model, x)
    explicit = operating_point_statistics(model, x, discard=7)
    assert np.array_equal(default["node"]["mean_gain"], explicit["node"]["mean_gain"])


def test_invalid_input_is_rejected():
    model, x = _fitted()
    with pytest.raises(RuntimeError):
        operating_point_statistics(build_rc(nodes=N, seed=0), x)
    with pytest.raises(TypeError):
        operating_point_statistics(model, x.tolist())
    with pytest.raises(ValueError):
        operating_point_statistics(model, x[0])
    with pytest.raises(ValueError):
        operating_point_statistics(model, x[:, :, :1])
    with pytest.raises(ValueError):
        operating_point_statistics(model, x, discard=x.shape[1])
    with pytest.raises(ValueError):
        operating_point_statistics(model, x, saturation_threshold=1.5)
