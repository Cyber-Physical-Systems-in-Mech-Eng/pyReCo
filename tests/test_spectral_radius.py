"""
Networks with spectral radius 0 (no directed cycle) cannot be rescaled to a target
spectral radius. Construction redraws such graphs, rescaling raises a clear error without
touching the layer, and pruning keeps the layer properties in sync.
"""

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.edge_pruning import EdgePruner
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_networks import (
    compute_density,
    compute_spec_rad,
    gen_ER_graph,
    set_spec_rad,
)


def _data(seed=0, n=8, t=30):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, t, 2))
    y = np.tanh(x[:, :, :1]) + 0.3 * x[:, :, 1:2]
    return x, y


def _model(nodes=10, seed=1, **layer_kwargs):
    x, y = _data()
    model = RC(seed=2)
    model.add(InputLayer(input_shape=x.shape[1:]))
    model.add(RandomReservoirLayer(nodes=nodes, density=0.3, seed=seed, **layer_kwargs))
    model.add(ReadoutLayer(output_shape=y.shape[1:]))
    model.compile()
    return model, x, y


def _upper_triangular(n, density=0.4, seed=0):
    rng = np.random.default_rng(seed)
    return np.triu(rng.random((n, n)) < density, 1).astype(float)


# ------------------------------------------------------------ graph generation


def test_small_sparse_graphs_are_always_finite_with_the_requested_spectral_radius():
    # without redrawing, ~40% of these graphs have spectral radius 0 and give NaN weights
    for seed in range(200):
        g = gen_ER_graph(nodes=5, density=0.05, spec_rad=0.9, seed=seed)
        assert np.isfinite(g).all()
        assert np.isclose(compute_spec_rad(g), 0.9)


def test_redrawing_is_deterministic():
    a = gen_ER_graph(nodes=5, density=0.05, seed=3)
    b = gen_ER_graph(nodes=5, density=0.05, seed=3)
    assert np.array_equal(a, b)
    c = gen_ER_graph(nodes=5, density=0.05, seed=np.random.default_rng(3))
    d = gen_ER_graph(nodes=5, density=0.05, seed=np.random.default_rng(3))
    assert np.array_equal(c, d)


@pytest.mark.parametrize("kwargs", [dict(nodes=10, density=0.0), dict(nodes=1, density=0.5)])
def test_impossible_graphs_raise(kwargs):
    with pytest.raises(ValueError, match="directed cycle"):
        gen_ER_graph(**kwargs)


def test_reservoir_layer_construction_succeeds_for_small_sparse_reservoirs():
    for seed in range(60):
        layer = RandomReservoirLayer(nodes=5, density=0.05, spec_rad=0.8, seed=seed)
        assert np.isfinite(layer.weights).all()
        assert np.isclose(layer.spec_rad, 0.8)
        assert np.isclose(layer.density, compute_density(layer.weights))


# --------------------------------------------------------------- rescaling


def test_set_spec_rad_scales_exactly_when_there_is_a_cycle():
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 2] = A[2, 0] = 2.0
    scaled = set_spec_rad(A, 0.5)
    assert np.isclose(compute_spec_rad(scaled), 0.5)
    assert np.allclose(scaled, A * (0.5 / compute_spec_rad(A)))


def test_set_spec_rad_on_a_network_without_cycle_raises():
    A = _upper_triangular(10)
    assert compute_spec_rad(A) == 0.0
    with pytest.raises(ValueError, match="spectral radius 0"):
        set_spec_rad(A, 0.9)


def test_layer_set_spec_rad_failure_leaves_the_layer_untouched():
    model, _, _ = _model()
    layer = model.reservoir_layer
    layer.set_weights(_upper_triangular(10))
    weights, spec_rad, density = layer.weights.copy(), layer.spec_rad, layer.density
    assert spec_rad == 0.0

    with pytest.raises(ValueError, match="spectral radius 0"):
        layer.set_spec_rad(0.9)

    assert np.array_equal(layer.weights, weights)
    assert layer.spec_rad == spec_rad and layer.density == density


def test_set_hp_spec_rad_on_a_pruned_to_feed_forward_model_keeps_the_model_usable():
    model, x, y = _model()
    model.reservoir_layer.set_weights(_upper_triangular(10))
    model.fit(x, y)
    before = model.predict(x)
    with pytest.raises(ValueError, match="spectral radius 0"):
        model.set_hp(spec_rad=0.9)
    assert np.array_equal(model.predict(x), before)


# ------------------------------------------- layer properties after edge removal


def test_remove_reservoir_edges_keeps_layer_properties_in_sync():
    model, _, _ = _model(nodes=20)
    layer = model.reservoir_layer
    rows, cols = np.nonzero(layer.weights)
    model.remove_reservoir_edges([(int(u), int(v)) for u, v in zip(rows[:15], cols[:15])])
    assert np.isclose(layer.density, compute_density(layer.weights))
    assert np.isclose(layer.spec_rad, compute_spec_rad(layer.weights))
    assert layer.spec_rad < 0.9


def test_removing_the_last_cycle_gives_spectral_radius_zero_without_error():
    model, _, _ = _model()
    A = np.zeros((10, 10))
    A[0, 1] = A[1, 2] = A[2, 0] = 0.5
    model.reservoir_layer.set_weights(A)
    assert model.reservoir_layer.spec_rad > 0
    model.remove_reservoir_edges([(2, 0)])
    assert model.reservoir_layer.spec_rad == 0.0


# ----------------------------------------------------------------- pruner


def test_edge_pruner_warns_when_spectral_radius_reaches_zero():
    # a single directed cycle: removing any edge removes the last cycle
    model, x, y = _model(nodes=6, seed=4)
    A = np.zeros((6, 6))
    for i in range(6):
        A[i, (i + 1) % 6] = 0.9
    model.reservoir_layer.set_weights(A)
    model.fit(x, y)

    pruner = EdgePruner(
        candidate_fraction=1.0, stopping_criterion=["min_edges"], min_num_edges=3, seed=1
    )
    with pytest.warns(RuntimeWarning, match="spectral radius is now 0"):
        pruned, _ = pruner.prune(model, (x, y), _data(seed=5))
    assert pruned.reservoir_layer.spec_rad == 0.0
