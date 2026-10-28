"""
Signed reservoir weights: the weight distributions of RandomReservoirLayer, and the places that
used to assume non-negative weights (density, edge properties, pruning).
"""

import networkx as nx
import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.edge_analyzer import EdgeAnalyzer
from pyreco.edge_pruning import EdgePruner
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.utils_networks import (
    WEIGHT_DISTRIBUTIONS,
    compute_density,
    compute_spec_rad,
    extract_node_pagerank,
    gen_ER_graph,
)

SIGNED = ("sign", "uniform", "normal")


def test_default_is_the_binary_graph_of_before():
    """weight_distribution="positive" gives the same 0/1 graph as before signed weights existed."""
    a = gen_ER_graph(nodes=30, density=0.2, seed=4)
    assert set(np.unique(a)) <= {0.0, a.max()}
    assert (a >= 0).all()
    assert np.array_equal(a, gen_ER_graph(nodes=30, density=0.2, seed=4, weight_distribution="positive"))


@pytest.mark.parametrize("distribution", WEIGHT_DISTRIBUTIONS)
def test_every_distribution_has_the_same_er_support_and_the_requested_spectral_radius(distribution):
    """The edge positions come from the same ER graph; only the weights differ and are rescaled to rho(A) = spec_rad."""
    reference = gen_ER_graph(nodes=40, density=0.15, seed=7)
    w = gen_ER_graph(nodes=40, density=0.15, spec_rad=0.8, seed=7, weight_distribution=distribution)
    assert np.array_equal(w != 0, reference != 0)
    assert np.isclose(compute_spec_rad(w), 0.8)
    assert np.all(np.diag(w) == 0)


@pytest.mark.parametrize("distribution", SIGNED)
def test_signed_distributions_have_both_signs_in_equal_parts(distribution):
    """About half of the weights are negative for "sign", "uniform" and "normal"."""
    w = gen_ER_graph(nodes=100, density=0.2, seed=1, weight_distribution=distribution)
    positive, negative = (w > 0).sum(), (w < 0).sum()
    assert positive > 0 and negative > 0
    assert 0.8 < positive / negative < 1.25


def test_sign_distribution_has_equal_magnitudes():
    """For "sign" all non-zero weights are +-c with the same c."""
    w = gen_ER_graph(nodes=50, density=0.2, seed=1, weight_distribution="sign")
    assert np.allclose(np.unique(np.abs(w[w != 0])), np.abs(w[w != 0])[0])


@pytest.mark.parametrize("distribution", SIGNED)
def test_weights_are_reproducible_from_the_seed(distribution):
    """The same layer seed gives the same signed weights."""
    kw = dict(nodes=30, density=0.2, weight_distribution=distribution)
    assert np.array_equal(gen_ER_graph(seed=3, **kw), gen_ER_graph(seed=3, **kw))
    assert not np.array_equal(gen_ER_graph(seed=3, **kw), gen_ER_graph(seed=4, **kw))


def test_small_sparse_signed_graphs_still_have_a_cycle():
    """Cancelling signs must not produce a (numerically) zero spectral radius; such graphs are redrawn."""
    for seed in range(100):
        w = gen_ER_graph(nodes=5, density=0.1, spec_rad=0.9, seed=seed, weight_distribution="sign")
        assert np.isfinite(w).all()
        assert np.isclose(compute_spec_rad(w), 0.9)


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError, match="weight_distribution"):
        gen_ER_graph(nodes=10, density=0.3, weight_distribution="cauchy")
    with pytest.raises(ValueError, match="weight_distribution"):
        RandomReservoirLayer(nodes=10, weight_distribution="cauchy")


@pytest.mark.parametrize("distribution", WEIGHT_DISTRIBUTIONS)
def test_layer_density_counts_negative_weights(distribution):
    """Density counts non-zero weights of either sign."""
    layer = RandomReservoirLayer(nodes=50, density=0.2, seed=2, weight_distribution=distribution)
    assert layer.weight_distribution == distribution
    assert np.isclose(layer.density, np.count_nonzero(layer.weights) / 50**2)
    assert np.isclose(layer.density, compute_density(layer.weights))


def test_compute_density_counts_negative_entries():
    assert compute_density(np.array([[0.0, -1.0], [2.0, 0.0]])) == 0.5


def test_pagerank_ignores_the_sign_of_the_weights():
    """The PageRank of a signed graph is a probability vector and equals the one of |w|."""
    w = gen_ER_graph(nodes=20, density=0.3, seed=1, weight_distribution="normal")
    ranks = [extract_node_pagerank(w, n) for n in range(20)]
    assert np.isclose(sum(ranks), 1.0)
    assert np.allclose(ranks, [extract_node_pagerank(np.abs(w), n) for n in range(20)])


def test_edge_properties_abs_weight_and_sign():
    """The edge analyzer reports |w| and sign(w)."""
    g = nx.DiGraph()
    g.add_edge(0, 1, weight=-2.5)
    g.add_edge(1, 0, weight=0.5)
    props = EdgeAnalyzer(quantities=["weight", "abs_weight", "sign"]).extract_properties(g, (0, 1))
    assert props == {"weight": -2.5, "abs_weight": 2.5, "sign": -1}
    batch = EdgeAnalyzer(quantities=["abs_weight", "sign"]).extract_properties_batch(g, [(0, 1), (1, 0)])
    assert batch == [{"abs_weight": 2.5, "sign": -1}, {"abs_weight": 0.5, "sign": 1}]
    matrix = np.array([[0.0, -3.0], [4.0, 0.0]])
    assert EdgeAnalyzer(quantities=["sign"]).extract_properties(matrix, (0, 1)) == {"sign": -1}


@pytest.mark.parametrize("distribution", SIGNED)
def test_signed_reservoir_trains_and_edges_can_be_pruned(distribution):
    """A signed reservoir trains and EdgePruner removes edges from it."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 30, 1))
    y = np.tanh(x) * 0.5
    model = RC(seed=1)
    model.add(InputLayer(input_shape=x.shape[1:]))
    model.add(RandomReservoirLayer(nodes=15, density=0.3, seed=2, weight_distribution=distribution))
    model.add(ReadoutLayer(output_shape=y.shape[1:]))
    model.compile()
    model.fit(x, y)
    layer = model.reservoir_layer
    n_before = int(np.count_nonzero(layer.weights))
    rows, cols = np.nonzero(layer.weights)
    model.remove_reservoir_edges([(int(u), int(v)) for u, v in zip(rows[:5], cols[:5])])
    assert np.count_nonzero(layer.weights) == n_before - 5
    assert np.isclose(layer.density, compute_density(layer.weights))
    assert np.isclose(layer.spec_rad, compute_spec_rad(layer.weights))
    # the negative edges are edges too: pruning down to a few edges selects from all of them
    pruned, _ = EdgePruner(
        candidate_fraction=0.5, stopping_criterion=["min_edges"], min_num_edges=n_before - 10, seed=1
    ).prune(model, (x, y), (x, y))
    assert np.count_nonzero(pruned.reservoir_layer.weights) <= n_before - 10
