"""
pyreco.features: named network and edge features, the registry, custom features, the context.
"""

import numpy as np
import pytest

from pyreco import features as F
from pyreco.custom_models import RC
from pyreco.edge_analyzer import EdgeAnalyzer
from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.optimizers import RidgeSK
from pyreco.utils_data import sequence_to_sequence as seq_2_seq


@pytest.fixture
def data():
    X_train, _, y_train, _ = seq_2_seq(name="sine_pred", n_batch=10, n_states=2, n_time=50)
    return X_train, y_train


@pytest.fixture
def model(data):
    X_train, y_train = data
    model = RC(seed=0)
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=12, density=0.4, spec_rad=0.9, leakage_rate=0.3, seed=1))
    model.add(ReadoutLayer(y_train.shape[1:], fraction_out=0.9))
    model.compile(optimizer=RidgeSK(alpha=0.5), metrics=["mean_squared_error"])
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def edges(model):
    return [tuple(int(i) for i in e) for e in np.argwhere(model.reservoir_layer.weights != 0)[:8]]


@pytest.fixture
def clean_registry():
    """Remove the features that a test registers."""
    before = (set(F._NETWORK_FEATURES), set(F._EDGE_FEATURES))
    yield
    for registry, names in zip((F._NETWORK_FEATURES, F._EDGE_FEATURES), before):
        for name in set(registry) - names:
            del registry[name]


# ------------------------------------------------------------------------------------------ registry


def test_defaults_are_registered_and_described():
    network, edge = F.available_network_features(), F.available_edge_features()
    assert set(F.DEFAULT_NETWORK_FEATURES) <= set(network)
    assert set(F.DEFAULT_EDGE_FEATURES) <= set(edge)
    assert "mean_sq_activity" in network and "source_mean_sq_activity" in edge
    assert all(description for description in list(network.values()) + list(edge.values()))


def test_no_name_is_used_on_both_levels():
    assert not set(F.available_network_features()) & set(F.available_edge_features())


def test_resolve_none_gives_the_defaults():
    network, edge = F.resolve_features(None)
    assert list(network) == F.DEFAULT_NETWORK_FEATURES
    assert list(edge) == F.DEFAULT_EDGE_FEATURES


def test_resolve_keeps_the_order_and_splits_the_levels():
    network, edge = F.resolve_features(["abs_weight", "spectral_radius", "density", "sign"])
    assert list(network) == ["spectral_radius", "density"]
    assert list(edge) == ["abs_weight", "sign"]
    assert F.resolve_features([]) == ({}, {})


def test_unknown_names_and_wrong_entries_raise():
    with pytest.raises(ValueError, match="Unknown feature 'nope'"):
        F.resolve_features(["nope"])
    with pytest.raises(TypeError, match="decorated"):
        F.resolve_features([lambda ctx: 1.0])
    with pytest.raises(TypeError, match="list of names"):
        F.resolve_features("density")
    with pytest.raises(ValueError, match="twice"):
        F.resolve_features(["density", "density"])


# ----------------------------------------------------------------------------------- custom features


def test_a_custom_network_feature_works_by_name_and_as_function(model, data, clean_registry):
    @F.network_feature("frobenius_norm")
    def frobenius_norm(ctx):
        """Frobenius norm of the reservoir matrix."""
        return np.linalg.norm(ctx.weights)

    expected = np.linalg.norm(model.reservoir_layer.weights)
    assert F.compute_network_features(model, data[0], ["frobenius_norm"]) == {"frobenius_norm": pytest.approx(expected)}
    assert F.compute_network_features(model, data[0], [frobenius_norm]) == {"frobenius_norm": pytest.approx(expected)}
    assert F.available_network_features()["frobenius_norm"] == "Frobenius norm of the reservoir matrix."


def test_a_custom_edge_feature_can_use_the_activity(model, data, edges, clean_registry):
    @F.edge_feature("signal_power")
    def signal_power(ctx, u, v):
        return ctx.weights[u, v] ** 2 * ctx.node_mean_sq_activity[u]

    values = F.compute_edge_features(model, data[0], edges, ["signal_power"])["signal_power"]
    weights = model.reservoir_layer.weights
    activity = F.NetworkContext(model, data[0]).node_mean_sq_activity
    assert values == pytest.approx([weights[u, v] ** 2 * activity[u] for u, v in edges])


def test_names_of_the_pruning_tables_are_reserved(clean_registry):
    for name in ("edges", "score"):
        with pytest.raises(ValueError, match="reserved"):
            F.network_feature(name)
        with pytest.raises(ValueError, match="reserved"):
            F.edge_feature(name)


def test_a_name_cannot_be_taken_twice(clean_registry):
    with pytest.raises(ValueError, match="exists already"):
        F.network_feature("density")(lambda ctx: 0.0)
    with pytest.raises(ValueError, match="exists already"):
        F.edge_feature("density")(lambda ctx, u, v: 0.0)  # taken by a network feature
    with pytest.raises(TypeError, match="non-empty string"):
        F.network_feature("")


def test_defining_the_same_function_again_replaces_it(clean_registry):
    def make():
        @F.network_feature("twice")
        def twice(ctx):
            return 1.0
        return twice

    make()
    second = make()
    assert F._NETWORK_FEATURES["twice"] is second


def test_a_feature_must_return_one_number(model, data, clean_registry):
    @F.network_feature("a_vector")
    def a_vector(ctx):
        return np.ones(3)

    with pytest.raises(TypeError, match="'a_vector' must return one number"):
        F.compute_network_features(model, data[0], ["a_vector"])


# ------------------------------------------------------------------------------------ the defaults


def test_the_structural_network_features_equal_the_graph_analyzer(model, data):
    values = F.compute_network_features(model, data[0])
    reference = GraphAnalyzer().extract_properties(graph=model.reservoir_layer.weights)
    assert values.keys() == reference.keys()
    for name in reference:
        assert values[name] == pytest.approx(reference[name])


def test_the_structural_edge_features_equal_the_edge_analyzer(model, data, edges):
    names = ["weight", "abs_weight", "sign", "is_reciprocal", "in_scc", "betweenness", "source_out_degree",
             "target_in_degree", "source_betweenness", "target_betweenness"]
    values = F.compute_edge_features(model, data[0], edges, names)
    reference = EdgeAnalyzer(quantities=names).extract_properties_batch(model.reservoir_layer.weights, edges)
    for name in names:
        assert values[name] == pytest.approx([row[name] for row in reference]), name


def test_edge_features_are_arrays_with_one_value_per_edge(model, data, edges):
    values = F.compute_edge_features(model, data[0], edges)
    assert list(values) == F.DEFAULT_EDGE_FEATURES
    assert all(v.shape == (len(edges),) and v.dtype == float for v in values.values())


# --------------------------------------------------------------------------------------- activity


def test_mean_sq_activity_is_the_second_moment_of_the_states_the_readout_sees(model, data):
    states = model.compute_reservoir_state(model._transform_x(data[0]))[:, model.num_transients_to_remove + 1:]
    per_node = (states ** 2).mean(axis=(0, 1))
    ctx = F.NetworkContext(model, data[0])
    assert ctx.node_mean_sq_activity == pytest.approx(per_node)
    value = F.compute_network_features(model, data[0], ["mean_sq_activity"])["mean_sq_activity"]
    assert value == pytest.approx(per_node.mean())
    assert 0 < value < 1  # tanh


def test_the_edge_activity_is_the_activity_of_its_end_nodes(model, data, edges):
    per_node = F.NetworkContext(model, data[0]).node_mean_sq_activity
    values = F.compute_edge_features(model, data[0], edges, ["source_mean_sq_activity", "target_mean_sq_activity"])
    assert values["source_mean_sq_activity"] == pytest.approx([per_node[u] for u, _ in edges])
    assert values["target_mean_sq_activity"] == pytest.approx([per_node[v] for _, v in edges])


def test_the_context_computes_the_activity_once(model, data, monkeypatch):
    calls = []
    original = model.compute_reservoir_state
    monkeypatch.setattr(model, "compute_reservoir_state", lambda x: calls.append(1) or original(x))
    ctx = F.EdgeContext(model, data[0])
    F.evaluate_network_features(F.resolve_features(["mean_sq_activity"])[0], ctx)
    F.evaluate_edge_features(F.resolve_features(["source_mean_sq_activity"])[1], ctx, [(0, 1)])
    assert len(calls) == 1


def test_structural_features_do_not_run_the_model(model, data, edges, monkeypatch):
    monkeypatch.setattr(model, "compute_reservoir_state", lambda x: pytest.fail("the states are not needed"))
    F.compute_network_features(model, data[0])
    F.compute_edge_features(model, data[0], edges)
