"""
EdgePruner(rescale_spec_rad=...): the spectral radius of the reservoir is kept fixed while edges are removed.
"""

import copy

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.edge_pruning import EdgePruner, _evaluate_candidate_standalone, _remove_edge
from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.optimizers import RidgeSK
from pyreco.utils_data import sequence_to_sequence as seq_2_seq


@pytest.fixture
def data():
    X_train, X_test, y_train, y_test = seq_2_seq(name="sine_pred", n_batch=10, n_states=2, n_time=50)
    return (X_train, y_train), (X_test, y_test)


@pytest.fixture
def model(data):
    X_train, y_train = data[0]
    model = RC(seed=0)
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=10, density=0.5, spec_rad=0.8, leakage_rate=0.3, seed=1))
    model.add(ReadoutLayer(y_train.shape[1:], fraction_out=0.9))
    model.compile(optimizer=RidgeSK(alpha=0.5), metrics=["mean_squared_error"])
    model.fit(X_train, y_train)
    return model


def prune(model, data, removals=6, **kwargs):
    """Prune a copy of the model by a few edges (so that a directed cycle is left)."""
    n_edges = int(np.count_nonzero(model.reservoir_layer.weights))
    pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=["min_edges"], min_num_edges=n_edges - removals,
                        return_best_model=False, seed=0, **kwargs)
    return pruner.prune(model=copy.deepcopy(model), data_train=data[0], data_val=data[1])[0]


@pytest.mark.parametrize("value", [0, -0.5, np.inf, np.nan])
def test_a_radius_that_is_not_positive_and_finite_raises(value):
    with pytest.raises(ValueError, match="rescale_spec_rad"):
        EdgePruner(rescale_spec_rad=value)


@pytest.mark.parametrize("value", ["start", True, [0.9]])
def test_a_radius_of_the_wrong_type_raises(value):
    with pytest.raises(TypeError, match="rescale_spec_rad"):
        EdgePruner(rescale_spec_rad=value)


def test_without_rescaling_the_radius_drifts(model, data):
    pruned = prune(model, data)
    assert not np.isclose(pruned.reservoir_layer.spec_rad, model.reservoir_layer.spec_rad, rtol=1e-3)


def test_initial_keeps_the_starting_radius(model, data):
    start = model.reservoir_layer.spec_rad
    pruned = prune(model, data, rescale_spec_rad="initial")
    assert np.isclose(pruned.reservoir_layer.spec_rad, start, rtol=1e-6)
    assert np.isclose(np.abs(np.linalg.eigvals(pruned.reservoir_layer.weights)).max(), start, rtol=1e-6)
    assert np.count_nonzero(pruned.reservoir_layer.weights) == np.count_nonzero(model.reservoir_layer.weights) - 6


def test_a_float_rescales_the_starting_model_and_the_pruned_one(model, data):
    start = model.reservoir_layer.spec_rad
    pruned = prune(model, data, rescale_spec_rad=0.5)
    assert not np.isclose(start, 0.5)
    assert np.isclose(pruned.reservoir_layer.spec_rad, 0.5, rtol=1e-6)
    assert np.isclose(model.reservoir_layer.spec_rad, start)  # the model that was passed is not changed


def test_the_candidates_are_scored_at_the_fixed_radius(model, data):
    (X_train, y_train), (X_val, y_val) = data
    edge = tuple(int(i) for i in np.argwhere(model.reservoir_layer.weights != 0)[0])
    scored, candidate, _ = _evaluate_candidate_standalone(
        model, edge, X_train, y_train, X_val, y_val, "mse", GraphAnalyzer(), target_spec_rad=0.8
    )
    assert np.isclose(candidate.reservoir_layer.spec_rad, 0.8, rtol=1e-6)
    assert candidate.reservoir_layer.weights[edge] == 0
    plain = _evaluate_candidate_standalone(model, edge, X_train, y_train, X_val, y_val, "mse", GraphAnalyzer())[1]
    assert not np.isclose(plain.reservoir_layer.spec_rad, 0.8, rtol=1e-3)


def test_a_reservoir_without_a_cycle_is_left_as_it_is(model):
    cycle = np.zeros((10, 10))
    for node in range(10):
        cycle[node, (node + 1) % 10] = 0.5
    model.reservoir_layer.set_weights(cycle)
    _remove_edge(model, (0, 1), target_spec_rad=0.9)  # the cycle is broken: radius 0, no error
    assert model.reservoir_layer.spec_rad == 0
    assert model.reservoir_layer.weights[0, 1] == 0
    assert np.count_nonzero(model.reservoir_layer.weights) == 9


def test_a_model_without_a_cycle_cannot_be_pruned_at_a_fixed_radius(model, data):
    model.reservoir_layer.set_weights(np.triu(model.reservoir_layer.weights, k=1))
    with pytest.raises(ValueError, match="directed cycle"):
        EdgePruner(rescale_spec_rad="initial", stopping_criterion=["min_edges"], min_num_edges=0).prune(
            model=model, data_train=data[0], data_val=data[1]
        )


def test_the_radius_is_kept_until_the_last_cycle_is_gone(model, data):
    """Pruning on to the end warns once that the radius is 0 and cannot be rescaled any more."""
    n_edges = int(np.count_nonzero(model.reservoir_layer.weights))
    with pytest.warns(RuntimeWarning, match="can no longer be rescaled"):
        pruned = prune(model, data, removals=n_edges, rescale_spec_rad="initial")
    assert pruned.reservoir_layer.spec_rad == 0
