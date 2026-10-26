"""
Reproducibility of the edge selector, the edge pruner, the plotting layout and the data
utilities. None of them may use or modify the global random state.
"""

import random

import numpy as np
import pytest

from pyreco.cross_validation import cross_val
from pyreco.custom_models import RC
from pyreco.edge_pruning import EdgePruner
from pyreco.edge_selector import EdgeSelector
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.plotting import visualize_reservoir_network_circle
from pyreco.utils_data import sequence_to_sequence, sine_pred, train_test_split


def _global_state():
    return np.random.get_state(), random.getstate()


def _unchanged(before):
    np_before, py_before = before
    np_now, py_now = np.random.get_state(), random.getstate()
    return (
        np.array_equal(np_before[1], np_now[1])
        and np_before[2:] == np_now[2:]
        and py_before == py_now
    )


def _adjacency(seed=3, n=30, density=0.2):
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) < density).astype(float)


# ---------------------------------------------------------------- EdgeSelector


def test_edge_selector_is_seeded():
    A = _adjacency()
    pick = lambda rng: EdgeSelector(graph=A, rng=rng).select_edges(fraction=0.2)
    a, b, c = pick(5), pick(5), pick(6)
    assert a == b and a != c
    assert len(a) == len(set(a)) == max(1, round(0.2 * int((A != 0).sum())))
    assert all(isinstance(e, tuple) and A[e] != 0 for e in a)


def test_edge_selector_shared_generator_advances():
    A = _adjacency()
    rng = np.random.default_rng(1)
    selector = EdgeSelector(graph=A, rng=rng)
    assert selector.select_edges(fraction=0.2) != selector.select_edges(fraction=0.2)


def test_edge_selector_does_not_touch_global_state():
    before = _global_state()
    EdgeSelector(graph=_adjacency()).select_edges(fraction=0.2)
    EdgeSelector(graph=_adjacency(), rng=1).select_edges(num=3)
    assert _unchanged(before)


# ----------------------------------------------------------------- EdgePruner


def _pruning_setup():
    data = sequence_to_sequence(name="sine_pred", n_batch=10, n_states=2, n_time=40, seed=1)
    X_train, X_test, y_train, y_test = data
    model = RC(seed=2)
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=10, density=0.4, fraction_input=0.5, seed=3))
    model.add(ReadoutLayer(y_train.shape[1:], fraction_out=0.9))
    model.compile()
    model.fit(X_train, y_train)
    n_edges = int(np.count_nonzero(model.reservoir_layer.weights))
    return model, (X_train, y_train), (X_test, y_test), n_edges


def _pruner(seed, n_edges):
    return EdgePruner(
        candidate_fraction=0.3,
        stopping_criterion=["min_edges"],
        min_num_edges=n_edges - 4,
        seed=seed,
    )


def test_edge_pruner_is_reproducible_and_leaves_global_state_alone():
    before = _global_state()
    results = []
    for _ in range(2):
        model, train, val, n_edges = _pruning_setup()
        pruned, history = _pruner(7, n_edges).prune(model, train, val)
        results.append((pruned.reservoir_layer.weights.copy(), len(history)))
    assert _unchanged(before)
    assert np.array_equal(results[0][0], results[1][0]) and results[0][1] == results[1][1]


def test_edge_pruner_restarts_its_stream_for_every_prune_call():
    import copy

    model, train, val, n_edges = _pruning_setup()
    pruner = _pruner(7, n_edges)
    first, _ = pruner.prune(copy.deepcopy(model), train, val)
    second, _ = pruner.prune(copy.deepcopy(model), train, val)  # same pruner, second call
    assert np.array_equal(first.reservoir_layer.weights, second.reservoir_layer.weights)


@pytest.mark.parametrize("bad", [1.5, "3", -1])
def test_edge_pruner_invalid_seed_raises(bad):
    with pytest.raises((TypeError, ValueError)):
        EdgePruner(seed=bad)


# ------------------------------------------------------------------- plotting


def test_visualization_layout_is_seeded_and_does_not_touch_global_state(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    rng = np.random.default_rng(0)
    A = (rng.random((8, 8)) < 0.3).astype(float)
    W_in, W_out = rng.normal(size=(8, 1)), rng.normal(size=(8, 1))
    colors = {
        k: "gray"
        for k in ["CWinp", "CWres_inp", "CWres_out", "CWres_both", "CWres_internal",
                  "CWout", "Winp", "Wout", "CWres"]
    }

    before = _global_state()
    visualize_reservoir_network_circle(A, W_in, W_out, 1, 1, seed=3, Node_colors=colors,
                                       Edge_Weights=0.7)
    assert _unchanged(before)
    plt.close("all")


# --------------------------------------------------------------- data utilities


def test_train_test_split_is_seeded_and_ignores_global_state():
    x = np.arange(40).reshape(20, 2, 1)
    y = x + 100
    np.random.seed(1)
    a = train_test_split(x, y, seed=5)
    np.random.seed(2)
    b = train_test_split(x, y, seed=5)
    c = train_test_split(x, y, seed=6)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("seed", [None, 1])
def test_data_generators_accept_seed(seed):
    out = sine_pred(n_batch=20, n_time_in=3, n_time_out=2, n_states=2, seed=seed)
    wrapped = sequence_to_sequence(name="sine_pred", n_batch=20, n_states=2, n_time=3, seed=seed)
    assert len(out) == len(wrapped) == 4
    if seed is not None:
        again = sequence_to_sequence(name="sine_pred", n_batch=20, n_states=2, n_time=3, seed=seed)
        assert all(np.array_equal(p, q) for p, q in zip(wrapped, again))


def test_cross_val_folds_are_seeded():
    class _Echo:
        """Minimal model: records the test indices through the data it is given."""

        def __init__(self):
            self.seen = []

        def fit(self, x, y, **kwargs):
            pass

        def evaluate(self, x, y, metrics=None):
            self.seen.append(float(x.sum()))
            return [0.0]

    x = np.arange(60, dtype=float).reshape(20, 3, 1)
    y = x.copy()
    runs = []
    for seed in (1, 1, 2):
        model = _Echo()
        np.random.seed(seed + 100)  # irrelevant
        cross_val(model, x, y, n_splits=4, seed=seed)
        runs.append(model.seen)
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]
