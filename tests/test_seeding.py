"""
Reproducibility: all randomness comes from explicit seeds, never from the global random
state (np.random.*, random).
"""

import random

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.initializer import NetworkInitializer
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.models import ReservoirComputer
from pyreco.node_selector import NodeSelector
from pyreco.utils_networks import gen_ER_graph


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(6, 40, 2))
    y = np.tanh(x[:, :, :1]) + 0.3 * x[:, :, 1:2]
    return x, y


def _model(layer_seed=1, model_seed=2, fraction_input=0.5, fraction_out=0.5, nodes=40):
    model = RC(seed=model_seed)
    model.add(InputLayer(input_shape=(40, 2)))
    model.add(
        RandomReservoirLayer(
            nodes=nodes, density=0.2, fraction_input=fraction_input, seed=layer_seed
        )
    )
    model.add(ReadoutLayer(output_shape=(40, 1), fraction_out=fraction_out))
    model.compile()
    return model


def _same(a: RC, b: RC) -> dict:
    return dict(
        A=np.array_equal(a.reservoir_layer.weights, b.reservoir_layer.weights),
        W_in=np.array_equal(a.input_layer.weights, b.input_layer.weights),
        r0=np.array_equal(
            a.reservoir_layer.initial_res_states, b.reservoir_layer.initial_res_states
        ),
        input_nodes=sorted(a.reservoir_layer.input_receiving_nodes)
        == sorted(b.reservoir_layer.input_receiving_nodes),
        readout_nodes=sorted(a.readout_layer.readout_nodes)
        == sorted(b.readout_layer.readout_nodes),
    )


def test_same_seeds_give_the_same_model_and_predictions():
    x, y = _data()
    a, b = _model(), _model()
    assert all(_same(a, b).values())
    a.fit(x, y)
    b.fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))


def test_different_seeds_give_different_models():
    base = _model()
    assert not _same(base, _model(layer_seed=3))["A"]
    other_model_seed = _same(base, _model(model_seed=3))
    assert other_model_seed["A"]  # the graph is controlled by the layer seed only
    assert not other_model_seed["W_in"]
    assert not other_model_seed["r0"]


def test_random_streams_are_independent():
    # changing one choice must not shift the others
    a = _model(fraction_out=0.5)
    b = _model(fraction_out=0.25)
    same = _same(a, b)
    assert same["A"] and same["W_in"] and same["r0"] and same["input_nodes"]
    assert not same["readout_nodes"]


def test_global_random_state_is_neither_used_nor_modified():
    x, y = _data()
    np.random.seed(123)
    random.seed(123)
    np_state, py_state = np.random.get_state(), random.getstate()

    model = _model()
    model.fit(x, y)
    model.predict(x)

    assert random.getstate() == py_state
    new_np_state = np.random.get_state()
    assert np.array_equal(new_np_state[1], np_state[1]) and new_np_state[2:] == np_state[2:]

    # and seeding the global generators differently does not change a seeded model
    np.random.seed(1)
    random.seed(1)
    a = _model()
    np.random.seed(2)
    random.seed(2)
    b = _model()
    assert all(_same(a, b).values())


def test_compile_twice_with_seed_gives_the_same_model():
    model = _model()
    first = model.input_layer.weights.copy()
    nodes = sorted(model.readout_layer.readout_nodes)
    model.compile()
    assert np.array_equal(model.input_layer.weights, first)
    assert sorted(model.readout_layer.readout_nodes) == nodes


def test_without_seed_every_compile_differs():
    model = RC()
    model.add(InputLayer(input_shape=(40, 2)))
    model.add(RandomReservoirLayer(nodes=40, density=0.2))
    model.add(ReadoutLayer(output_shape=(40, 1)))
    model.compile()
    first = model.input_layer.weights.copy()
    model.compile()
    assert not np.array_equal(model.input_layer.weights, first)


@pytest.mark.parametrize("bad", [1.5, "3", -1, True])
def test_invalid_seed_raises(bad):
    with pytest.raises((TypeError, ValueError)):
        RC(seed=bad)


def test_er_graph_is_reproducible_incl_isolated_node_repair():
    # small sparse graphs often have isolated nodes, which are repaired randomly
    for seed in range(30):
        np.random.seed(1)
        a = gen_ER_graph(nodes=15, density=0.2, seed=seed)
        np.random.seed(2)
        b = gen_ER_graph(nodes=15, density=0.2, seed=seed)
        assert np.array_equal(a, b)


@pytest.mark.parametrize("method", ["random", "random_normal", "ones", "zeros"])
@pytest.mark.parametrize("shape", [(5,), (5, 2)])
def test_network_initializer_shapes_and_seed(method, shape):
    a = NetworkInitializer(method=method, rng=7).gen_initial_states(shape)
    b = NetworkInitializer(method=method, rng=7).gen_initial_states(shape)
    assert a.shape == shape
    assert np.array_equal(a, b)


def test_node_selector_is_seeded_and_returns_python_ints():
    a = NodeSelector(total_nodes=50, rng=3).select_nodes(fraction=0.3)
    b = NodeSelector(total_nodes=50, rng=3).select_nodes(fraction=0.3)
    c = NodeSelector(total_nodes=50, rng=4).select_nodes(fraction=0.3)
    assert a == b and a != c
    assert len(set(a)) == len(a) == 15
    assert all(type(i) is int for i in a)


def test_reservoir_computer_wrapper_is_seeded():
    x, y = _data()
    preds = []
    for _ in range(2):
        model = ReservoirComputer(num_nodes=30, density=0.2, seed=5)
        model.fit(x, y)
        preds.append(model.predict(x))
    assert np.array_equal(preds[0], preds[1])
