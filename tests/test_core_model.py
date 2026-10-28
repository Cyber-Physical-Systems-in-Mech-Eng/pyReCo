"""
Core model API: building and compiling, the common task shapes (vector, sequence to scalar, sequence to
sequence), hyperparameters, node and edge removal, and the selection of input and readout nodes.
Invalid data is covered in test_data_checks.py, the warm-up in test_warmup.py.
"""

from functools import partial

import numpy as np
import pytest

from helpers import build_rc, make_xy
from pyreco.custom_models import RC
from pyreco.layers import FeedbackLayer, InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.optimizers import RidgeSK
from pyreco.utils_networks import compute_spec_rad

# 8 sequences of 30 steps with 2 input channels and a 30 node reservoir; K, n_in, n_out can be overridden
_data = partial(make_xy, n=8, T=30)
_model = partial(build_rc, T=30, nodes=30, density=0.2, seed=1, layer_seed=2, readout=dict(fraction_out=0.8))


# ------------------------------------------------------------------------------- building


def test_add_assigns_layers_by_type_and_rejects_everything_else():
    model = RC()
    layers = [InputLayer((10, 1)), RandomReservoirLayer(nodes=10, density=0.3, seed=1), ReadoutLayer((10, 1))]
    for layer in layers:
        model.add(layer)
    assert (model.input_layer, model.reservoir_layer, model.readout_layer) == tuple(layers)

    feedback = FeedbackLayer((10, 1))
    model.add(feedback)
    assert model.feedback_layer is feedback

    with pytest.raises(TypeError, match="Layer"):
        model.add("not a layer")


def test_compile_flags_the_layers_and_sets_the_model_dimensions():
    model = _model(T=20, K=20, n_in=3, n_out=2, nodes=25)
    assert all(l._is_compiled for l in (model.input_layer, model.reservoir_layer, model.readout_layer))
    assert model.num_nodes == 25 and model.num_states_in == 3 and model.num_states_out == 2
    assert model.input_layer.weights.shape == (25, 3)


# ---------------------------------------------------------------------------------- shapes


@pytest.mark.parametrize(
    "T, K, d, n_in, n_out",
    [
        (1, 1, 0, 1, 1),  # vector to vector
        (20, 1, 19, 1, 1),  # sequence to scalar (the last state is paired with the target)
        (20, 20, 0, 2, 2),  # sequence to sequence with several states
        (25, 20, 5, 3, 1),  # targets shorter than the inputs, warm-up 5
    ],
)
def test_common_task_shapes(T, K, d, n_in, n_out):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(10, T, n_in)), rng.normal(size=(10, K, n_out))
    model = _model(T=T, K=K, n_in=n_in, n_out=n_out, discard_transients=d)
    model.fit(x, y)
    assert model.predict(x).shape == (10, T - d, n_out)
    assert len(model.evaluate(x, y, metrics=["mse", "mae"])) == 2


def test_sequence_to_scalar_learns_a_function_of_the_last_input():
    # target: mean of the last three inputs. The last state has to remember them.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 20, 1))
    y = x[:, -3:, :].mean(axis=1, keepdims=True)
    model = RC(seed=0)
    model.add(InputLayer((20, 1)))
    model.add(RandomReservoirLayer(nodes=60, density=0.2, leakage_rate=1.0, seed=0))
    model.add(ReadoutLayer((1, 1)))
    model.compile(discard_transients=19, metrics=["nrmse"])
    model.set_hp(alpha=1e-6)
    model.fit(x[:300], y[:300])
    assert model.evaluate(x[300:], y[300:])[0] < 0.2


# ---------------------------------------------------------------------- hyperparameters


def test_set_hp_and_get_hp_round_trip():
    """Every hyperparameter that is set can be read back unchanged."""
    model = _model()
    model.set_hp(spec_rad=0.7, leakage_rate=0.3, activation="sigmoid", alpha=0.01, input_scaling=[0.5, 2.0])
    hp = model.get_hp()
    assert np.isclose(hp["spec_rad"], 0.7) and hp["leakage_rate"] == 0.3
    assert hp["activation"] == "sigmoid" and hp["alpha"] == 0.01
    assert np.allclose(hp["input_scaling"], [0.5, 2.0])
    assert set(model.get_hp("spec_rad", "alpha")) == {"spec_rad", "alpha"}


def test_set_hp_changes_the_reservoir_and_the_readout():
    """spec_rad rescales the weights to rho(A) = spec_rad, the activation is the named function, and the
    ridge penalty alpha changes the fitted readout (a strong penalty shrinks the weights)."""
    x, y = _data()
    model = _model()
    model.fit(x, y)

    model.set_hp(spec_rad=0.9)
    assert np.isclose(compute_spec_rad(model.reservoir_layer.weights), 0.9)

    model.set_hp(leakage_rate=0.15)
    assert np.isclose(model.get_hp()["leakage_rate"], 0.15)

    v = np.array([0.5, -1.0, 0.0])
    model.set_hp(activation="tanh")
    assert np.allclose(model.reservoir_layer.activation_fun(v), np.tanh(v))
    model.set_hp(activation="sigmoid")  # 1 / (1 + exp(-v))
    assert np.allclose(model.reservoir_layer.activation_fun(v), 1 / (1 + np.exp(-v)))

    norms = {}
    for alpha in (1e-4, 1e1):
        model.set_hp(alpha=alpha)
        assert np.isclose(model.get_hp("alpha")["alpha"], alpha)
        model.fit(x, y)
        norms[alpha] = np.linalg.norm(model.readout_layer.weights)
    assert norms[1e1] < norms[1e-4]


def test_set_hp_rejects_unknown_or_invalid_values():
    """Unknown names and values outside the valid range are errors, not ignored."""
    model = _model()
    with pytest.raises(ValueError, match="Unsupported"):
        model.set_hp(learning_rate=0.1)
    with pytest.raises(ValueError):
        model.set_hp(leakage_rate=0.0)
    with pytest.raises(ValueError):
        model.set_hp(activation="relu")
    with pytest.raises(ValueError):
        model.set_hp(alpha=-1.0)


def test_the_ridge_penalty_of_a_given_optimizer_is_settable():
    model = _model(optimizer=RidgeSK())
    model.set_hp(alpha=1e-3)
    assert model.optimizer.get_alpha() == 1e-3


def test_unknown_optimizer_and_metric_are_rejected_at_compile():
    with pytest.raises(ValueError, match="not implemented"):
        _model(optimizer="lasso")
    with pytest.raises(ValueError, match="not implemented"):
        _model(metrics=["accuracy"])


# ---------------------------------------------------------------------------- node removal


def test_remove_reservoir_nodes_shrinks_every_dependent_array_and_the_model_still_trains():
    x, y = _data()
    model = _model(nodes=30)
    model.fit(x, y)
    readout_before = list(model.readout_layer.readout_nodes)
    model.remove_reservoir_nodes([0, 5, 6])

    layer = model.reservoir_layer
    assert layer.nodes == 27 and layer.weights.shape == (27, 27)
    assert model.input_layer.weights.shape == (27, 2)
    assert layer.initial_res_states.shape == (27,)
    assert max(model.readout_layer.readout_nodes) < 27
    assert len(model.readout_layer.readout_nodes) == len(set(readout_before) - {0, 5, 6})
    assert max(layer.input_receiving_nodes) < 27

    model.fit(x, y)  # the readout is retrained for the smaller reservoir
    assert model.predict(x).shape == (8, 30, 1)


@pytest.mark.parametrize(
    "nodes, error",
    [
        ((1, 2), TypeError),  # not a list
        ([31], ValueError),  # beyond the reservoir
        ([-1], ValueError),
        (list(range(30)), ValueError),  # would remove everything
    ],
)
def test_remove_reservoir_nodes_rejects_invalid_requests(nodes, error):
    model = _model(nodes=30)
    with pytest.raises(error):
        model.remove_reservoir_nodes(nodes)
    assert model.reservoir_layer.nodes == 30  # nothing happened


def test_remove_reservoir_edges_zeroes_exactly_these_edges():
    model = _model(nodes=30)
    weights = model.reservoir_layer.weights
    rows, cols = np.nonzero(weights)
    edges = [(int(rows[0]), int(cols[0])), (int(rows[1]), int(cols[1]))]
    model.remove_reservoir_edges(edges)
    assert all(model.reservoir_layer.weights[u, v] == 0 for u, v in edges)
    assert np.count_nonzero(model.reservoir_layer.weights) == len(rows) - 2

    with pytest.raises(TypeError, match="list"):
        model.remove_reservoir_edges((0, 1))


def test_read_in_weights_have_to_match_the_reservoir_and_the_input():
    model = _model(nodes=30, n_in=2)
    model._set_readin_weights(np.ones((30, 2)))
    assert np.all(model.input_layer.weights == 1.0)
    model._set_readin_weights(np.zeros((30, 2)).tolist())  # lists are accepted
    with pytest.raises(ValueError, match="shape"):
        model._set_readin_weights(np.ones((29, 2)))
    with pytest.raises(TypeError, match="numpy array or a list"):
        model._set_readin_weights("weights")


# ---------------------------------------------------------------------- set_normalization


def test_set_normalization_validates_its_argument():
    model = _model()
    model.set_normalization(normalize_inputs=False)
    assert model.normalize_inputs is False
    model.set_normalization()  # None leaves the setting unchanged
    assert model.normalize_inputs is False
    with pytest.raises(TypeError, match="boolean"):
        model.set_normalization(normalize_inputs="yes")


# ------------------------------------------------- no input nodes, no readout nodes, no compile


def test_a_fraction_that_selects_no_input_node_is_an_error():
    model = RC(seed=0)
    model.add(InputLayer((20, 2)))
    model.add(RandomReservoirLayer(nodes=5, density=0.5, fraction_input=0.1, seed=0))
    model.add(ReadoutLayer((20, 1)))
    with pytest.raises(ValueError, match="selects 0 of the 5 reservoir nodes to receive the input"):
        model.compile()


def test_a_fraction_that_selects_no_readout_node_is_an_error():
    model = RC(seed=0)
    model.add(InputLayer((20, 2)))
    model.add(RandomReservoirLayer(nodes=5, density=0.5, seed=0))
    model.add(ReadoutLayer((20, 1), fraction_out=0.1))
    with pytest.raises(ValueError, match="selects 0 of the 5 reservoir nodes to be read out"):
        model.compile()


def test_the_smallest_fractions_that_select_a_node_work():
    model = RC(seed=0)
    model.add(InputLayer((20, 2)))
    model.add(RandomReservoirLayer(nodes=5, density=0.5, fraction_input=0.3, seed=0))
    model.add(ReadoutLayer((20, 1), fraction_out=0.3))
    model.compile()
    assert len(model.reservoir_layer.input_receiving_nodes) == 2
    assert len(model.readout_layer.readout_nodes) == 2


def test_explicitly_empty_readout_nodes_are_rejected():
    model = _model()
    with pytest.raises(ValueError, match="At least one"):
        model._set_readout_nodes([])


def test_removing_all_input_nodes_is_caught_at_fit():
    x, y = _data()
    model = RC(seed=0)
    model.add(InputLayer((30, 2)))
    model.add(RandomReservoirLayer(nodes=10, density=0.3, fraction_input=0.2, seed=0))
    model.add(ReadoutLayer((30, 1)))
    model.compile()
    model.remove_reservoir_nodes(list(model.reservoir_layer.input_receiving_nodes))
    with pytest.raises(ValueError, match="No reservoir node receives the input"):
        model.fit(x, y)


def test_removing_all_readout_nodes_is_caught_at_fit():
    x, y = _data()
    model = RC(seed=0)
    model.add(InputLayer((30, 2)))
    model.add(RandomReservoirLayer(nodes=10, density=0.3, seed=0))
    model.add(ReadoutLayer((30, 1), fraction_out=0.2))
    model.compile()
    model.remove_reservoir_nodes(list(model.readout_layer.readout_nodes))
    with pytest.raises(ValueError, match="No reservoir node is read out"):
        model.fit(x, y)


def test_nodes_can_be_removed_before_the_first_fit_and_the_fraction_follows():
    model = RC(seed=0)
    model.add(InputLayer((30, 2)))
    model.add(RandomReservoirLayer(nodes=20, density=0.3, seed=0))
    model.add(ReadoutLayer((30, 1), fraction_out=0.5))
    model.compile()
    assert len(model.readout_layer.readout_nodes) == 10
    model.remove_reservoir_nodes([0, 1, 2, 3])
    n_readout = len(model.readout_layer.readout_nodes)
    assert np.isclose(model.readout_layer.fraction_out, n_readout / 16)
    x, y = _data()
    model.fit(x, y)
    assert model.predict(x).shape == (8, 30, 1)
