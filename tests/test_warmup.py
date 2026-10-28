"""
Warm-up and initial state of the reservoir, checked against an independent implementation of

    r(t+1) = (1 - a) r(t) + a tanh(r(t) A + W_in x~(t)),     x~ = scaling * (x - mean) / std

The target at time step j is paired with the state r(d + j + 1), d being `discard_transients`.
"""

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer


def _model(T, K, d=0, n_in=2, nodes=40, fraction_out=0.8, **layer_kwargs):
    model = RC(seed=3)
    model.add(InputLayer(input_shape=(T, n_in)))
    model.add(
        RandomReservoirLayer(
            nodes=nodes, density=0.2, leakage_rate=0.4, seed=5, fraction_input=0.7, **layer_kwargs
        )
    )
    model.add(ReadoutLayer(output_shape=(K, 1), fraction_out=fraction_out))
    model.compile(discard_transients=d)
    return model


def _reference_states(model, x):
    xt = model._input_scaling_fit * (x - model.input_mean) / model.input_std
    A, W_in = model.reservoir_layer.weights, model.input_layer.weights
    a = model.reservoir_layer.leakage_rate
    r = np.zeros((x.shape[0], x.shape[1] + 1, A.shape[0]))
    r[:, 0] = model.reservoir_layer.initial_res_states
    for t in range(x.shape[1]):
        r[:, t + 1] = (1 - a) * r[:, t] + a * np.tanh(r[:, t] @ A + xt[:, t] @ W_in.T)
    return r


# ------------------------------------------------------------ pairing of states and targets


@pytest.mark.parametrize(
    "T, K, d",
    [(30, 30, 0), (30, 23, 7), (30, 1, 29), (30, 1, 0), (30, 5, 3), (30, 10, 20), (30, 29, 1), (30, 25, 5)],
)
def test_target_j_is_paired_with_state_d_plus_j_plus_1(T, K, d):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(6, T, 2)), rng.normal(size=(6, K, 1))
    model = _model(T, K, d)
    model.fit(x, y)

    states = _reference_states(model, x)[:, d + 1 : d + 1 + K][:, :, model.readout_layer.readout_nodes]

    # predictions: one per input step after the warm-up; the first K belong to the targets
    prediction = model.predict(x)
    assert prediction.shape == (6, T - d, 1)
    reference = states @ model.readout_layer.weights + model.readout_layer.bias
    assert np.allclose(prediction[:, :K], reference)

    # the readout is the ridge solution of exactly this pairing (centered, intercept separate)
    A, b = states.reshape(-1, states.shape[-1]), y.reshape(-1, 1)
    A_c, b_c = A - A.mean(0), b - b.mean(0)
    alpha = model.optimizer.alpha
    W = np.linalg.solve(A_c.T @ A_c + alpha * np.eye(A.shape[1]), A_c.T @ b_c)
    assert np.allclose(model.readout_layer.weights, W, atol=1e-8)


def test_default_warm_up_is_zero_and_all_inputs_are_used():
    model = _model(30, 30)
    assert model.discard_transients == 0
    x = np.random.default_rng(0).normal(size=(4, 30, 2))
    model.fit(x, np.random.default_rng(1).normal(size=(4, 30, 1)))
    assert model.predict(x).shape == (4, 30, 1)


@pytest.mark.parametrize("T, K, d", [(30, 30, 1), (30, 23, 8), (30, 1, 30), (30, 10, 21)])
def test_warm_up_larger_than_the_extra_inputs_is_rejected(T, K, d):
    rng = np.random.default_rng(0)
    model = _model(T, K, d)
    with pytest.raises(ValueError, match="must not exceed"):
        model.fit(rng.normal(size=(4, T, 2)), rng.normal(size=(4, K, 1)))


def test_largest_possible_warm_up_pairs_the_last_states():
    # sequence to scalar: warm-up T - 1 pairs the only target with the state after all T inputs
    T = 25
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(8, T, 2)), rng.normal(size=(8, 1, 1))
    model = _model(T, 1, T - 1)
    model.fit(x, y)
    last_state = _reference_states(model, x)[:, T, model.readout_layer.readout_nodes]
    assert np.allclose(
        model.predict(x)[:, 0], last_state @ model.readout_layer.weights + model.readout_layer.bias
    )


def test_evaluate_compares_the_predictions_with_the_targets_of_the_same_time_steps():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(6, 30, 2)), rng.normal(size=(6, 20, 1))
    model = _model(30, 20, 4)
    model.fit(x, y)
    reference = float(np.mean((model.predict(x)[:, :20] - y) ** 2))
    assert np.isclose(model.evaluate(x, y, metrics=["mse"])[0], reference)


@pytest.mark.parametrize("bad, error", [(-1, ValueError), (2.5, TypeError), ("3", TypeError)])
def test_invalid_discard_transients_in_compile(bad, error):
    with pytest.raises(error):
        _model(30, 30, bad)


# -------------------------------------------------------------------------- initial state


def test_initial_state_is_zero_by_default():
    model = _model(20, 20)
    assert np.array_equal(model.reservoir_layer.initial_res_states, np.zeros(40))


@pytest.mark.parametrize("method", ["ones", "random", "random_normal"])
def test_other_initial_states_can_be_requested(method):
    state = _model(20, 20, init_res_sampling=method).reservoir_layer.initial_res_states
    assert state.shape == (40,)
    assert np.isclose(np.abs(state).max(), 1.0)  # scaled to a maximum absolute value of 1
    if method == "ones":
        assert np.all(state == 1.0)


def test_unknown_initial_state_method_is_rejected():
    with pytest.raises(ValueError, match="init_res_sampling"):
        RandomReservoirLayer(nodes=10, density=0.3, init_res_sampling="gaussian")


def test_explicit_initial_state_is_used_in_training_and_prediction():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(5, 25, 2)), rng.normal(size=(5, 25, 1))
    model = _model(25, 25)
    r0 = rng.uniform(-1, 1, size=40)
    model.reservoir_layer.set_initial_state(r0)
    model.fit(x, y)

    assert np.array_equal(model.reservoir_layer.initial_res_states, r0)
    states = _reference_states(model, x)[:, 1:][:, :, model.readout_layer.readout_nodes]
    assert np.allclose(model.predict(x), states @ model.readout_layer.weights + model.readout_layer.bias)

    with pytest.raises(ValueError, match="initial reservoir state"):
        model.reservoir_layer.set_initial_state(np.zeros(7))


def test_a_nonzero_start_changes_the_first_states_but_the_effect_fades():
    x = np.random.default_rng(0).normal(size=(3, 120, 2))
    differences = []
    for r0 in (np.zeros(40), np.full(40, 0.8)):
        model = _model(120, 120, n_in=2)
        model.reservoir_layer.set_initial_state(r0)
        model.fit(x, np.random.default_rng(1).normal(size=(3, 120, 1)))
        differences.append(_reference_states(model, x))
    gap = np.linalg.norm(differences[0] - differences[1], axis=-1).mean(axis=0)
    assert gap[1] > 0.1 and gap[-1] < 1e-3 * gap[1]  # echo state property: the start is forgotten


# ----------------------------------------------------------------------------- dynamics


def test_one_step_of_the_reservoir_update_by_hand():
    # two nodes, leakage 0.5, tanh: r(1) = 0.5 r(0) + 0.5 tanh(r(0) A + W_in x(0))
    model = _model(5, 5, nodes=4)
    A = np.zeros((4, 4))
    A[0, 1], A[1, 0], A[2, 3] = 0.5, -0.25, 0.3
    W_in = np.zeros((4, 2))
    W_in[0, 0], W_in[1, 1], W_in[2, 0] = 1.0, -0.5, 0.2
    model.reservoir_layer.weights = A
    model.input_layer.weights = W_in
    model.reservoir_layer.leakage_rate = 0.5
    r0 = np.array([0.1, -0.2, 0.3, 0.4])
    model.reservoir_layer.initial_res_states = r0

    x = np.array([[[0.7, -1.2]]])
    states = model.compute_reservoir_state(x)
    drive = r0 @ A + W_in @ x[0, 0]
    expected = 0.5 * r0 + 0.5 * np.tanh(drive)
    assert np.allclose(states[0, 0], r0)
    assert np.allclose(states[0, 1], expected)
    # by hand for node 0: r(0)=0.1; drive = 0.5*(-0.2)*... A[1,0] = -0.25 acts on node 0
    assert np.isclose(states[0, 1, 0], 0.5 * 0.1 + 0.5 * np.tanh(-0.2 * -0.25 + 0.7))


def test_without_input_a_contracting_reservoir_returns_to_zero():
    model = _model(60, 60, nodes=30)
    model.reservoir_layer.set_spec_rad(0.5)
    model.reservoir_layer.leakage_rate = 1.0
    model.input_layer.weights = np.zeros_like(model.input_layer.weights)
    model.reservoir_layer.set_initial_state(np.random.default_rng(0).uniform(-1, 1, size=30))
    states = model.compute_reservoir_state(np.zeros((1, 60, 2)))
    assert np.abs(states[0, -1]).max() < 1e-6
    assert np.abs(states[0, 0]).max() > 0.1


def test_evaluate_rejects_targets_that_are_longer_than_the_predictions():
    # targets that still contain the warm-up used to be compared with the wrong predictions
    # without any message (prediction j was paired with target j instead of d + j)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 80, 2))
    y = rng.normal(size=(1, 80, 1))
    model = _model(80, 70, d=10)
    model.fit(x, y[:, 10:])
    assert model.predict(x).shape == (1, 70, 1)
    with pytest.raises(ValueError, match="leave out the first 10 targets"):
        model.evaluate(x, y)
    assert np.isfinite(model.evaluate(x, y[:, 10:])).all()
    # shorter targets (e.g. the first few steps, or a single target) stay valid
    assert np.isfinite(model.evaluate(x, y[:, 10:20])).all()
