"""
Tests for the metrics: per-channel r2 and nrmse, rmse, shape validation, the direction
registry, and the use of the metrics in fit(), evaluate() and the pruners.
"""

import warnings

import numpy as np
import pytest
from sklearn.metrics import r2_score

from pyreco.custom_models import RC
from pyreco.edge_pruning import EdgePruner
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.metrics import (
    assign_metric,
    available_metrics,
    is_loss,
    mae,
    mse,
    nrmse,
    r2,
    rmse,
)
from pyreco.pruning import NetworkPruner


def _two_channels(seed=0, n=10, t=50):
    # channel 0: mean 0, std 1; channel 1: mean 100, std 50
    rng = np.random.default_rng(seed)
    y = np.concatenate(
        [rng.normal(size=(n, t, 1)), 100 + 50 * rng.normal(size=(n, t, 1))], axis=-1
    )
    return y, rng


# ------------------------------------------------------------------- values


def test_perfect_prediction():
    y, _ = _two_channels()
    assert mse(y, y) == mae(y, y) == rmse(y, y) == nrmse(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_mean_predictor_gives_nrmse_one_and_r2_zero():
    y, _ = _two_channels()
    mean = np.broadcast_to(y.reshape(-1, 2).mean(axis=0), y.shape)
    assert np.isclose(nrmse(y, mean), 1.0)
    assert np.isclose(r2(y, mean), 0.0)


def test_rmse_is_sqrt_of_mse():
    y, rng = _two_channels()
    pred = y + rng.normal(size=y.shape)
    assert np.isclose(rmse(y, pred), np.sqrt(mse(y, pred)))


def test_nrmse_and_r2_are_linked_for_per_channel_data():
    y, rng = _two_channels()
    pred = y + rng.normal(size=y.shape) * np.array([1.0, 40.0])
    assert np.isclose(nrmse(y, pred) ** 2, 1.0 - r2(y, pred))


def test_failing_channel_is_not_hidden_by_a_large_channel():
    # perfect on the large channel, noise on the small one
    y, rng = _two_channels()
    pred = y.copy()
    pred[:, :, 0] = rng.normal(size=y.shape[:2])
    per_channel = [
        r2_score(y[:, :, c].ravel(), pred[:, :, c].ravel()) for c in range(2)
    ]
    assert np.isclose(r2(y, pred), np.mean(per_channel))
    assert r2(y, pred) < 0.1  # pooling the channels would give 0.9997
    assert nrmse(y, pred) > 0.9  # pooling the channels would give 0.03


def test_nrmse_is_invariant_to_the_scale_of_the_data():
    y, rng = _two_channels()
    pred = y + rng.normal(size=y.shape)
    scale = np.array([1e-3, 1e4])
    assert np.isclose(nrmse(y, pred), nrmse(y * scale, pred * scale))
    assert not np.isclose(mse(y, pred), mse(y * scale, pred * scale))


def test_two_dimensional_and_one_dimensional_inputs():
    y = np.arange(20, dtype=float).reshape(10, 2)
    assert np.isclose(nrmse(y, y + 1.0), np.sqrt(np.mean(1.0 / y.var(axis=0))))
    v = np.arange(10, dtype=float)
    assert np.isclose(nrmse(v, v + 1.0), 1.0 / v.std())


# ----------------------------------------------------------- shapes, constants


@pytest.mark.parametrize("metric", [mse, mae, rmse, nrmse, r2])
def test_mismatched_shapes_raise(metric):
    y = np.zeros((4, 5, 1)) + np.arange(5)[None, :, None]
    with pytest.raises(ValueError, match="same shape"):
        metric(y, y[:, :, 0])  # would silently broadcast to (4, 5, 5) without the check


def test_constant_channel_is_left_out_with_a_warning():
    y, rng = _two_channels()
    y[:, :, 0] = 3.0
    pred = y + rng.normal(size=y.shape)
    with pytest.warns(RuntimeWarning, match="constant"):
        value = nrmse(y, pred)
    only_channel_1 = nrmse(y[:, :, 1:], pred[:, :, 1:])
    assert np.isclose(value, only_channel_1)


def test_all_constant_target_raises():
    y = np.full((4, 5, 2), 2.0)
    with pytest.raises(ValueError, match="constant"):
        nrmse(y, y + 1.0)


def test_no_warning_without_constant_channels():
    y, rng = _two_channels()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nrmse(y, y + rng.normal(size=y.shape))


# ------------------------------------------------------------------ registry


@pytest.mark.parametrize(
    "alias, name",
    [
        ("mean_squared_error", "mse"),
        ("mean_absolute_error", "mae"),
        ("root_mean_squared_error", "rmse"),
        ("normalized_root_mean_squared_error", "nrmse"),
        ("r2_score", "r2"),
    ],
)
def test_aliases_resolve_to_the_same_metric(alias, name):
    assert alias in available_metrics() and name in available_metrics()
    assert assign_metric(alias) is assign_metric(name)
    assert is_loss(alias) == is_loss(name)


def test_direction():
    assert all(is_loss(m) for m in ["mse", "mae", "rmse", "nrmse"])
    assert not is_loss("r2")


@pytest.mark.parametrize("bad", ["accuracy", 3, None])
def test_unknown_metric_raises(bad):
    with pytest.raises(ValueError, match="not implemented"):
        assign_metric(bad)
    with pytest.raises(ValueError):
        is_loss(bad)


# ----------------------------------------------------- model and the pruners


def _model(metrics):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 40, 2))
    y = np.concatenate([np.tanh(x[:, :, :1]), 100 + 50 * x[:, :, 1:2]], axis=-1)
    model = RC(seed=1)
    model.add(InputLayer(input_shape=(40, 2)))
    model.add(RandomReservoirLayer(nodes=30, density=0.2, seed=2))
    model.add(ReadoutLayer(output_shape=(40, 2)))
    model.compile(metrics=metrics)
    return model, x, y


def test_fit_history_holds_all_compiled_metrics():
    model, x, y = _model(["nrmse", "mse", "r2"])
    history = model.fit(x, y)
    assert list(history["train_metrics"]) == ["nrmse", "mse", "r2"]
    assert history["train_score"] == history["train_metrics"]["nrmse"]
    assert np.allclose(
        list(history["train_metrics"].values()), model.evaluate(x, y, metrics=["nrmse", "mse", "r2"])
    )


def test_evaluate_returns_a_list_in_the_given_order():
    model, x, y = _model("mse")
    model.fit(x, y)
    values = model.evaluate(x, y, metrics=["r2", "nrmse"])
    assert isinstance(values, list) and len(values) == 2
    assert np.isclose(values[0], 1.0 - values[1] ** 2)


@pytest.mark.parametrize("metric", ["r2", "r2_score"])
def test_edge_pruner_rejects_scores_as_performance_criterion(metric):
    with pytest.raises(ValueError, match="higher is better"):
        EdgePruner(performance_criterion=metric)


def test_edge_pruner_accepts_losses_as_criterion_and_scores_for_tracking():
    for loss in ["mse", "mae", "rmse", "nrmse"]:
        assert EdgePruner(performance_criterion=loss).performance_criterion == loss
    assert EdgePruner(metrics=["mse", "r2"]).metrics == ["mse", "r2"]


def test_node_pruner_rejects_scores_and_unknown_metrics():
    with pytest.raises(ValueError, match="higher is better"):
        NetworkPruner(criterion="r2")
    with pytest.raises(ValueError, match="not implemented"):
        NetworkPruner(criterion="accuracy")
    assert NetworkPruner(criterion="nrmse").criterion == "nrmse"
