"""
fit simulates the reservoir once, and the activation functions are stable for large pre-activations.
"""

import warnings

import numpy as np
import pytest

from helpers import build_rc, make_xy
from pyreco.layers import RandomReservoirLayer


def test_fit_simulates_the_reservoir_once_and_scores_like_predict(monkeypatch):
    x, y = make_xy(n=6, T=40, n_in=2, seed=0)
    model = build_rc(nodes=20, seed=0)

    calls = []
    original = type(model).compute_reservoir_state
    monkeypatch.setattr(type(model), "compute_reservoir_state", lambda self, x_: calls.append(1) or original(self, x_))
    history = model.fit(x, y)
    assert len(calls) == 1

    # the training score is the one of predict() on the training data
    y_pred = model.predict(x)[:, : y.shape[1]]
    assert history["train_score"] == pytest.approx(float(np.mean((y - y_pred) ** 2)))


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_activations_do_not_overflow(activation):
    layer = RandomReservoirLayer(nodes=5, activation=activation)
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = layer.activation_fun(z)
    assert np.all(np.isfinite(out))
    expected = np.tanh(z) if activation == "tanh" else 1 / (1 + np.exp(-np.clip(z, -700, 700)))
    assert out == pytest.approx(expected)
