"""
Tests for the return value of fit() (there is a single fit per call).
"""

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer


def _setup(discard_transients=0):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 40, 2))
    y = np.tanh(x[:, :, :1]) + 0.3 * x[:, :, 1:2] + 7.0
    model = RC()
    model.add(InputLayer(input_shape=(40, 2)))
    model.add(RandomReservoirLayer(nodes=40, density=0.3))
    model.add(ReadoutLayer(output_shape=(40, 1)))
    model.compile(discard_transients=discard_transients)
    return model, x, y


@pytest.mark.parametrize("discard_transients", [0, 10])
def test_train_score_matches_evaluate(discard_transients):
    # score from fit() and evaluate() are computed on the same scale
    model, x, y = _setup(discard_transients)
    history = model.fit(x, y)
    assert np.isclose(history["train_score"], model.evaluate(x, y)[0])


def test_history_keys():
    model, x, y = _setup()
    assert set(model.fit(x, y)) == {"train_score"}
    model, x, y = _setup()
    history = model.fit(x, y, store_states=True)
    assert set(history) == {"train_score", "res_states"}
    assert history["res_states"].shape[:2] == (5, 40)


def test_n_init_is_gone():
    model, x, y = _setup()
    with pytest.raises(TypeError):
        model.fit(x, y, n_init=2)


def test_fit_is_deterministic():
    model, x, y = _setup()
    model.fit(x, y)
    first = model.predict(x)
    model.fit(x, y)
    assert np.array_equal(model.predict(x), first)
