"""
Tests for the return value of fit() (there is a single fit per call).
"""

import numpy as np
import pytest

from helpers import build_rc, make_xy


def _setup(discard_transients=0):
    # the targets are shorter than the inputs by the warm-up
    x, y = make_xy(n=5, T=40, K=40 - discard_transients)
    return build_rc(T=40, K=40 - discard_transients, nodes=40, discard_transients=discard_transients), x, y + 7.0


@pytest.mark.parametrize("discard_transients", [0, 10])
def test_train_score_matches_evaluate(discard_transients):
    # score from fit() and evaluate() are computed on the same scale
    model, x, y = _setup(discard_transients)
    history = model.fit(x, y)
    assert np.isclose(history["train_score"], model.evaluate(x, y)[0])


def test_history_keys():
    model, x, y = _setup()
    assert set(model.fit(x, y)) == {"train_score", "train_metrics"}
    model, x, y = _setup()
    history = model.fit(x, y, store_states=True)
    assert set(history) == {"train_score", "train_metrics", "res_states"}
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
