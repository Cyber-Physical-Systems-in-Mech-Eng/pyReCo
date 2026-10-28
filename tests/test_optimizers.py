"""
Tests for the readout optimizer: the ridge penalty alpha must be the one that was asked for.
"""

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.optimizers import RidgeSK, assign_optimizer


def _regression(n=200, k=15, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, k))
    b = A @ rng.normal(size=(k, 2)) + 0.1 * rng.normal(size=(n, 2))
    return A, b


def _ridge(A, b, alpha):
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)


def test_default_alpha_is_one():
    assert RidgeSK().alpha == 1.0 and RidgeSK().get_alpha() == 1.0
    assert assign_optimizer("ridge").alpha == 1.0 and assign_optimizer("Ridge").alpha == 1.0


@pytest.mark.parametrize("alpha", [0.0, 1e-6, 0.5, 3, 100.0])
def test_constructor_alpha_is_used(alpha):
    optimizer = RidgeSK(alpha=alpha)
    assert optimizer.get_alpha() == alpha
    A, b = _regression()
    assert np.allclose(optimizer.solve(A, b), _ridge(A, b, alpha), atol=1e-8)


def test_alpha_changes_the_solution():
    """A larger ridge penalty alpha in min |R W - y|^2 + alpha |W|^2 changes (shrinks) the solution."""
    A, b = _regression()
    weak, strong = RidgeSK(alpha=1e-6).solve(A, b), RidgeSK(alpha=1e3).solve(A, b)
    assert np.linalg.norm(strong) < 0.5 * np.linalg.norm(weak)


def test_set_alpha_after_construction():
    optimizer = RidgeSK(alpha=2.0)
    optimizer.set_alpha(0.01)
    assert optimizer.get_alpha() == 0.01
    A, b = _regression()
    assert np.allclose(optimizer.solve(A, b), _ridge(A, b, 0.01), atol=1e-8)


@pytest.mark.parametrize("bad, error", [(-0.1, ValueError), (np.nan, ValueError), (np.inf, ValueError), ("1", TypeError), (None, TypeError), (True, TypeError)])
def test_invalid_alpha_is_rejected(bad, error):
    with pytest.raises(error):
        RidgeSK(alpha=bad)
    optimizer = RidgeSK()
    with pytest.raises(error):
        optimizer.set_alpha(bad)
    assert optimizer.alpha == 1.0  # unchanged


def test_a_given_optimizer_instance_is_used_by_the_model():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(6, 25, 2)), rng.normal(size=(6, 25, 1))
    weights = {}
    for alpha in (1e-6, 1e3):
        model = RC(seed=0)
        model.add(InputLayer((25, 2)))
        model.add(RandomReservoirLayer(nodes=30, density=0.2, seed=1))
        model.add(ReadoutLayer((25, 1)))
        model.compile(optimizer=RidgeSK(alpha=alpha))
        assert model.get_hp("alpha") == {"alpha": alpha}
        model.fit(x, y)
        weights[alpha] = model.readout_layer.weights
    assert np.linalg.norm(weights[1e3]) < 0.1 * np.linalg.norm(weights[1e-6])


def test_unknown_optimizer_names_and_types():
    with pytest.raises(ValueError, match="not implemented"):
        assign_optimizer("lasso")
    with pytest.raises(TypeError, match="Optimizer"):
        assign_optimizer(3)
