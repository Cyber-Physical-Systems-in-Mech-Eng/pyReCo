"""
Accuracy regression on the benchmark systems: a fixed seeded reservoir must keep learning them.

The thresholds are about 1.5 times the values measured when the tests were written (NRMSE is 1
for predicting the mean). They guard against a change that silently breaks the reservoir
dynamics, the warm-up handling or the readout training.
"""

import pytest

from pyreco.custom_models import RC
from pyreco.datasets import load
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer


def _test_nrmse(x_train, y_train, x_test, y_test, discard_transients=0, alpha=None, **reservoir):
    model = RC(seed=0)
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=200, density=0.1, seed=0, **reservoir))
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    model.compile(discard_transients=discard_transients)
    if alpha is not None:
        model.set_hp(alpha=alpha)
    model.fit(x_train, y_train)
    return model.evaluate(x_test, y_test, metrics=["nrmse"])[0]


@pytest.mark.parametrize(
    "name, limit",
    [("lorenz", 0.07), ("mackey_glass", 0.12)],  # measured: 0.043, 0.074
)
def test_one_step_forecast_of_chaotic_systems(name, limit):
    x_train, y_train, x_test, y_test, _ = load(
        name, n_samples=2500, n_in=30, n_out=1, standardize=True
    )
    nrmse = _test_nrmse(
        x_train, y_train, x_test, y_test, discard_transients=29,  # n_in - n_out
        spec_rad=0.9, leakage_rate=0.5,
    )
    assert nrmse < limit


@pytest.mark.parametrize(
    "name, limit",
    [("narma5", 0.6), ("narma10", 0.85)],  # measured: 0.393, 0.585 (the mean predictor gives 1)
)
def test_narma(name, limit):
    x_train, y_train, x_test, y_test = load(name, n_samples=3000, n_in=50, n_out=1, seed=1)
    nrmse = _test_nrmse(
        x_train, y_train, x_test, y_test, discard_transients=49,  # n_in - n_out
        alpha=1e-6, spec_rad=0.9, leakage_rate=0.9,
    )
    assert nrmse < limit
