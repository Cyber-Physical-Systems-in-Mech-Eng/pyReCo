"""
Tests for the hyperparameter search (pyreco.tuner.Tuner, built on optuna).
"""

import io
import sys

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.tuner import Tuner
from pyreco.utils_data import sequence_to_sequence

SEARCH_SPACE = {
    "spec_rad": ("float", 0.3, 1.0),
    "leakage_rate": ("float", 0.1, 1.0),
    "activation": ("categorical", ["tanh", "sigmoid"]),
    "alpha": ("float", 1e-6, 1.0, "log"),
}


@pytest.fixture(scope="module")
def data():
    return sequence_to_sequence("sine_pred", n_batch=30, n_states=1, n_time=20, seed=0)


def _model(data):
    X_train, _, y_train, _ = data
    model = RC(seed=1)
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=30, density=0.2, seed=2))
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    model.compile(metrics=["mse"])
    return model


def _tuner(data, **kwargs):
    X_train, X_test, y_train, y_test = data
    options = dict(
        search_space=SEARCH_SPACE,
        x_train=X_train,
        y_train=y_train,
        x_val=X_test,
        y_val=y_test,
        n_trials=6,
    )
    options.update(kwargs)
    return Tuner(_model(data), **options)


def test_optimize_returns_the_best_model_and_importances(data):
    tuner = _tuner(data)
    study = tuner.optimize()

    assert len(study.trials) == 6
    assert np.isclose(tuner.best_score, study.best_value)
    assert isinstance(tuner.best_model, RC)
    assert set(tuner.importances) == set(SEARCH_SPACE)

    # the stored best model carries the best hyperparameters
    best = study.best_trial.params
    hp = tuner.best_model.get_hp()
    assert np.isclose(hp["spec_rad"], best["spec_rad"], atol=1e-6)
    assert np.isclose(hp["leakage_rate"], best["leakage_rate"])
    assert hp["activation"] == best["activation"]
    assert np.isclose(hp["alpha"], best["alpha"])

    # and reproduces the best validation score
    _, X_test, _, y_test = data
    assert np.isclose(tuner.best_model.evaluate(X_test, y_test)[0], tuner.best_score)


def test_report_runs_after_optimize(data, capsys):
    tuner = _tuner(data, n_trials=3)
    tuner.optimize()
    capsys.readouterr()
    tuner.report()
    assert "Best Params" in capsys.readouterr().out


def test_verbose_output_survives_a_non_unicode_console(data, monkeypatch):
    # a Windows console with a legacy code page used to crash on the arrow in the trial log
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    _tuner(data, n_trials=3, verbose=True).optimize()
    stream.flush()


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_samplers(data, sampler):
    assert len(_tuner(data, sampler_type=sampler, n_trials=3).optimize().trials) == 3


def test_grid_sampler_visits_a_small_discrete_space(data):
    space = {"activation": ("categorical", ["tanh", "sigmoid"])}
    tuner = _tuner(data, search_space=space, sampler_type="grid", n_trials=2)
    study = tuner.optimize()
    assert {t.params["activation"] for t in study.trials} == {"tanh", "sigmoid"}


def test_maximize_picks_the_largest_score(data):
    scores = iter([0.1, 0.9, 0.5])
    tuner = _tuner(
        data, direction="maximize", n_trials=3, evaluator=lambda model, trial: next(scores)
    )
    study = tuner.optimize()
    assert tuner.best_score == 0.9 and study.best_value == 0.9


def test_custom_evaluator_overrides_the_validation_score(data):
    tuner = _tuner(data, n_trials=3, evaluator=lambda model, trial: 1.0 + trial.number)
    tuner.optimize()
    assert tuner.best_score == 1.0


def test_invalid_scores_never_become_the_best(data):
    scores = iter([float("nan"), 0.7, float("inf")])
    tuner = _tuner(data, n_trials=3, evaluator=lambda model, trial: next(scores))
    tuner.optimize()
    assert tuner.best_score == 0.7


def test_invalid_arguments_raise(data):
    with pytest.raises(ValueError, match="direction"):
        _tuner(data, direction="sideways")
    with pytest.raises(ValueError, match="Unsupported parameter type"):
        _tuner(data, search_space={"spec_rad": ("complex", 0, 1)}, n_trials=1).optimize()
    with pytest.raises(ValueError, match="evaluator or"):
        _tuner(data, x_val=None, y_val=None, n_trials=1).optimize()
