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


# ----------------------------------------------------------------------------------- seed


def _trial_values(study):
    return [(t.params, t.value) for t in study.trials]


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_same_seed_gives_the_same_search(data, sampler):
    runs = []
    for _ in range(2):
        tuner = _tuner(data, sampler_type=sampler, n_trials=5, seed=7)
        runs.append((_trial_values(tuner.optimize()), tuner.best_score, tuner.best_model.get_hp()))
    assert runs[0] == runs[1]


def test_different_seeds_propose_different_hyperparameters(data):
    first = _trial_values(_tuner(data, n_trials=4, seed=1).optimize())
    second = _trial_values(_tuner(data, n_trials=4, seed=2).optimize())
    assert [p for p, _ in first] != [p for p, _ in second]


def test_grid_sampler_accepts_a_seed(data):
    space = {"activation": ("categorical", ["tanh", "sigmoid"])}
    study = _tuner(data, search_space=space, sampler_type="grid", n_trials=2, seed=3).optimize()
    assert {t.params["activation"] for t in study.trials} == {"tanh", "sigmoid"}


def test_without_a_seed_runs_differ(data):
    first = _trial_values(_tuner(data, n_trials=4).optimize())
    second = _trial_values(_tuner(data, n_trials=4).optimize())
    assert [p for p, _ in first] != [p for p, _ in second]


@pytest.mark.parametrize("bad", [1.5, "3", -1, True])
def test_invalid_seed_is_rejected(data, bad):
    with pytest.raises((TypeError, ValueError)):
        _tuner(data, seed=bad)


# --------------------------------------------------------- the model of the caller is left alone


def test_the_model_that_is_passed_in_is_not_modified(data):
    X_train, X_test, y_train, y_test = data
    model = _model(data)
    model.fit(X_train, y_train)
    hp_before, prediction_before = model.get_hp(), model.predict(X_test)
    weights_before = model.reservoir_layer.weights.copy()

    tuner = Tuner(model, SEARCH_SPACE, X_train, y_train, x_val=X_test, y_val=y_test, n_trials=5, seed=0)
    tuner.optimize()

    assert model.get_hp() == hp_before
    assert np.array_equal(model.predict(X_test), prediction_before)
    assert np.array_equal(model.reservoir_layer.weights, weights_before)
    assert tuner.best_model is not model
    assert tuner.best_model.get_hp() != hp_before


def test_the_best_model_is_independent_of_later_trials(data):
    # the stored best model must not change when later trials retrain the working copy
    _, X_test, _, y_test = data
    tuner = _tuner(data, n_trials=6, seed=1)
    tuner.optimize()
    assert np.isclose(tuner.best_model.evaluate(X_test, y_test)[0], tuner.best_score)


def test_optimizing_twice_starts_from_the_original_model(data):
    first = _tuner(data, n_trials=4, seed=5)
    first_values = _trial_values(first.optimize())
    # a second optimization of the same tuner proposes the same hyperparameters again with a
    # fresh sampler and gives the same scores, because nothing carries over from the first run
    first.best_score, first.best_model = float("inf"), None
    assert _trial_values(first.optimize()) == first_values


def test_the_evaluator_receives_the_working_copy(data):
    seen = []
    tuner = _tuner(data, n_trials=2, evaluator=lambda model, trial: seen.append(model) or 1.0)
    tuner.optimize()
    assert all(m is not tuner.model for m in seen)


# ------------------------------------------------------------------------------ quiet output


def test_a_quiet_tuner_prints_nothing_and_restores_the_logging_level(data, capsys):
    import optuna

    levels = []
    optuna.logging.set_verbosity(optuna.logging.INFO)
    tuner = _tuner(data, n_trials=3, evaluator=lambda model, trial: levels.append(optuna.logging.get_verbosity()) or 1.0)
    tuner.optimize()

    assert levels == [optuna.logging.WARNING] * 3  # during the trials
    assert optuna.logging.get_verbosity() == optuna.logging.INFO  # restored
    assert capsys.readouterr().out == ""
    assert tuner.importances is not None


def test_the_logging_level_is_restored_when_a_trial_fails(data):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO)
    with pytest.raises(ValueError):
        _tuner(data, search_space={"nodes": ("int", 5, 20)}, n_trials=1).optimize()
    assert optuna.logging.get_verbosity() == optuna.logging.INFO


def test_a_verbose_tuner_prints_the_trials_and_the_importances(data, capsys):
    import optuna

    levels = []
    tuner = _tuner(
        data, n_trials=3, verbose=True,
        evaluator=lambda model, trial: levels.append(optuna.logging.get_verbosity()) or 1.0,
    )
    tuner.optimize()
    out = capsys.readouterr().out
    assert "Trial 00" in out and "Hyperparameter importances" in out
    assert levels == [optuna.logging.get_verbosity()] * 3  # the level of the caller is kept


def test_a_single_trial_gives_no_importances_and_no_warning(data):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tuner = _tuner(data, n_trials=1)
        tuner.optimize()
    assert tuner.importances == {}


def test_saving_the_best_model_is_quiet_unless_verbose(data, tmp_path, capsys):
    import joblib

    tuner = _tuner(data, n_trials=2)
    tuner.optimize()
    tuner.save_best_model(tmp_path / "best.joblib")
    assert capsys.readouterr().out == ""
    assert joblib.load(tmp_path / "best.joblib").get_hp() == tuner.best_model.get_hp()


def test_saving_without_a_best_model_warns(data, tmp_path):
    with pytest.warns(UserWarning, match="No best model"):
        _tuner(data).save_best_model(tmp_path / "none.joblib")
    assert not (tmp_path / "none.joblib").exists()
