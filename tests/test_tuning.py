import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import from your actual pyreco module structure
from pyreco.tuner import (
    TPETuner, GridTuner, RandomTuner, GPTuner,
    ReservoirObjective, BaseTuner
)

from pyreco.utils_data import sequence_to_sequence, sequence_to_scalar



def test_tuning(search_space, X_train, y_train, X_test, y_test, n_trials, Tuner:str):
    objective = ReservoirObjective(search_space, X_train, y_train, X_test, y_test, "mse")
    if Tuner == "Random":
        tuner = RandomTuner(objective_fn=objective, n_trials= n_trials)
    elif Tuner == "GPTuner":
        tuner = GPTuner(objective_fn=objective, n_trials= n_trials)
    elif Tuner == "TPETuner":
        tuner = TPETuner(objective_fn=objective, n_trials= n_trials)
    elif Tuner == "Grid":
        tuner = GridTuner(objective_fn=objective, n_trials= n_trials)
    else:
        raise (NotImplementedError)
    study = tuner.run()
    tuner.report()

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = sequence_to_scalar(
        name="sine_prediction",
        n_states=1,
        n_batch=200,
        n_time_in=20,
    )

    search_space = {
        "spectral_radius": ("float", 0.1, 1.5),
        "fraction_input": ("float", 0.01, 1.0),
        "reservoir_size": ("int", 100, 1000),
        "leak_rate": ("categorical", [0.1, 0.3, 0.5, 0.7]),
    }
    test_tuning(search_space= search_space, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_trials=10, Tuner="GPTuner")