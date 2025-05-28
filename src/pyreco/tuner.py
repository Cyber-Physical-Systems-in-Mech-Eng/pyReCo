import optuna # also torch is needed for the gp sampler
from abc import ABC, abstractmethod
from pyreco.models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.metrics import assign_metric
import numpy as np

class BaseTuner:
    def __init__(self, objective_fn, n_trials=50): # Base class for other tuners
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.study = None

    def get_sampler(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def create_study(self, direction="minimize"):
        self.study = optuna.create_study(direction=direction, sampler=self.get_sampler())

    def run(self):
        self.create_study()
        self.study.optimize(self.objective_fn, n_trials=self.n_trials)
        return self.study

    def report(self):
        print("Best Params:", self.study.best_trial.params)
        print("Best Score:", self.study.best_trial.value)

class TPETuner(BaseTuner):   # This is bayesian optimization, but with another type of surrogate model
    def get_sampler(self):
        return optuna.samplers.TPESampler()

class GridTuner(BaseTuner): # Gridsearch
    @staticmethod
    def make_grid(search_space, num_float_steps=5):
        grid = {}
        for k, v in search_space.items():
            if v[0] == "categorical":
                grid[k] = v[1]
            elif v[0] == "int":
                grid[k] = list(range(v[1], v[2]+1))
            elif v[0] == "float":
                grid[k] = list(np.linspace(v[1], v[2], num=num_float_steps))
        return grid

    def get_sampler(self):
        grid = self.make_grid(self.objective_fn.search_space)
        return optuna.samplers.GridSampler(grid)

class RandomTuner (BaseTuner):
    def get_sampler(self):
        return optuna.samplers.RandomSampler()

class GPTuner (BaseTuner):    # This is the classical Bayesian optimization
    def get_sampler(self):
        return optuna.samplers.GPSampler()

class BaseObjective(ABC):
    def __init__(self, search_space, metric):
        self.search_space = search_space
        self.evaluator = assign_metric(metric)

    def suggest_parameters(self, trial):
        params = {}
        for name, (ptype, *args) in self.search_space.items():
            if ptype == "float":
                params[name] = trial.suggest_float(name, *args)
            elif ptype == "int":
                params[name] = trial.suggest_int(name, *args)
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, args[0])
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")
        return params

    @abstractmethod
    def __call__(self, trial):
        pass

class ReservoirObjective(BaseObjective):
    def __init__(self, search_space, x_train, y_train, x_val, y_val, metric):
        super().__init__(search_space,metric)
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def build_model(self, input_shape, output_shape, params):
        model = RC()
        model.add(InputLayer(input_shape))
        model.add(RandomReservoirLayer(
            nodes=params["reservoir_size"],
            activation="tanh",
            fraction_input=params["fraction_input"],
            spec_rad=params["spectral_radius"],
            leakage_rate=params["leak_rate"]
        ))
        model.add(ReadoutLayer(output_shape))
        return model

    def __call__(self, trial):
        params = self.suggest_parameters(trial)
        model = self.build_model(self.x_train.shape[1:], self.y_train.shape[1:], params)
        model.compile(optimizer='ridge', metrics=['mean_squared_error'])
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_val)
        return self.evaluator(self.y_val, y_pred)



