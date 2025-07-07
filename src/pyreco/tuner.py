import optuna
import numpy as np

class Tuner:
    def __init__(
        self,
        model,
        search_space,
        x_train, y_train,
        x_val=None, y_val=None,
        metric=None,
        n_trials=50,
        sampler_type='tpe',        # 'tpe', 'random', 'grid'
        num_float_steps=5,
        evaluator=None,            # optional: custom(model, trial) → score
        direction='minimize',
        verbose=False
    ):
        self.model = model
        self.search_space = search_space
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.metric = metric
        self.n_trials = n_trials
        self.sampler_type = sampler_type
        self.num_float_steps = num_float_steps
        self.evaluator = evaluator
        self.direction = direction
        self.verbose = verbose
        self.study = None

    def make_grid(self):
        grid = {}
        for k, v in self.search_space.items():
            if v[0] == "categorical":
                grid[k] = v[1]
            elif v[0] == "int":
                grid[k] = list(range(v[1], v[2] + 1))
            elif v[0] == "float":
                grid[k] = list(np.linspace(v[1], v[2], num=self.num_float_steps))
            else:
                raise ValueError(f"Unsupported parameter type: {v[0]}")
        return grid

    def get_sampler(self):
        if self.sampler_type == 'tpe':
            return optuna.samplers.TPESampler()
        elif self.sampler_type == 'random':
            return optuna.samplers.RandomSampler()
        elif self.sampler_type == 'grid':
            return optuna.samplers.GridSampler(self.make_grid())
        else:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")

    def suggest_parameters(self, trial):
        params = {}
        for name, (ptype, *args) in self.search_space.items():
            if ptype == "float":
                if len(args) == 3 and args[2] == "log":
                    params[name] = trial.suggest_float(name, args[0], args[1], log=True)
                else:
                    params[name] = trial.suggest_float(name, *args)
            elif ptype == "int":
                params[name] = trial.suggest_int(name, *args)
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, args[0])
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")
        return params

    def objective(self, trial):
        params = self.suggest_parameters(trial)
        self.model.set_hp(**params)
        self.model.fit(self.x_train, self.y_train)

        # Use custom evaluator if given
        if self.evaluator is not None:
            score = self.evaluator(self.model, trial)

        # Otherwise, use validation metric with model.evaluate()
        elif self.x_val is not None and self.y_val is not None and self.metric is not None:
            if isinstance(self.metric, str):
                metric_names = [self.metric]
            elif isinstance(self.metric, list):
                metric_names = self.metric
            else:
                raise ValueError("Metric must be a string or list of strings.")

            results = self.model.evaluate(self.x_val, self.y_val, metrics=metric_names)
            score = results[0]  # use the first metric only

        else:
            raise ValueError("No evaluator or validation+metric provided.")

        if self.verbose:
            print(f"[Trial {trial.number}] Params={params} → Score={score:.5f}")
        return score

    def optimize(self):
        self.study = optuna.create_study(direction=self.direction, sampler=self.get_sampler())
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return self.study

    def report(self):
        print("Best Hyperparameters:", self.study.best_trial.params)
        print("Best Score:", self.study.best_trial.value)
