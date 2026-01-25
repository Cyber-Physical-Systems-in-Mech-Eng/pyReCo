import copy
import json

import joblib
import numpy as np
import optuna

from pyreco.metrics import assign_metric


class Tuner:
    """
    Hyperparameter tuner for pyReCo models using Optuna.

    The Tuner provides a lightweight yet flexible interface for automated
    hyperparameter optimization of reservoir computing models. It supports
    Optuna-based samplers (TPE, random, grid), custom evaluation loops, and
    immediate persistence of the best-performing model during optimization.

    The tuner is designed to work seamlessly with pyReCo models that expose
    a scikit-learn–like API (`fit`, `predict`, `evaluate`) and support dynamic
    hyperparameter updates via `model.set_hp()`.

    Key Features
    ------------
    - Optuna-based hyperparameter optimization (TPE, random, grid search)
    - Support for categorical, integer, and float search spaces
    - Optional log-scaled float parameters
    - Custom evaluator functions for control-loop or task-specific metrics
    - Automatic deep-copying and storage of the best model during tuning
    - Hyperparameter importance analysis after optimization
    - Compatible with cross-validation and closed-loop evaluation workflows

    Parameters
    ----------
    model : pyreco.CustomModel
        A compiled pyReCo model instance supporting `set_hp`, `fit`, and
        either `evaluate` or `predict`.

    search_space : dict
        Dictionary defining the hyperparameter search space.
        Each entry must follow the format:
            {
                "param_name": ("float", low, high),
                "param_name": ("float", low, high, "log"),
                "param_name": ("int", low, high),
                "param_name": ("categorical", [v1, v2, v3])
            }

    x_train : np.ndarray
        Training input data.

    y_train : np.ndarray
        Training target data.

    x_val : np.ndarray, optional
        Validation input data. Required if no custom evaluator is provided.

    y_val : np.ndarray, optional
        Validation target data. Required if no custom evaluator is provided.

    metric : str, optional
        Name of a pyReCo metric (e.g. "mse", "rmse"). If provided, predictions
        on validation data are scored using this metric.

    n_trials : int, default=50
        Number of Optuna trials to run.

    sampler_type : str, default="tpe"
        Optuna sampler to use. Supported values:
        {"tpe", "random", "grid"}.

    num_float_steps : int, default=5
        Number of discretization steps for float parameters when using
        grid search.

    evaluator : callable, optional
        Custom evaluation function with signature:
            evaluator(model, trial) -> float
        This allows task-specific scoring (e.g. closed-loop control RMSE).

    direction : str, default="minimize"
        Optimization direction. Must be either "minimize" or "maximize".

    verbose : bool, default=False
        If True, prints per-trial progress and best-model updates.

    run_id : int, default=0
        Optional identifier for experiment tracking.

    Attributes
    ----------
    study : optuna.Study
        Optuna study object created during optimization.

    best_model : pyreco.CustomModel
        Deep copy of the best-performing model found so far.

    best_score : float
        Best objective value encountered during optimization.

    importances : dict
        Hyperparameter importances computed after optimization.
    """
    def __init__(
        self,
        model,
        search_space,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        metric=None,
        n_trials=50,
        sampler_type="tpe",
        num_float_steps=5,
        evaluator=None,
        direction="minimize",
        verbose=False,
        run_id: int = 0,
    ):
        self.model = model
        self.search_space = search_space
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.metric_name = metric
        self.metric = assign_metric(metric) if metric else None

        self.n_trials = n_trials
        self.sampler_type = sampler_type
        self.num_float_steps = num_float_steps
        self.evaluator = evaluator
        self.direction = direction
        self.verbose = verbose
        self.run_id = run_id

        self.study = None
        self.best_model = None

        if self.direction == "minimize":
            self.best_score = float("inf")
        elif self.direction == "maximize":
            self.best_score = -float("inf")
        else:
            raise ValueError("direction must be 'minimize' or 'maximize'")

        # store HP importances after optimize()
        self.importances = None

    def _is_better(self, score: float) -> bool:
        return score < self.best_score if self.direction == "minimize" else score > self.best_score

    def set_model(self, model):
        self.model = model
        self.best_model = None
        self.study = None
        self.importances = None
        self.best_score = float("inf") if self.direction == "minimize" else -float("inf")

    def make_grid(self):
        grid = {}
        for k, v in self.search_space.items():
            ptype = v[0]
            if ptype == "categorical":
                grid[k] = v[1]
            elif ptype == "int":
                grid[k] = list(range(v[1], v[2] + 1))
            elif ptype == "float":
                grid[k] = list(np.linspace(v[1], v[2], num=self.num_float_steps))
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")
        return grid

    def get_sampler(self):
        if self.sampler_type == "tpe":
            return optuna.samplers.TPESampler()
        elif self.sampler_type == "random":
            return optuna.samplers.RandomSampler()
        elif self.sampler_type == "grid":
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

        # -------------------------
        # Scoring logic (with default model.evaluate)
        # -------------------------
        if self.evaluator is not None:
            score = self.evaluator(self.model, trial)

        elif self.x_val is not None and self.y_val is not None:
            # If user provided a metric function, compute it from predictions
            if self.metric is not None:
                y_pred = self.model.predict(self.x_val)
                score = self.metric(self.y_val, y_pred)
            else:
                # DEFAULT: use model.evaluate directly
                score = self.model.evaluate(self.x_val, self.y_val)

                # normalize possible return types
                if isinstance(score, (list, tuple)):
                    score = score[0]
                elif isinstance(score, dict):
                    score = next(iter(score.values()))

        else:
            raise ValueError("Tuner requires either an evaluator or (x_val, y_val).")

        # handle invalid scores safely
        if score is None or (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
            score = float("inf") if self.direction == "minimize" else -float("inf")

        if self.verbose:
            print(f"Trial {trial.number:02}: Params={params} → Score={float(score):.6f}")

        # save best model immediately
        if self._is_better(float(score)):
            self.best_score = float(score)
            self.best_model = copy.deepcopy(self.model)
            if self.verbose:
                print(f"[Tuner] New best model saved at trial {trial.number:02} with score {float(score):.6f}")

        return float(score)

    def optimize(self, save_importance_path: str | None = None):
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=self.get_sampler()
        )
        self.study.optimize(self.objective, n_trials=self.n_trials)

        from optuna.importance import get_param_importances
        self.importances = get_param_importances(self.study)

        print("\n[Optuna] Hyperparameter importances:")
        for k, v in self.importances.items():
            print(f"  {k:>20s}: {v:.6f}")

        if save_importance_path is not None:
            with open(save_importance_path, "w", encoding="utf-8") as f:
                json.dump(self.importances, f, indent=2)

        return self.study

    def save_best_model(self, path):
        if self.best_model is not None:
            joblib.dump(self.best_model, path)
            print(f"[Tuner] Best model saved to: {path}")
        else:
            print("[Tuner] No best model to save. Did you run optimize()?")

    def report(self):
        if self.study is None:
            print("[Tuner] No study yet. Run optimize().")
            return

        print("\n=== TUNER REPORT ===")
        print("Best Params:")
        for k, v in self.study.best_trial.params.items():
            print(f"  {k}: {v}")

        print(f"\nBest Score: {self.study.best_trial.value:.6f}")

        if self.importances is not None:
            print("\nHyperparameter Importances:")
            for k, v in self.importances.items():
                print(f"  {k:>20s}: {v:.6f}")
        else:
            print("\nHyperparameter Importances: not computed (run optimize() first)")