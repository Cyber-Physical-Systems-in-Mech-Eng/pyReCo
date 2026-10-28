import copy
import json
import warnings

import joblib
import numpy as np
import optuna

from pyreco.metrics import assign_metric
from pyreco.utils_random import check_seed


class Tuner:
    r"""
    Hyperparameter tuner for pyReCo models using Optuna.

    The Tuner provides a lightweight yet flexible interface for automated
    hyperparameter optimization of reservoir computing models. It supports
    Optuna-based samplers (TPE, random, grid), custom evaluation loops, and
    immediate persistence of the best-performing model during optimization.

    The model that is passed to the tuner is never modified: every trial runs on a copy of it, and
    the best copy is available as `best_model`.

    The tuner is designed to work seamlessly with pyReCo models that expose
    a scikit-learn–like API (`fit`, `predict`, `evaluate`) and support dynamic
    hyperparameter updates via `model.set_hp()`.

    The search minimizes (or maximizes) the score :math:`s(\theta)` of the hyperparameters :math:`\theta`,
    :math:`\theta^* = \arg\min_\theta s(\theta)`, where :math:`s` is the validation metric of the model fitted
    with :math:`\theta`.

    Key features:

    * Optuna-based hyperparameter optimization (TPE, random, grid search)
    * Support for categorical, integer, and float search spaces
    * Optional log-scaled float parameters
    * Custom evaluator functions for control-loop or task-specific metrics
    * Automatic deep-copying and storage of the best model during tuning
    * Hyperparameter importance analysis after optimization
    * Compatible with cross-validation and closed-loop evaluation workflows

    Parameters
    ----------
    model : pyreco.CustomModel
        A compiled pyReCo model instance supporting `set_hp`, `fit`, and
        either `evaluate` or `predict`.

    search_space : dict
        Dictionary defining the hyperparameter search space. Each entry must follow the format

        .. code-block:: python

            {
                "param_name": ("float", low, high),
                "param_name": ("float", low, high, "log"),
                "param_name": ("int", low, high),
                "param_name": ("categorical", [v1, v2, v3]),
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
        Custom evaluation function with the signature ``evaluator(model, trial) -> float``.
        This allows task-specific scoring (e.g. closed-loop control RMSE).

    direction : str, default="minimize"
        Optimization direction. Must be either "minimize" or "maximize".

    verbose : bool, default=False
        If True, prints per-trial progress, best-model updates and the hyperparameter
        importances, and leaves the logging of Optuna as it is. If False, the tuner is quiet:
        Optuna only logs warnings while the optimization runs (the logging level of the caller
        is restored afterwards).

    run_id : int, default=0
        Optional identifier for experiment tracking.

    seed : int, optional
        Seed of the Optuna sampler, which draws the hyperparameters of the trials. With a seed,
        two tuning runs of the same (seeded) model and data propose the same hyperparameters
        and find the same best model. With seed=None the search is different on every run.
        The seed of the model itself is set separately, see `RC(seed=...)`.

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
        seed: int = None,
    ):
        check_seed(seed)
        self.seed = seed
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
        self._working_model = None  # the copy of `model` that the trials set up and train

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
        """Use another model; this resets the working copy, the best model, the study and the importances."""
        self.model = model
        self._working_model = None
        self.best_model = None
        self.study = None
        self.importances = None
        self.best_score = float("inf") if self.direction == "minimize" else -float("inf")

    def make_grid(self):
        """
        Grid of the search space for the grid sampler.

        Categorical parameters use their choices, integers all values in [low, high], floats
        ``num_float_steps`` equidistant values in [low, high].
        """
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
        """The Optuna sampler for ``sampler_type`` ("tpe", "random" or "grid"), seeded with ``seed``."""
        if self.sampler_type == "tpe":
            return optuna.samplers.TPESampler(seed=self.seed)
        elif self.sampler_type == "random":
            return optuna.samplers.RandomSampler(seed=self.seed)
        elif self.sampler_type == "grid":
            return optuna.samplers.GridSampler(self.make_grid(), seed=self.seed)
        else:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")

    def suggest_parameters(self, trial):
        """Ask the Optuna ``trial`` for one value per entry of the search space; returns them as a dict."""
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
        r"""
        One trial: set the suggested hyperparameters on the working copy, fit it and score it.

        The score is the value to minimize (or maximize) over the trials, :math:`s(\theta)`, for the
        hyperparameters :math:`\theta`: the output of ``evaluator`` if given, else ``metric`` or the first
        compiled metric of the model evaluated on the validation data. Invalid scores (NaN, infinite)
        never become the best. A better score stores a copy of the model as ``best_model``.

        Parameters
        ----------
        trial : optuna.Trial
            The trial.

        Returns
        -------
        float
            The score of the trial.
        """
        params = self.suggest_parameters(trial)

        # the trials run on a copy: the model of the caller stays as it is
        if self._working_model is None:
            self._working_model = copy.deepcopy(self.model)
        model = self._working_model

        model.set_hp(**params)
        model.fit(self.x_train, self.y_train)

        # -------------------------
        # Scoring logic (with default model.evaluate)
        # -------------------------
        if self.evaluator is not None:
            score = self.evaluator(model, trial)

        elif self.x_val is not None and self.y_val is not None:
            # If user provided a metric function, compute it from predictions
            if self.metric is not None:
                y_pred = model.predict(self.x_val)
                score = self.metric(self.y_val, y_pred)
            else:
                # DEFAULT: use model.evaluate directly
                score = model.evaluate(self.x_val, self.y_val)

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
            print(f"Trial {trial.number:02}: Params={params} -> Score={float(score):.6f}")

        # save best model immediately
        if self._is_better(float(score)):
            self.best_score = float(score)
            self.best_model = copy.deepcopy(model)
            if self.verbose:
                print(f"[Tuner] New best model saved at trial {trial.number:02} with score {float(score):.6f}")

        return float(score)

    def optimize(self, save_importance_path: str | None = None):
        r"""
        Search the hyperparameters, :math:`\theta^* = \arg\min_\theta s(\theta)` (or ``argmax``).

        Every optimization starts from a fresh copy of the model that was passed in.

        Parameters
        ----------
        save_importance_path : str, optional
            If given, the hyperparameter importances are written there as JSON.

        Returns
        -------
        optuna.Study
            The study; the best model is ``best_model``, importances are in ``importances`` (empty for a
            single trial).
        """
        # every optimization starts from a fresh copy of the model that was passed in
        self._working_model = copy.deepcopy(self.model)
        # quiet unless asked otherwise: restore the logging level of the caller afterwards
        previous_level = optuna.logging.get_verbosity()
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            self.study = optuna.create_study(
                direction=self.direction,
                sampler=self.get_sampler()
            )
            self.study.optimize(self.objective, n_trials=self.n_trials)
        finally:
            optuna.logging.set_verbosity(previous_level)

        # importances need at least two trials
        if len(self.study.trials) >= 2:
            from optuna.importance import get_param_importances

            # Optuna warns for small studies that the importances are not reliable; show
            # that only to a verbose tuner
            with warnings.catch_warnings():
                if not self.verbose:
                    warnings.simplefilter("ignore", UserWarning)
                self.importances = get_param_importances(self.study)
        else:
            self.importances = {}

        if self.verbose:
            print("\n[Optuna] Hyperparameter importances:")
            for k, v in self.importances.items():
                print(f"  {k:>20s}: {v:.6f}")

        if save_importance_path is not None:
            with open(save_importance_path, "w", encoding="utf-8") as f:
                json.dump(self.importances, f, indent=2)

        return self.study

    def save_best_model(self, path):
        """Save ``best_model`` to ``path`` with joblib; warns if there is none (``optimize`` was not run)."""
        if self.best_model is not None:
            joblib.dump(self.best_model, path)
            if self.verbose:
                print(f"[Tuner] Best model saved to: {path}")
        else:
            warnings.warn("No best model to save. Did you run optimize()?", UserWarning)

    def report(self):
        """Print the best parameters, the best score and the importances of the last study."""
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