"""
Minimal working example: how small can a reservoir computer for NARMA10 be?

The script answers the question in two ways:

1. Size sweep. For several numbers of reservoir nodes, the hyperparameters (spectral radius,
   leakage rate, input scaling, ridge penalty) are tuned on a validation set for a few random
   reservoirs. The smallest size whose validation error is within TOLERANCE of the best size is
   "small but still accurate".
2. Edge pruning. The edges of the tuned reservoir of that size are removed one after the other.
   The sparsest network whose validation error is within TOLERANCE of the unpruned one is
   "sparse but still accurate".

The result is put into perspective by three references: predicting the mean (NRMSE 1), a linear
model on the last 20 inputs and a quadratic model on the last 20 inputs.

The script is also an extended smoke test. Every stage is run in a guarded block that records
exceptions, warnings and the amount of text printed by the library, and a number of properties that
should hold (reproducibility, consistency of the pruner's bookkeeping, ...) are checked and
reported instead of stopping the script. The report at the end lists everything unexpected.
"""

import contextlib
import copy
import io
import itertools
import time
import warnings
from collections import Counter

import numpy as np
import optuna
from sklearn.linear_model import Ridge

from pyreco.custom_models import RC
from pyreco.datasets import load
from pyreco.edge_pruning import EdgePruner
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.metrics import nrmse
from pyreco.tuner import Tuner

# ------------------------------------------------------------------------------ settings
SIZES = [5, 10, 20, 40, 60]  # numbers of reservoir nodes to try
SEEDS = [0, 1]  # random reservoirs per size
N_TRIALS = 15  # hyperparameter trials per reservoir
TOLERANCE = 0.10  # "still accurate": validation NRMSE within 10% of the best
DENSITY = 0.2
N_PRUNE = 800  # training windows used for the pruning (every candidate edge is retrained)
N_IN = 30  # input window; the target belongs to the last input step (warm-up N_IN - 1)

SEARCH_SPACE = {
    "spec_rad": ("float", 0.3, 1.0),
    "leakage_rate": ("float", 0.1, 1.0),
    "input_scaling": ("float", 0.1, 3.0),
    "alpha": ("float", 1e-9, 1.0, "log"),  # ridge penalty of the readout
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------------- smoke test tools
class SmokeTest:
    """Collects exceptions, warnings, printed text and failed checks of the stages."""

    def __init__(self):
        self.findings = []  # (stage, kind, message)
        self.stage_name = ""
        self.chatter = {}

    @contextlib.contextmanager
    def stage(self, name):
        self.stage_name = name
        start, printed = time.time(), io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(printed):
            warnings.simplefilter("always")
            try:
                yield
            except Exception as error:  # record and go on with the next stage
                self.findings.append((name, "ERROR", f"{type(error).__name__}: {error}"))
        for w in caught:
            self.findings.append((name, "WARNING", f"{w.category.__name__}: {w.message}"))
        self.chatter[name] = (printed.getvalue().count("\n"), time.time() - start)

    def check(self, condition, message):
        if not condition:
            self.findings.append((self.stage_name, "UNEXPECTED", message))
        return condition

    def report(self):
        print("\n" + "=" * 90 + "\nSMOKE TEST REPORT\n" + "=" * 90)
        print("stage                              seconds   lines printed by the library")
        for name, (lines, seconds) in self.chatter.items():
            print(f"  {name:32s} {seconds:7.1f}   {lines}")
        counts = Counter(self.findings)
        if not counts:
            print("\nno errors, warnings or violated checks")
        for (stage, kind, message), n in sorted(counts.items(), key=lambda item: item[0][1]):
            print(f"\n[{kind}] {stage}" + (f" (x{n})" if n > 1 else "") + f"\n    {message}")


smoke = SmokeTest()


def build(nodes, seed):
    model = RC(seed=seed)
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=nodes, density=DENSITY, seed=seed))
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    model.compile(discard_transients=N_IN - 1, metrics=["nrmse"])
    return model


def tune(nodes, seed, n_trials=N_TRIALS):
    model = build(nodes, seed)
    tuner = Tuner(model, SEARCH_SPACE, x_train, y_train, x_val=x_val, y_val=y_val, n_trials=n_trials)
    tuner.optimize()
    return tuner.best_model, model


# ----------------------------------------------------------------------------------- data
with smoke.stage("data"):
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = load(
        "narma10", n_samples=4000, n_in=N_IN, n_out=1, seed=1, val_fraction=0.2
    )
    print(f"windows: {len(x_train)} train, {len(x_val)} validation, {len(x_test)} test")
    smoke.check(np.isfinite(x_train).all() and np.isfinite(y_train).all(), "data contains NaN/inf")

# ------------------------------------------------------------------------------ references
with smoke.stage("references"):
    def lags(x, quadratic):
        last = x[:, -20:, 0]
        if not quadratic:
            return last
        products = [last[:, i] * last[:, j] for i, j in itertools.combinations_with_replacement(range(20), 2)]
        return np.concatenate([last, np.stack(products, axis=1)], axis=1)

    references = {"predict the mean": 1.0}
    for name, quadratic in [("linear model, 20 lags", False), ("quadratic model, 20 lags", True)]:
        ridge = Ridge(alpha=1e-6).fit(lags(x_train, quadratic), y_train[:, 0, 0])
        prediction = ridge.predict(lags(x_test, quadratic))[:, None, None]
        references[name] = float(nrmse(y_test, prediction))

# ---------------------------------------------------------------------------------- sweep
results = {}  # (nodes, seed) -> dict
with smoke.stage("size sweep"):
    for nodes, seed in itertools.product(SIZES, SEEDS):
        start = time.time()
        best, passed_in = tune(nodes, seed)
        default = build(nodes, seed)
        default.fit(x_train, y_train)
        results[nodes, seed] = dict(
            model=best,
            val=best.evaluate(x_val, y_val)[0],
            test=best.evaluate(x_test, y_test)[0],
            default_val=default.evaluate(x_val, y_val)[0],
            seconds=time.time() - start,
        )
        smoke.check(
            results[nodes, seed]["val"] <= results[nodes, seed]["default_val"] + 1e-9,
            "the tuned model is worse on validation than the untuned one",
        )
        smoke.check(
            passed_in.get_hp() == build(nodes, seed).get_hp(),
            "Tuner changed the hyperparameters of the model that was passed in",
        )
        layer = results[nodes, seed]["model"].reservoir_layer
        smoke.check(
            np.isclose(layer.spec_rad, np.max(np.abs(np.linalg.eigvals(layer.weights)))),
            "spectral radius stored in the tuned model differs from the one of its weights",
        )

# ---------------------------------------------------------------------- choose the size
smallest = None
with smoke.stage("choose size"):
    median_val = {n: float(np.median([results[n, s]["val"] for s in SEEDS])) for n in SIZES}
    median_test = {n: float(np.median([results[n, s]["test"] for s in SEEDS])) for n in SIZES}
    smoke.check(all(np.isfinite(v) and 0 < v < 1.5 for v in median_val.values()), "NRMSE outside (0, 1.5)")
    smallest = next(n for n in SIZES if median_val[n] <= (1 + TOLERANCE) * min(median_val.values()))
    smoke.check(
        all(b <= a * 1.05 for a, b in zip(median_val.values(), list(median_val.values())[1:])),
        "the validation error increases by more than 5% when the reservoir gets larger",
    )
    smoke.check(
        min(median_test.values()) <= references["linear model, 20 lags"],
        "even the best reservoir is worse than a linear model on the last 20 inputs",
    )

# ------------------------------------------------------------------------------ pruning
pruned_summary = None
with smoke.stage("edge pruning"):
    seed = min(SEEDS, key=lambda s: results[smallest, s]["val"])
    tuned = copy.deepcopy(results[smallest, seed]["model"])
    n_edges = int(np.count_nonzero(tuned.reservoir_layer.weights))

    pruner = EdgePruner(
        performance_criterion="nrmse", candidate_fraction=0.1, stopping_criterion=["min_edges"],
        min_num_edges=3, metrics=["nrmse"], seed=0,
    )
    x_prune, y_prune = x_train[:N_PRUNE], y_train[:N_PRUNE]
    tuned.fit(x_prune, y_prune)  # the pruner starts from a model trained on the pruning data
    reference_val = tuned.evaluate(x_val, y_val)[0]
    _, history = pruner.prune(copy.deepcopy(tuned), (x_prune, y_prune), (x_val, y_val))
    steps = [history[k] for k in sorted(history)]
    edges = [steps[0]["starting_model"]["num_edges"]] + [s["final_model"]["num_edges"] for s in steps]
    losses = [steps[0]["starting_model"]["loss"]] + [s["final_model"]["loss"] for s in steps]

    smoke.check(edges[0] == n_edges, f"history starts with {edges[0]} edges, the model has {n_edges}")
    smoke.check(all(a - b == 1 for a, b in zip(edges, edges[1:])), "an iteration did not remove exactly one edge")
    smoke.check(np.isclose(losses[0], reference_val), "loss at the start of pruning differs from evaluate()")

    # the sparsest network within the tolerance: rebuild it from the stored weights and refit
    ok = [i for i, loss in enumerate(losses) if loss <= (1 + TOLERANCE) * reference_val]
    sparsest = max(ok)
    sparse_model = copy.deepcopy(tuned)
    sparse_model.reservoir_layer.set_weights(history[sorted(history)[sparsest - 1]]["final_model"]["weights"].toarray()
                                             if sparsest > 0 else tuned.reservoir_layer.weights)
    sparse_model.fit(x_prune, y_prune)
    refit_val = sparse_model.evaluate(x_val, y_val)[0]
    smoke.check(
        np.isclose(refit_val, losses[sparsest], rtol=1e-6),
        f"validation NRMSE after rebuilding the pruned network ({refit_val:.4f}) differs from the "
        f"loss recorded by the pruner ({losses[sparsest]:.4f})",
    )
    layer = sparse_model.reservoir_layer
    smoke.check(int(np.count_nonzero(layer.weights)) == edges[sparsest], "edge count of the rebuilt network")
    smoke.check(
        np.isclose(layer.spec_rad, np.max(np.abs(np.linalg.eigvals(layer.weights)))),
        "spectral radius stored in the layer is stale after set_weights()",
    )
    pruned_summary = dict(
        edges_before=n_edges, edges_after=edges[sparsest], val=refit_val,
        test=sparse_model.evaluate(x_test, y_test)[0],
    )

# ----------------------------------------------------------------------- reproducibility
with smoke.stage("reproducibility"):
    first, second = build(SIZES[0], 0), build(SIZES[0], 0)
    first.fit(x_train, y_train)
    second.fit(x_train, y_train)
    smoke.check(
        first.evaluate(x_val, y_val)[0] == second.evaluate(x_val, y_val)[0],
        "two seeded models give different results",
    )
    a, _ = tune(SIZES[0], 0, n_trials=6)
    b, _ = tune(SIZES[0], 0, n_trials=6)
    smoke.check(
        a.get_hp() == b.get_hp(),
        "two tuning runs with identical seeds find different hyperparameters (the Tuner has no seed)",
    )

# -------------------------------------------------------------------------------- results
print(f"\nNARMA10, windows of {N_IN} inputs; NRMSE (1 = predicting the mean)")
print("\nreferences (test):", ", ".join(f"{k} {v:.3f}" for k, v in references.items()))
if smallest is not None:
    print(f"\n{'nodes':>6} {'val (median)':>13} {'test (median)':>14} {'seconds':>9}")
    for n in SIZES:
        seconds = sum(results[n, s]["seconds"] for s in SEEDS)
        mark = "  <- smallest accurate size" if n == smallest else ""
        print(f"{n:6d} {median_val[n]:13.3f} {median_test[n]:14.3f} {seconds:9.1f}{mark}")
    print(
        f"\nsmallest accurate reservoir: {smallest} nodes (validation NRMSE within "
        f"{TOLERANCE:.0%} of the best size)"
    )
if pruned_summary:
    print(
        f"edge pruning of that reservoir: {pruned_summary['edges_before']} -> "
        f"{pruned_summary['edges_after']} edges, validation NRMSE {pruned_summary['val']:.3f}, "
        f"test NRMSE {pruned_summary['test']:.3f}"
    )

smoke.report()
