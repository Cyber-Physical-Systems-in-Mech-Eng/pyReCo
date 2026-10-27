"""
Minimal working example: the best reservoir computer for NARMA10 with at most 200 nodes.

A reservoir is described by its size, its connectivity and a handful of hyperparameters. `Tuner`
searches the hyperparameters of a reservoir that is given (`set_hp`); here the size and the density
of the network are searched as well, so the study is written with optuna directly: every trial
builds, trains and validates one reservoir.

NARMA10 has a random input, so a new seed is a new sequence. The training, validation and test
windows come from three independent sequences and nothing leaks between them.

Result: a near-linear regime of the units (small input scaling) and a sparse network are best.
NARMA10 contains products of inputs, which a reservoir without a bias cannot represent well (see
the references: a quadratic model on the last 20 inputs is much better than a linear one), so the
reservoir ends up at about the level of the linear model.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.linear_model import Ridge

from pyreco.custom_models import RC
from pyreco.datasets import load
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.metrics import nrmse

MAX_NODES = 200
N_TRIALS = 80
N_IN = 40  # input window; the target belongs to the last input step (warm-up N_IN - 1)
TOLERANCE = 0.05  # "small and accurate": validation NRMSE within 5% of the best

# search ranges of the float hyperparameters: (low, high, logarithmic)
SPACE = {
    "density": (0.02, 0.5, True),  # probability of an edge between two nodes
    "spec_rad": (0.1, 1.3, False),
    "leakage_rate": (0.05, 1.0, False),
    "input_scaling": (1e-3, 10.0, True),
    "alpha": (1e-12, 1.0, True),  # ridge penalty of the readout
    "fraction_input": (0.2, 1.0, False),  # fraction of the nodes that receive the input
}

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------------------- data
x_train, y_train, _, _ = load("narma10", n_samples=3000, n_in=N_IN, n_out=1, seed=1)
_, _, x_val, y_val = load("narma10", n_samples=3000, n_in=N_IN, n_out=1, seed=2)
_, _, x_test, y_test = load("narma10", n_samples=3000, n_in=N_IN, n_out=1, seed=3)


def lags(x, quadratic):
    last = x[:, -20:, 0]
    if not quadratic:
        return last
    products = [last[:, i] * last[:, j] for i, j in itertools.combinations_with_replacement(range(20), 2)]
    return np.concatenate([last, np.stack(products, axis=1)], axis=1)


references = {"predict the mean": 1.0}
for name, quadratic in [("linear model, 20 lags", False), ("quadratic model, 20 lags", True)]:
    ridge = Ridge(alpha=1e-6).fit(lags(x_train, quadratic), y_train[:, 0, 0])
    references[name] = float(nrmse(y_test, ridge.predict(lags(x_test, quadratic))[:, None, None]))


# ----------------------------------------------------------------------------- search
def build(params, seed=0):
    model = RC(seed=seed)
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(
        RandomReservoirLayer(
            nodes=params["nodes"],
            density=params["density"],
            activation="tanh",  # sigmoid units were worse in all searches that we tried
            spec_rad=params["spec_rad"],
            leakage_rate=params["leakage_rate"],
            fraction_input=params["fraction_input"],
            seed=seed,
        )
    )
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    model.compile(discard_transients=N_IN - 1, input_scaling=params["input_scaling"], metrics=["nrmse"])
    model.set_hp(alpha=params["alpha"])  # ridge penalty of the readout
    return model


def objective(trial):
    params = {"nodes": trial.suggest_int("nodes", 20, MAX_NODES)}
    for name, (low, high, log) in SPACE.items():
        params[name] = trial.suggest_float(name, low, high, log=log)
    try:
        model = build(params)
    except ValueError as error:  # e.g. a sparse small network without a directed cycle
        trial.set_user_attr("error", str(error))
        return 10.0  # worse than anything that works (NRMSE 1 = predicting the mean)
    model.fit(x_train, y_train)
    return model.evaluate(x_val, y_val)[0]


def progress(study, trial):
    if (trial.number + 1) % 10 == 0:
        print(f"  trial {trial.number + 1:3d}/{N_TRIALS}: best validation NRMSE so far {study.best_value:.3f}")


print(f"searching {N_TRIALS} reservoirs with up to {MAX_NODES} nodes ...")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress])

# ------------------------------------------------------------------------- results
best = study.best_trial
valid = [t for t in study.trials if t.value < 10.0]
small = min(
    (t for t in valid if t.value <= (1 + TOLERANCE) * best.value),
    key=lambda t: (t.params["nodes"], t.value),
)


def describe(label, trial):
    model = build(trial.params)
    model.fit(x_train, y_train)
    edges = int(np.count_nonzero(model.reservoir_layer.weights))
    test = model.evaluate(x_test, y_test)[0]
    print(f"{label:>34s} {trial.params['nodes']:6d} {edges:7d} {trial.value:8.3f} {test:8.3f}")
    return model, test


print(f"\nNARMA10, windows of {N_IN} inputs; NRMSE (1 = predicting the mean)")
print("references (test):", ", ".join(f"{k} {v:.3f}" for k, v in references.items()))
print(f"\n{'':>34s} {'nodes':>6s} {'edges':>7s} {'val':>8s} {'test':>8s}")
best_model, best_test = describe("best reservoir", best)
describe(f"smallest within {TOLERANCE:.0%} of the best", small)

print("\nbest hyperparameters:")
for key, value in best.params.items():
    print(f"  {key:>15s}: {value}")
print("\nwhat mattered (optuna importances):")
for key, value in list(optuna.importance.get_param_importances(study).items())[:5]:
    print(f"  {key:>15s}: {value:.2f}")

# an optimum at the edge of a search range means that the range may be too narrow
for name, (low, high, log) in SPACE.items():
    value = best.params[name]
    position = np.log(value / low) / np.log(high / low) if log else (value - low) / (high - low)
    if position < 0.02 or position > 0.98:
        edge = "lower" if position < 0.5 else "upper"
        print(f"note: {name} = {value:.3g} is at the {edge} edge of its range [{low}, {high}]")

# ---------------------------------------------------------------------------- plots
fig, axes = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
history = np.minimum.accumulate([t.value for t in study.trials])
axes[0].plot(np.arange(1, len(history) + 1), history, marker=".")
axes[0].axhline(references["linear model, 20 lags"], color="gray", linestyle=":", label="linear model on 20 lags")
axes[0].set_xlabel("trial")
axes[0].set_ylabel("best validation NRMSE")
axes[0].legend()
prediction = best_model.predict(x_test)
axes[1].plot(y_test[:150, 0, 0], label="target", color="#1D3557")
axes[1].plot(prediction[:150, 0, 0], label="prediction", color="#E63946", linestyle="--")
axes[1].set_title(f"best reservoir ({best.params['nodes']} nodes): test NRMSE {best_test:.3f}")
axes[1].set_xlabel("window")
axes[1].legend()
plt.show()
