"""
Minimal working example: reconstruct the hidden delayed state of the Mackey-Glass delay equation.

Mackey-Glass is a delay differential equation,

    dx/dt = a x(t - tau) / (1 + x(t - tau)^n) - b x(t),

so its state at time t is the whole history over [t - tau, t]. Only x(t) is measured. The
unmeasured quantities are the delayed state x(t - tau) and the production term
p(t) = a x(t - tau) / (1 + x(t - tau)^n) that drives the equation (p = dx/dt + b x). A reservoir
computer ("reservoir observer", see mwe_lorenz_reconstruction.py) is trained to infer them from x(t)
at the same time step, which needs a memory of tau time units: the reservoir has to hold the history
that the equation carries in its delay.

A model without memory, a ridge regression on x(t) alone, cannot do it: x(t) does not determine x(t - tau)
(the series passes every value twice, rising and falling). At lag 17 the reservoir finds the hidden state
with an NRMSE of 1e-4 on the test data.

The reconstruction error as a function of the lag shows the memory of the reservoir: past lags up to
a few tau are reconstructed well, and the "lags" in the future (negative) are predictions. Mackey-Glass with
tau = 17 is only weakly chaotic (Lyapunov exponent about 0.006 per time unit), so even far-ahead
predictions work for a while before the error grows. A target without any relation to the input
(the same values in random order) stays at NRMSE 1 and is the control for leaks.

Data layout: one long sequence per data set, shape (1, n_time, 1); the targets leave out the first
`WASHOUT` steps (`compile(discard_transients=WASHOUT)`). For a lag L the target at time step t is
x(t - L), so the inputs of a negative lag (the future) end earlier. The hyperparameters were found by a
50 trial optuna search on the validation data for lag tau with 200 nodes.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge

from pyreco.custom_models import RC
from pyreco.datasets import _generate_mackey_glass
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.metrics import nrmse

# the almost unregularized readout (alpha ~ 1e-12) of a reservoir with nearly collinear states is badly
# conditioned on purpose; the solution is still the one that is wanted
warnings.filterwarnings("ignore", category=LinAlgWarning)

TAU, A, B, N_HILL = 17, 0.2, 0.1, 10  # defaults of the generator; sampled once per time unit
WASHOUT = 200
N_BURN_IN, N_TRAIN, N_VAL, N_TEST = 1000, 5000, 3000, 3000

HP = dict(spec_rad=1.017, leakage_rate=0.773, input_scaling=0.0749, input_bias=0.159, alpha=3.4e-12)
RESERVOIR = dict(density=0.284, weight_distribution="uniform")

# ------------------------------------------------------------------------------- data
series = _generate_mackey_glass(N_BURN_IN + N_TRAIN + N_VAL + N_TEST)[N_BURN_IN:, 0]
train = series[:N_TRAIN]
val = series[N_TRAIN : N_TRAIN + N_VAL]
test = series[N_TRAIN + N_VAL :]


def production(x_delayed):
    """The term that drives the equation: p = dx/dt + b x."""
    return A * x_delayed / (1 + x_delayed**N_HILL)


def pair(data, lag, transform=None):
    """Input x(t) in full length and the target f(x(t - lag)) from time step WASHOUT on.

    A positive lag is the past (memory), a negative lag is the future (prediction).
    Returns arrays of shape (1, n_time, 1).
    """
    if lag > WASHOUT:
        raise ValueError(f"lag {lag} is longer than the warm-up {WASHOUT}: the target would be missing")
    end = len(data) if lag >= 0 else len(data) + lag  # no target beyond the end of the series
    x = data[None, :end, None]
    y = data[None, WASHOUT - lag : end - lag, None]
    return x, (transform(y) if transform else y)


def build(lag, nodes=200, seed=0, transform=None, shuffle_targets=False, **overrides):
    x, y = pair(train, lag, transform)
    if shuffle_targets:  # control: targets without any relation to the inputs
        y = np.random.default_rng(seed).permutation(y[0])[None]
    reservoir = {**RESERVOIR, **overrides.pop("reservoir", {})}
    hp = {**HP, **overrides}
    model = RC(seed=seed)
    model.add(InputLayer(input_shape=x.shape[1:]))
    model.add(
        RandomReservoirLayer(
            nodes=nodes, spec_rad=hp["spec_rad"], leakage_rate=hp["leakage_rate"],
            fraction_input=1.0, seed=seed, **reservoir,
        )
    )
    model.add(ReadoutLayer(output_shape=y.shape[1:]))
    model.compile(
        discard_transients=WASHOUT,
        input_scaling=hp["input_scaling"],
        input_bias=hp["input_bias"],
        metrics=["nrmse"],
    )
    model.set_hp(alpha=hp["alpha"])
    model.fit(x, y)
    return model


def score(model, data, lag, transform=None):
    """NRMSE: 1 = predicting the mean of the target, 0 = perfect."""
    return model.evaluate(*pair(data, lag, transform), metrics=["nrmse"])[0]


def memoryless_baseline(lag, transform=None):
    """Ridge regression on x(t) alone, with the same samples as the reservoir."""
    x, y = pair(train, lag, transform)
    ridge = Ridge(alpha=1e-6).fit(x[0, WASHOUT : WASHOUT + y.shape[1]], y[0])
    x_test, y_test = pair(test, lag, transform)
    y_pred = ridge.predict(x_test[0, WASHOUT : WASHOUT + y_test.shape[1]]).reshape(y_test.shape)
    return float(nrmse(y_test, y_pred))


# ----------------------------------------------------------------------------- the tasks
print(f"Mackey-Glass, tau = {TAU}: {N_TRAIN} training steps, warm-up {WASHOUT} steps\n")
print(f"{'hidden quantity (from x(t))':40s} {'reservoir':>10s} {'memoryless':>11s}   (test NRMSE)")
tasks = [("delayed state x(t - tau)", TAU, None), ("production term a x(t - tau)/(1 + x(t - tau)^n)", TAU, production)]
models = {}
for label, lag, transform in tasks:
    models[label] = build(lag, transform=transform)
    print(f"{label[:40]:40s} {score(models[label], test, lag, transform):10.4f} {memoryless_baseline(lag, transform):11.4f}")
control = build(TAU, shuffle_targets=True)
print(f"{'control: targets in random order':40s} {score(control, test, TAU):10.4f}")

# ------------------------------------------------------------------ memory and prediction
lags = [-300, -100, -50, -30, -10, -1, 1, 5, 10, 17, 25, 40, 60, 100, 150]
errors = [score(build(lag), test, lag) for lag in lags]
print("\ntest NRMSE as a function of the lag (negative: the future)")
for lag, error in zip(lags, errors):
    print(f"  lag {lag:5d}: {error:.5f}")

# ------------------------------------------------------------------------------- plots
x_test, y_test = pair(test, TAU)
prediction = models[tasks[0][0]].predict(x_test)[0, :, 0]
t = np.arange(len(prediction)) + WASHOUT

fig, axes = plt.subplots(2, 1, figsize=(10, 7), tight_layout=True)
axes[0].plot(t[:300], test[WASHOUT : WASHOUT + 300], label="x(t), measured", color="#A8DADC")
axes[0].plot(t[:300], y_test[0, :300, 0], label="x(t - tau), truth", color="#1D3557")
axes[0].plot(t[:300], prediction[:300], label="x(t - tau), reconstructed", color="#E63946", linestyle="--")
axes[0].set_xlabel("time step")
axes[0].legend()
axes[1].semilogy(lags, errors, marker="o", color="#1D3557")
axes[1].axvline(TAU, color="gray", linestyle=":", label="tau")
axes[1].axvline(0, color="gray", linestyle="-", linewidth=0.5)
axes[1].set_xlabel("lag (negative: the future)")
axes[1].set_ylabel("test NRMSE")
axes[1].legend()
plt.show()
