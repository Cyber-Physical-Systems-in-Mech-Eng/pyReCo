"""
Minimal working example: reconstruct the third state of the Lorenz system from the first two.

Only x(t) and y(t) of the Lorenz-63 system are measured, z(t) is known during a training phase
only. A reservoir computer ("reservoir observer", Lu, Pathak, Hunt, Girvan, Brockett and Ott,
Chaos 27, 041102 (2017)) is trained to infer the unmeasured z(t) at the same time step from the
measured states, with no model of the equations. This is not a forecast: input and target belong to
the same time step, and the reservoir has to supply the memory that is needed because z follows
dz/dt = x y - beta z, i.e. z is a leaky integral of the product x y.

Data layout: one long sequence per data set, shape (1, n_time, n_states). The reservoir starts
from the zero state, so the first `WASHOUT` time steps of every sequence are only a warm-up: the
inputs keep their full length and the targets leave out the first `WASHOUT` steps
(`compile(discard_transients=WASHOUT)`). `evaluate` raises an error if the targets are given in full.

Controls: x -> y and x -> z (z is observable from x alone, but much harder), and z -> x, which is
impossible: the Lorenz equations are unchanged under (x, y, z) -> (-x, -y, z), so z does not tell
the sign of x and any prediction of x from z is no better than predicting the mean (NRMSE >= 1; a flexible readout
fitted to the training data does even worse on the test data).
That the controls fail where they must is also a check that no information leaks from the targets.

The reservoir uses signed weights and an input bias, see README. The hyperparameters were found by a
60 trial optuna search on the validation data with 300 nodes; with the all-positive weights and no
bias the same hyperparameters give 0.09, and the layer defaults (spectral radius 0.9, leakage 0.5,
unit input scaling) stay at about 0.6.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyreco.custom_models import RC
from pyreco.datasets import generate_lorenz
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

DT = 0.01  # integration and sampling time step of the Lorenz system
WASHOUT = 200
N_BURN_IN, N_TRAIN, N_VAL, N_TEST = 1000, 5000, 3000, 3000
X, Y, Z = 0, 1, 2

HP = dict(spec_rad=0.545, leakage_rate=0.127, input_scaling=1.07, input_bias=0.174, alpha=1.4e-9)
RESERVOIR = dict(density=0.05, weight_distribution="sign")

# ------------------------------------------------------------------------------- data
# one trajectory, the first steps are the transient from the initial condition to the attractor
trajectory = generate_lorenz(N_TRAIN + N_VAL + N_TEST, dt=DT, washout=N_BURN_IN)
train = trajectory[:N_TRAIN]
val = trajectory[N_TRAIN : N_TRAIN + N_VAL]
test = trajectory[N_TRAIN + N_VAL :]


def split(data, inputs, targets):
    """Inputs in full length, targets without the warm-up; shapes (1, n_time, n_states)."""
    return data[None, :, inputs], data[None, WASHOUT:, targets]


def build(inputs, targets, nodes=300, seed=0, **overrides):
    x, y = split(train, inputs, targets)
    reservoir = {**RESERVOIR, **overrides.pop("reservoir", {})}
    hp = {**HP, **overrides}
    model = RC(seed=seed)
    model.add(InputLayer(input_shape=x.shape[1:]))
    model.add(
        RandomReservoirLayer(
            nodes=nodes,
            spec_rad=hp["spec_rad"],
            leakage_rate=hp["leakage_rate"],
            fraction_input=1.0,
            seed=seed,
            **reservoir,
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


def score(model, data, inputs, targets):
    """NRMSE: 1 = predicting the mean of the target, 0 = perfect."""
    x, y = split(data, inputs, targets)
    return model.evaluate(x, y, metrics=["nrmse"])[0]


# ----------------------------------------------------------------------- the main task
print(f"Lorenz-63, dt = {DT}: {N_TRAIN} training steps, warm-up {WASHOUT} steps\n")
model = build([X, Y], [Z])
print("reconstruct z from (x, y), 300 nodes")
print(f"  NRMSE  validation {score(model, val, [X, Y], [Z]):.4f}   test {score(model, test, [X, Y], [Z]):.4f}")

# the all-positive weights of the 2.x reservoir and no bias, with the same hyperparameters
default = build([X, Y], [Z], input_bias=0.0, reservoir={"weight_distribution": "positive"})
print(f"  with positive weights and no bias: test {score(default, test, [X, Y], [Z]):.4f}")

print("\nsize of the reservoir (test NRMSE of z from (x, y))")
for nodes in (30, 60, 100, 200, 300):
    small = build([X, Y], [Z], nodes=nodes)
    print(f"  {nodes:4d} nodes: {score(small, test, [X, Y], [Z]):.4f}")

# ----------------------------------------------------------------------------- controls
print("\ncontrols (test NRMSE)")
controls = [
    ("y from x", [X], [Y]),
    ("z from x", [X], [Z]),
    ("x from z (impossible: symmetry)", [Z], [X]),
    ("x, y from z (impossible: symmetry)", [Z], [X, Y]),
]
for label, inputs, targets in controls:
    print(f"  {label:38s} {score(build(inputs, targets), test, inputs, targets):.4f}")

# ------------------------------------------------------------------------------- plots
x_test, y_test = split(test, [X, Y], [Z])
z_pred = model.predict(x_test)[0, :, 0]
z_true = y_test[0, :, 0]
t = (np.arange(len(z_true)) + WASHOUT) * DT

fig, axes = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True, sharex=True)
axes[0].plot(t[:800], z_true[:800], label="z (truth)", color="#1D3557")
axes[0].plot(t[:800], z_pred[:800], label="z reconstructed from (x, y)", color="#E63946", linestyle="--")
axes[0].set_ylabel("z")
axes[0].legend()
axes[1].plot(t[:800], z_pred[:800] - z_true[:800], color="#457B9D")
axes[1].set_ylabel("error")
axes[1].set_xlabel("time")
plt.show()
