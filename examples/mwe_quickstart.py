"""
Quick start: train a reservoir computer to forecast the Lorenz system one step ahead.

Question: what are the minimal steps to get a reservoir computer (RC) running in pyreco?

    1. get data         ->  windows of shape [n_samples, n_time, n_states]
    2. build the model  ->  input layer, random reservoir, readout layer
    3. compile and fit  ->  only the readout is trained (ridge regression), which takes a moment
    4. evaluate         ->  on data that the model has never seen

Run it with `python examples/mwe_quickstart.py`. Expect a test NRMSE of about 0.001
(NRMSE = 1 is the error of always predicting the mean, 0 is a perfect prediction).
"""

import matplotlib.pyplot as plt

from pyreco.custom_models import RC
from pyreco.datasets import load
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.optimizers import RidgeSK

# ---------------------------------------------------------------------------
# 1) DATA
# ---------------------------------------------------------------------------
# The Lorenz system has 3 states. `load` simulates it and cuts the series into windows:
# the input is the last n_in = 30 values of the state, the target is the next n_out = 1 value.
# The first 70 % of the series are used for training, the remaining 30 % for testing (the split
# is chronological, so the test data is truly "the future"). With `standardize=True` every state
# is scaled to zero mean and unit variance, using the statistics of the training data only.
x_train, y_train, x_test, y_test, scaler = load(
    "lorenz", n_samples=2500, n_in=30, n_out=1, train_fraction=0.7, standardize=True
)
print(f"training data: x {x_train.shape}, y {y_train.shape}")  # [n_samples, n_time, n_states]
print(f"test data:     x {x_test.shape}, y {y_test.shape}")

# ---------------------------------------------------------------------------
# 2) MODEL
# ---------------------------------------------------------------------------
# An RC consists of three layers. The `seed` makes the random parts reproducible.
model = RC(seed=0)

# input layer: only needs to know the shape of one input window, [n_time, n_states]
model.add(InputLayer(input_shape=x_train.shape[1:]))

# reservoir: a random recurrent network that is NOT trained.
#   nodes        number of neurons in the reservoir
#   density      fraction of all possible connections between nodes that exist
#   spec_rad     spectral radius of the reservoir matrix (< 1 gives a fading memory)
#   leakage_rate how fast the nodes forget their previous state (1 = no leakage)
model.add(
    RandomReservoirLayer(nodes=200, density=0.1, spec_rad=0.9, leakage_rate=1.0, seed=0)
)

# readout layer: the only trained part; maps reservoir states to the target shape [n_out, n_states]
model.add(ReadoutLayer(output_shape=y_train.shape[1:]))

# ---------------------------------------------------------------------------
# 3) COMPILE AND FIT
# ---------------------------------------------------------------------------
# Every window starts with the reservoir at rest (all states zero), so the first states are not
# informative yet. The target is paired with the state after the whole window has been fed in:
# we skip the first n_in - n_out = 29 states ("warm-up", `discard_transients`).
# The readout is a ridge regression; alpha is the strength of the regularization (small, because
# the data is noise-free).
model.compile(optimizer=RidgeSK(alpha=1e-6), discard_transients=29)
model.fit(x_train, y_train)

# ---------------------------------------------------------------------------
# 4) EVALUATE AND PLOT
# ---------------------------------------------------------------------------
# evaluate() returns one value per requested metric, in the same order
nrmse = model.evaluate(x_test, y_test, metrics=["nrmse"])[0]
print(f"one-step-ahead NRMSE on the test set: {nrmse:.4f}")

# predictions have the same shape as the targets
y_pred = model.predict(x_test)

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, tight_layout=True)
for state, ax in enumerate(axes):
    ax.plot(y_test[:300, 0, state], label="truth", color="#1D3557")
    ax.plot(y_pred[:300, 0, state], label="prediction", color="#E63946", linestyle="--")
    ax.set_ylabel(f"state {state} (standardized)")
axes[0].set_title(f"Lorenz, one step ahead on unseen data (NRMSE = {nrmse:.4f})")
axes[0].legend()
axes[-1].set_xlabel("time step")
plt.show()

# Next steps: mwe_task_shapes.py (other input/output shapes), mwe_hyperparameter_tuning.py
# (find better hyperparameters), mwe_edge_pruning.py (thin out the reservoir).
