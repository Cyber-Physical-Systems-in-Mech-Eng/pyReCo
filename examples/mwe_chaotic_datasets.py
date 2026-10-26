"""
Minimal working example: one-step-ahead forecasting of chaotic systems.

pyreco.datasets generates the Lorenz-63 attractor (3 states) and the Mackey-Glass delay
system (1 state) and cuts them into windows: the inputs are the last n_in values, the targets
the following n_out values. The series is split chronologically into a training and a test
part, and the data is standardized with the statistics of the training part only.

Note that the series start from the initial condition of the system, so the first samples
are a transient. The windows are not a problem for the reservoir: every window is a fresh
start from the initial state (zeros) and only its last state, which has seen the whole window,
is paired with the target (`discard_transients` = n_in - n_out).
"""

import matplotlib.pyplot as plt

from pyreco.custom_models import RC
from pyreco.datasets import load
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

fig, axes = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)

for ax, name in zip(axes, ["lorenz", "mackey_glass"]):
    x_train, y_train, x_test, y_test, scaler = load(
        name, n_samples=2500, n_in=30, n_out=1, train_fraction=0.7, standardize=True
    )

    model = RC(seed=0)
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(
        RandomReservoirLayer(nodes=200, density=0.1, spec_rad=0.9, leakage_rate=0.5, seed=0)
    )
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    # the last reservoir state of every window predicts the next value: warm-up = n_in - n_out
    model.compile(discard_transients=29)
    model.fit(x_train, y_train)

    # nrmse is 1 for predicting the mean of the series and 0 for a perfect prediction
    nrmse = model.evaluate(x_test, y_test, metrics=["nrmse"])[0]
    print(f"{name}: one-step-ahead NRMSE on the test set: {nrmse:.4f}")

    y_pred = model.predict(x_test)
    ax.plot(y_test[:300, 0, 0], label="truth", color="#1D3557")
    ax.plot(y_pred[:300, 0, 0], label="prediction", color="#E63946", linestyle="--")
    ax.set_title(f"{name} (first state, standardized)")
    ax.set_xlabel("time step")
    ax.legend()

plt.show()
