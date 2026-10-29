"""
Minimal working example: the NARMA5 and NARMA10 benchmarks.

NARMA is a system identification task: the input u(t) is random noise and the target is the
output of a nonlinear autoregressive moving-average system of order 5 or 10, which depends on
the last 5 or 10 inputs, on a product of two inputs and on the own history. A reservoir needs
memory and nonlinearity to solve it.

    y(t+1) = 0.3 y(t) + 0.05 y(t) sum_{i<n} y(t-i) + 1.5 u(t-n+1) u(t) + 0.1,  u ~ U[0, 0.5]

pyreco.datasets.load cuts the sequence into windows: the inputs are n_in values of u, the
target is the system output right after the last input of the window. The first n_in - 1
reservoir states of every window are discarded as warm-up (discard_transients = n_in - n_out),
because only the last state is paired with the target.

The NRMSE is 1 for predicting the mean of the target and 0 for a perfect prediction.
"""

import matplotlib.pyplot as plt

from pyreco.custom_models import RC
from pyreco.datasets import load
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

fig, axes = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)

for ax, name in zip(axes, ["narma5", "narma10"]):
    x_train, y_train, x_test, y_test = load(name, n_samples=3000, n_in=50, n_out=1, seed=1)

    model = RC(seed=0)
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(
        RandomReservoirLayer(nodes=200, density=0.1, spec_rad=0.9, leakage_rate=0.9, seed=0)
    )
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    model.compile(metrics=["nrmse"], discard_transients=49)  # warm-up: n_in - n_out
    model.set_hp(alpha=1e-6)  # weak regularization of the readout
    model.fit(x_train, y_train)

    nrmse = model.evaluate(x_test, y_test, metrics=["nrmse"])[0]
    print(f"{name}: test NRMSE {nrmse:.3f}")

    y_pred = model.predict(x_test)
    ax.plot(y_test[:200, 0, 0], label="target", color="#1D3557")
    ax.plot(y_pred[:200, 0, 0], label="prediction", color="#E63946", linestyle="--")
    ax.set_title(f"{name}: test NRMSE {nrmse:.3f}")
    ax.set_xlabel("window")
    ax.legend()

plt.show()
