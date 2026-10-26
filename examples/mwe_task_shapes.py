"""
Minimal working example: input and output shapes of a task, with one data set.

All data has the shape [n_batch, n_time, n_states]. The data set is `sine_pred`: one sine per
state (every state has its own frequency), cut into windows. The inputs are n_time_in
consecutive values and the targets are the n_time_out values that follow the input window.

The cases below only differ in the number of time steps and states of inputs and targets:

1. vector to vector       1 input step  -> 1 target step
2. sequence to scalar     20 input steps -> 1 target step
3. sequence to sequence   20 input steps -> 20 target steps
4. several states         3 input states -> 2 target states
5. inputs longer than targets, with a warm-up (see below)

Warm-up: the target at time step j is paired with the reservoir state after d + j + 1 inputs,
where d is `discard_transients` (default 0). The first states depend on the initial state of the
reservoir (zeros), so a warm-up helps if the data does not start in the middle of the signal.
d must not exceed the number of input steps minus the number of target steps. For a sequence to
a scalar, d = n_time_in - 1 pairs the target with the last state, which has seen the whole input.
"""

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import x_to_x


def fit_and_evaluate(title, n_states_in, n_states_out, n_time_in, n_time_out, warm_up=0):
    X_train, X_test, y_train, y_test = x_to_x(
        "sine_pred",
        n_batch=200,
        n_states_in=n_states_in,
        n_states_out=n_states_out,
        n_time_in=n_time_in,
        n_time_out=n_time_out,
        seed=0,
    )

    model = RC(seed=0)
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=100, leakage_rate=0.3, seed=0))
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    model.compile(discard_transients=warm_up)
    model.fit(X_train, y_train)

    mae = model.evaluate(X_test, y_test, metrics=["mae"])[0]
    print(
        f"{title}: inputs {X_train.shape[1:]} -> targets {y_train.shape[1:]}, "
        f"warm-up {warm_up}, test MAE {mae:.4f}"
    )
    r2_scatter(y_true=y_test[:, :, :], y_pred=model.predict(X_test)[:, : y_test.shape[1]])


fit_and_evaluate("1 vector to vector", 1, 1, n_time_in=1, n_time_out=1)
fit_and_evaluate("2 sequence to scalar", 1, 1, n_time_in=20, n_time_out=1, warm_up=19)
fit_and_evaluate("3 sequence to sequence", 1, 1, n_time_in=20, n_time_out=20)
fit_and_evaluate("4 several states", 3, 2, n_time_in=20, n_time_out=20)
fit_and_evaluate("5 longer inputs, warm-up", 1, 1, n_time_in=30, n_time_out=20, warm_up=10)
