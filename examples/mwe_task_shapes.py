"""
Minimal working example: the shapes of a supervised sequence task.

Data always has the shape [n_batch, n_time, n_states]. The three tasks below only differ in
the number of time steps of the inputs and of the targets:

- vector to vector:     1 input step   -> 1 target step
- sequence to scalar:   many input steps -> 1 target step
- sequence to sequence: many input steps -> as many target steps

The last part evaluates the sequence-to-scalar model with k-fold cross-validation and draws
the reservoir network.
"""

from pyreco.cross_validation import cross_val
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import sequence_to_scalar, sequence_to_sequence, vector_to_vector


def fit_and_evaluate(title, data, discard_transients=0, **reservoir_kwargs):
    X_train, X_test, y_train, y_test = data

    model = RC(seed=0)
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(RandomReservoirLayer(nodes=100, **reservoir_kwargs))
    model.add(ReadoutLayer(output_shape=y_train.shape[1:]))
    model.compile(discard_transients=discard_transients)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = model.evaluate(X_test, y_test, metrics=["mae"])[0]
    print(
        f"{title}: input {X_train.shape[1:]} -> output {y_train.shape[1:]}, "
        f"test MAE {mae:.4f}"
    )
    r2_scatter(y_true=y_test, y_pred=y_pred)
    return model, X_train, y_train


# predict sin(t + dt) from sin(t)
fit_and_evaluate(
    "vector to vector",
    vector_to_vector("sine_pred", n_batch=200, n_states=1, seed=0),
    activation="sigmoid",
    leakage_rate=0.1,
)

# predict sin(T + dt) from the sequence sin([0, ..., T]); the first 10 reservoir states
# are discarded as a warm-up
model, X_train, y_train = fit_and_evaluate(
    "sequence to scalar",
    sequence_to_scalar("sine_pred", n_batch=200, n_states=1, n_time_in=20, seed=0),
    discard_transients=10,
    leakage_rate=0.1,
)

# translate a sequence of sines into a sequence of cosines
fit_and_evaluate(
    "sequence to sequence",
    sequence_to_sequence("sine_to_cosine", n_batch=100, n_states=1, n_time=20, seed=0),
)

# k-fold cross-validation of the sequence-to-scalar model: estimate bias and variance
scores, mean, std = cross_val(model, X_train, y_train, n_splits=5, metric="mse", seed=0)
print(f"Cross-validation MSE per fold: {scores}")
print(f"Mean MSE: {mean:.4f}, standard deviation: {std:.4f}")

# draw the reservoir network (pass save=True, file_name=..., file_type="svg" to write a file)
model.model_visualize()
