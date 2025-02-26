# test the pruning

from pyreco.utils_data import sequence_to_sequence as seq_2_seq
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.optimizers import RidgeSK

# get some data
X_train, X_test, y_train, y_test = seq_2_seq(
    name="sine_pred", n_batch=20, n_states=2, n_time=150
)

input_shape = X_train.shape[1:]
output_shape = y_train.shape[1:]

# build a classical RC
model = RC()
model.add(InputLayer(input_shape=input_shape))
model.add(
    RandomReservoirLayer(
        nodes=100,
        density=0.1,
        activation="tanh",
        leakage_rate=0.1,
        fraction_input=1.0,
    ),
)
model.add(ReadoutLayer(output_shape, fraction_out=0.9))

# Compile the model
optim = RidgeSK(alpha=0.5)
model.compile(
    optimizer=optim,
    metrics=["mean_squared_error"],
)

# Train the model
model.fit(X_train, y_train)

print(f"score: \t\t\t{model.evaluate(x=X_test, y=y_test)[0]:.4f}")


"""
Now prune the given model
"""
from pyreco.pruning import NetworkPruner

# prune the model
pruner = NetworkPruner(
    stop_at_minimum=True,
    min_num_nodes=20,
    patience=5,
    candidate_fraction=0.2,
    criterion="mse",
    metrics=["mse", "mae"],
    maintain_spectral_radius=False,
    remove_isolated_nodes=False,
)

model_pruned, history = pruner.prune(
    model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
)
