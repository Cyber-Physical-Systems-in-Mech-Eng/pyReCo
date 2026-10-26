# test the pruning
import numpy as np
import matplotlib.pyplot as plt
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
        nodes=50,
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
Pruning Part
"""
from pyreco.pruning import NetworkPruner

# prune the model
pruner = NetworkPruner(
    stop_at_minimum=True,
    min_num_nodes=20,
    patience=2,
    candidate_fraction=0.1,  # 1.0 would try out every possible node
    remove_isolated_nodes=False,
    metrics=["mse"],
    maintain_spectral_radius=False,
    return_best_model=True,
)

model_pruned, history = pruner.prune(
    model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
)

print(f"took {history['iteration'][-1]+1} iterations to prune the model")
for key in history["graph_props"].keys():
    print(
        f"{key}: \t initial model {history['graph_props'][key][0]:.4f}; \t final model: {history['graph_props'][key][-1]:.4f}"
    )

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history["num_nodes"], history["loss"], label="loss")
plt.xlabel("number of nodes")
plt.ylabel("loss")
plt.subplot(1, 2, 2)
for key in history["graph_props"].keys():
    plt.plot(history["num_nodes"], history["graph_props"][key], label=key)
plt.xlabel("number of nodes")
plt.yscale("log")
plt.legend()
plt.show()


# investigate a single decision: which node was pruned, and which pruning candidate properties do we have?
iteration = 2
plt.figure()
plt.hist(x=history["candidate_node_props"][iteration]["degree"])
plt.title(
    f'pruned node degree was {history["del_node_props"]["degree"][iteration]} in iteration {iteration}'
)
plt.legend(["pruning candidate nodes"])
plt.xlabel("degree")
plt.show()
