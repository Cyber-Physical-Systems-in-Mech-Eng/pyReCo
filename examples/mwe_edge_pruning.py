"""
Minimal working example: pruning the edges of a reservoir.

Question: how many edges can EdgePruner remove from a trained reservoir computer before the validation error gets worse?
"""

# Test edge pruning
import matplotlib.pyplot as plt
from pyreco.utils_data import sequence_to_sequence as seq_2_seq
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.optimizers import RidgeSK
from pyreco.edge_pruning import EdgePruner

# Get some data
X_train, X_test, y_train, y_test = seq_2_seq(
    name="sine_pred", n_batch=20, n_states=2, n_time=150
)

input_shape = X_train.shape[1:]
output_shape = y_train.shape[1:]

# Build classical RC
model = RC()
model.add(InputLayer(input_shape=input_shape))
model.add(
    RandomReservoirLayer(
        nodes=50,
        density=0.2,
        activation="tanh",
        leakage_rate=0.1,
        fraction_input=1.0,
    ),
)
model.add(ReadoutLayer(output_shape, fraction_out=0.9))

# Compile model
optim = RidgeSK(alpha=0.5)
model.compile(
    optimizer=optim,
    metrics=["mean_squared_error"],
)

# Train the model
model.fit(X_train, y_train)

print(f"score: \t\t\t{model.evaluate(x=X_test, y=y_test)[0]:.4f}")


"""
Edge Pruning Part
"""

# prune the model
pruner = EdgePruner(
    stopping_criterion=["min_edges", "patience"],
    patience=2,
    min_num_edges=400,
    candidate_fraction=0.2,  # 1.0 would try out every possible edge per iteration
    remove_isolated_nodes=True,
    metrics=["mse"],
    return_best_model=True,
    record_candidates=True,  # also keep a table of all candidates of every iteration
)

model_pruned, history = pruner.prune(
    model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
)

# The history holds the reservoir before the pruning ("initial") and, per iteration, the one that survived it
iterations = [i for i in history if isinstance(i, int)]
print(f"took {len(iterations)} iterations to prune the model")
states = [history["initial"]] + [history[i]["state"] for i in iterations]

# Extract the loss, the node count, and the edge count over the iterations
losses = [state["val_loss"] for state in states]
num_edges = [len(state["edges"]) for state in states]
num_nodes = [state["weights"].shape[0] for state in states]

# Extract the network features (density, spectral radius, ...) over the iterations
graph_prop_keys = list(history["initial"]["network"])
graph_props_over_time = {key: [state["network"][key] for state in states] for key in graph_prop_keys}

for key in graph_prop_keys:
    print(
        f"{key}: \t initial model {graph_props_over_time[key][0]:.4f}; "
        f"\t final model: {graph_props_over_time[key][-1]:.4f}"
    )

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(num_edges, losses, label="loss")
plt.xlabel("number of edges")
plt.ylabel("loss")
plt.subplot(1, 2, 2)
for key in graph_prop_keys:
    plt.plot(num_edges, graph_props_over_time[key], label=key)
plt.xlabel("number of edges")
plt.yscale("log")
plt.legend()
plt.show()


# Investigate a single decision: which edge was pruned and what were the
#   properties of the candidate edges?
iteration = min(2, len(iterations) - 1)
candidates = history[iteration]["candidates"]  # arrays with one entry per candidate
winner = history[iteration]["winner_edge"]

plt.figure()
plt.hist(x=candidates["removed_edge"]["betweenness"])
plt.title(
    f"pruned edge {winner} had betweenness "
    f"{history[iteration]['removed_edge']['betweenness']:.4f} in iteration {iteration}"
)
plt.legend(["pruning candidate edges"])
plt.xlabel("betweenness")
plt.show()
