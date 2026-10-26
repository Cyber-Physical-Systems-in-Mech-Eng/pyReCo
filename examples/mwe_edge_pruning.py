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
)

model_pruned, history = pruner.prune(
    model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
)

n_iterations = len(history)
print(f"took {n_iterations} iterations to prune the model")

# Extract per-iteration loss, node count, and edge count from history
losses = [history[i]["starting_model"]["loss"] for i in range(n_iterations)]
num_edges = [history[i]["starting_model"]["num_edges"] for i in range(n_iterations)]
num_nodes = [history[i]["starting_model"]["num_nodes"] for i in range(n_iterations)]

# Extract graph properties across iterations
# graph_props is a dict of property name -> scalar, collect into arrays
graph_prop_keys = list(history[0]["starting_model"]["graph_props"].keys())
graph_props_over_time = {
    key: [history[i]["starting_model"]["graph_props"][key] for i in range(n_iterations)]
    for key in graph_prop_keys
}

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
#   candidate edge properties?
iteration = 2
candidates = history[iteration]["candidates"]
winner = history[iteration]["winner"]

candidate_edges = list(candidates.keys())
candidate_scores = [candidates[e]["score"] for e in candidate_edges]
candidate_betweenness = [candidates[e]["edge_props"]["betweenness"] for e in
                         candidate_edges]

plt.figure()
plt.hist(x=candidate_betweenness)
plt.title(
    f"pruned edge {winner} had betweenness "
    f"{candidates[winner]['edge_props']['betweenness']:.4f} in iteration {iteration}"
)
plt.legend(["pruning candidate edges"])
plt.xlabel("betweenness")
plt.show()
