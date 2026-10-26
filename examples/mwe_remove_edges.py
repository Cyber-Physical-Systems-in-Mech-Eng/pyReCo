import numpy as np
from matplotlib import pyplot as plt
from pyreco.utils_data import sequence_to_sequence as seq_2_seq
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.optimizers import RidgeSK

"""
Use case: we train a RC, and then choose to delete a set of edges from the reservoir.
"""


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
        nodes=30, density=0.1, activation="tanh", leakage_rate=0.1, fraction_input=1.0
    ),
)
model.add(ReadoutLayer(output_shape, fraction_out=0.9))

# Compile model
optim = RidgeSK(alpha=0.5)
model.compile(
    optimizer=optim,
    metrics=["mean_squared_error"],
)

# Train model
model.fit(X_train, y_train)

print(f"score: \t\t\t{model.evaluate(x=X_test, y=y_test)[0]:.4f}")


"""
Now choose to delete some edges from the reservoir
"""

edges_to_delete = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

model.remove_reservoir_edges(edges_to_delete)
model.fit(X_train, y_train)
print(f"score after edge removal: \t{model.evaluate(x=X_test, y=y_test)[0]:.4f}")

weights = model.reservoir_layer.weights
rows, cols = np.where(weights != 0)
print(f"nodes in reservoir: \t{model.reservoir_layer.nodes}")
print(f"edges in reservoir: \t{len(rows)}")
print(f"density of reservoir: \t{model.reservoir_layer.density:.3f}")


"""
Now let's cut reservoir edges and see how the performance changes
"""
num_edge_prune = 5
scores = []
num_edges = []

while True:
    weights = model.reservoir_layer.weights
    rows, cols = np.where(weights != 0)
    current_edges = list(zip(rows.tolist(), cols.tolist()))

    if len(current_edges) - num_edge_prune <= 0:
        break

    edges_to_delete = [
        current_edges[i]
        for i in np.random.choice(len(current_edges), num_edge_prune, replace=False)
    ]
    print(f"edges to delete: {edges_to_delete}")
    model.remove_reservoir_edges(edges_to_delete)
    print(f"number of edges in the RC: {len(current_edges) - num_edge_prune}\n")
    model.fit(X_train, y_train)
    scores.append(model.evaluate(x=X_test, y=y_test)[0])
    num_edges.append(len(current_edges) - num_edge_prune)

plt.figure()
plt.plot(num_edges, scores, "o-")
plt.xlabel("Number of edges in reservoir")
plt.ylabel("MSE")
plt.title("Loss vs. number of edges in reservoir")
plt.show()
