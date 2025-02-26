"""
Testing the deletion of nodes in the reservoir computer
"""

from matplotlib import pyplot as plt
from pyreco.utils_data import sequence_to_sequence as seq_2_seq
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.optimizers import RidgeSK
import numpy as np

"""
use case: we train a RC, and then choose to delete a set of nodes from the reservoir.
"""


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
        nodes=200, density=0.1, activation="tanh", leakage_rate=0.1, fraction_input=1.0
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
Now choose to delete some nodes from the reservoir
"""

nodes_to_delete = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# delete nodes
model.remove_reservoir_nodes(nodes_to_delete)
model.fit(X_train, y_train)
print(f"score after node removal: \t{model.evaluate(x=X_test, y=y_test)[0]:.4f}")

print(f"nodes in reservoir: \t{model.reservoir_layer.nodes}")
print(f"density of reservoir: \t{model.reservoir_layer.density:.3f}")


"""
Now let's cut reservoir nodes and see how the performance changes
"""
num_node_prune = 3
scores = []
num_nodes = []
remaining_nodes = model.reservoir_layer.nodes
while (remaining_nodes - num_node_prune) > 0:
    nodes_to_delete = np.random.choice(
        model.reservoir_layer.nodes, num_node_prune, replace=False
    ).tolist()
    print(f"nodes to delete: {nodes_to_delete}")
    print(f"nodes before deletion: {model.readout_layer.readout_nodes}")
    model.remove_reservoir_nodes(nodes_to_delete)
    print(f"number of nodes in the RC: {model.reservoir_layer.nodes}\n")
    model.fit(X_train, y_train)
    scores.append(model.evaluate(x=X_test, y=y_test)[0])
    num_nodes.append(model.reservoir_layer.nodes)
    remaining_nodes = model.reservoir_layer.nodes

plt.figure()
plt.plot(num_nodes, scores, "o-")
plt.xlabel("Number of nodes in reservoir")
plt.ylabel("MSE")
plt.title("Loss vs. number of nodes in reservoir")
plt.show()
