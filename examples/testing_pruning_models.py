import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import platform


# make pyreco available locally
if platform.system() == 'Windows':  # WINDOWS
    curr_loc = os.getcwd()
    pyreco_path = os.path.join('C:\\',*curr_loc.split('\\')[1:-1], 'src')
    sys.path.append(pyreco_path)
elif platform.system() == 'Darwin':  # MAC
    curr_loc = os.getcwd()
    pyreco_path = curr_loc + '/src'
    sys.path.append(pyreco_path)
    
elif platform.system() == 'Linux':  # LINUX
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, 'src')
    sys.path.insert(0, src_path)

from pyreco.utils_data import sequence_to_sequence
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.plotting import r2_scatter
from pyreco.network_prop_extractor import NetworkQuantifier

"""
Classic RC example
"""

# some testing data: predict a sine signal.
X_train, X_test, y_train, y_test = sequence_to_sequence(name='sine_pred', n_batch=10, n_states=2, n_time=1000)

# set the dimensions
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

num_init_nodes = 40
max_perf_drop = 0.5  # allow the network to reduce performance by x% w.r.t. to original one

model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(RandomReservoirLayer(nodes=num_init_nodes, density=0.2, activation='tanh', leakage_rate=0.9,
                                  fraction_input=0.5))
model_rc.add(ReadoutLayer(output_shape, fraction_out=0.5))

# Compile the model
model_rc.compile(optimizer='ridge', metrics=['mean_squared_error'])

# Create a custom NetworkPropExtractor
prop_extractor = NetworkQuantifier(quantities=['density', 'spectral_radius', 'clustering_coefficient'])

# Train the model with pruning
history = model_rc.fit_prune(X_train, y_train, max_perf_drop=max_perf_drop, frac_rem_nodes=0.2, #patience=5,
                             prop_extractor=prop_extractor)

# Plot pruning results
plt.figure(figsize=(12, 8))
plt.axhline(y=history['pruned_nodes_scores'][0] * (1+max_perf_drop), linestyle='dashed', color='gray',
            label=f'threshold (+{max_perf_drop*100:.0f}%)')
plt.plot(history['num_nodes'], history['pruned_nodes_scores'], marker='.',
         markersize=8, label='pruned RC')
plt.plot(history['num_nodes'][0], history['pruned_nodes_scores'][0], color='red', marker='.', markersize=12, label='original RC')
plt.ylabel('loss')
plt.xlabel('number of nodes')
plt.title(f'initial reservoir size = {num_init_nodes}, final reservoir size = {history["num_nodes"][-1]}')
plt.legend()
# plt.savefig('pruning_results.png')
plt.show()

# Plot network properties
fig, axs = plt.subplots(len(history['network_properties'][0]), 1, figsize=(12, 4*len(history['network_properties'][0])))
fig.suptitle('Network Properties During Pruning')

for i, prop in enumerate(history['network_properties'][0].keys()):
    values = [props[prop] for props in history['network_properties']]
    axs[i].plot(history['num_nodes'], values, marker='o')
    axs[i].set_title(prop.capitalize())
    axs[i].set_xlabel('Number of Nodes')
    axs[i].set_ylabel('Value')
    axs[i].grid(True)

plt.tight_layout()
# plt.savefig('network_properties.png')
plt.show()

# Make predictions for new data using the pruned reservoir
y_pred = model_rc.predict(X_test)
print(f'shape of predictions on test set: {y_pred.shape}')

# Evaluate the model
loss_rc = model_rc.evaluate(X_test, y_test, metrics=['mae'])
print(f'Test model loss: {loss_rc}')

# plot predictions vs. ground truth
r2_scatter(y_true=y_test, y_pred=y_pred)

plt.figure()
plt.plot(y_test[0,:,0], label='ground truth', marker='.')
plt.plot(y_pred[0,:,0], label='prediction', marker='.')
plt.title('test set')
plt.legend()
plt.xlabel('time')
plt.ylabel('x1')
plt.show()

def print_and_compare_weights(model, history):
    """
    Prints and compares the reservoir and readout weights after pruning.
    
    Args:
    model (RC): The trained and pruned RC model.
    history (dict): The history dictionary returned by fit_prune.
    """
    print("\nComparing weights after pruning:")
    
    # Get the final weights
    final_reservoir_weights = model.reservoir_layer.weights
    final_readout_weights = model.readout_layer.weights
    
    # Get the original number of nodes
    original_nodes = history['num_nodes'][0]
    final_nodes = history['num_nodes'][-1]
    
    print(f"Original nodes: {original_nodes}, Final nodes: {final_nodes}")
    
    print("\nFinal Reservoir Weights (shape: {}):\n".format(final_reservoir_weights.shape))
    print(final_reservoir_weights)
    
    print("\nFinal Readout Weights (shape: {}):\n".format(final_readout_weights.shape))
    print(final_readout_weights)
    
    # Check consistency
    pruned_nodes = set(range(original_nodes)) - set(range(final_nodes))
    inconsistencies = []
    
    for node in pruned_nodes:
        if node < final_readout_weights.shape[0] and not np.allclose(final_readout_weights[node], 0):
            inconsistencies.append(node)
    
    if inconsistencies:
        print("\nInconsistencies found for nodes:", inconsistencies)
        print("These nodes were pruned from the reservoir but have non-zero readout weights.")
        print("\nInconsistent Readout Weights:")
        for node in inconsistencies:
            print(f"Node {node}: {final_readout_weights[node]}")
        
        print("\nCorresponding Reservoir Weights:")
        for node in inconsistencies:
            if node < final_reservoir_weights.shape[0]:
                print(f"Node {node} row: {final_reservoir_weights[node]}")
                print(f"Node {node} column: {final_reservoir_weights[:, node]}")
            else:
                print(f"Node {node} is outside the range of the reservoir weights matrix.")
    else:
        print("\nNo inconsistencies found. Pruning is correctly reflected in both layers.")

print_and_compare_weights(model_rc, history)



