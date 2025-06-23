"""
Implements tests for the ReservoirLayer class.
"""

import pytest
import numpy as np

from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer

from pyreco.utils_networks import compute_density, compute_spec_rad




"""
Setting the reservoir weights (adjacency matrix, reservoir network) form the outside
"""

def test_set_weights():

    # create a simple RC model and then set the weights
    model = RC()
    model.add(InputLayer(input_shape=(1, 100)))
    model.add(RandomReservoirLayer(nodes=100, density=0.1))
    model.add(ReadoutLayer(output_shape=(1, 100)))

    # Compile the model
    model.compile()

    # obtain current weights
    curr_weights = model.reservoir_layer.weights
    curr_density = compute_density(curr_weights)
    curr_spec_rad = compute_spec_rad(curr_weights)

    # now let pytest check that the current weights are correctly set in the layer
    assert np.isclose(curr_density, model.reservoir_layer.density)
    assert np.isclose(curr_spec_rad, model.reservoir_layer.spec_rad)


    # Create new weights with the same shape and set them
    new_weights = np.ones_like(curr_weights)
    model.reservoir_layer.set_weights(new_weights)

    new_density = compute_density(new_weights)
    new_spec_rad = compute_spec_rad(new_weights)
    print(f"New density: {new_density}, New spectral radius: {new_spec_rad}")

    # now let pytest check that the new weights are correctly updated in the layer
    assert np.array_equal(model.reservoir_layer.weights, new_weights)

    # now let pytest check that the new density is correctly updated in the layer
    assert np.isclose(new_density, model.reservoir_layer.density)
    assert np.isclose(new_spec_rad, model.reservoir_layer.spec_rad)
    
    








    # Create a simple reservoir network
    network = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
    
    # Initialize the reservoir layer with the network
    model = ReservoirLayer(network=network)

    # Check initial weights
    assert np.array_equal(model.reservoir_layer.get_weights(), network)

    # Create new weights to set
    weights_new = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])

    # Set new weights

model.reservoir_layer.set_weights(weights_new)