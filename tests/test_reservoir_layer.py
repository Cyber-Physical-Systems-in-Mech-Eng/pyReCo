"""
Implements tests for the ReservoirLayer class.
"""

import pytest
import numpy as np
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_networks import compute_density, compute_spec_rad


def test_set_weights():
    """
    Setting the reservoir weights (adjacency matrix, reservoir network) form the outside
    """

    # create a simple RC model and then set the weights
    model = RC()
    model.add(InputLayer(input_shape=(1, 100)))
    model.add(RandomReservoirLayer(nodes=100, density=0.1))
    model.add(ReadoutLayer(output_shape=(1, 100)))
    model.compile()

    # check compilation boolean flag
    assert model.reservoir_layer._is_compiled is True

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

    # now let pytest check that the new weights are correctly updated in the layer
    assert np.array_equal(model.reservoir_layer.weights, new_weights)

    # now let pytest check that the new density is correctly updated in the layer
    assert np.isclose(new_density, model.reservoir_layer.density)
    assert np.isclose(new_density, 1.0)  # since we set all weights to 1
    assert np.isclose(new_spec_rad, model.reservoir_layer.spec_rad)

    # check for inconsistent shape
    with pytest.raises(ValueError):
        model.reservoir_layer.set_weights(np.ones((50, 50)))  # wrong shape, should be (100, 100)

    # check for wrong shape (i.e. not square)
    with pytest.raises(ValueError):
        model.reservoir_layer.set_weights(np.ones((100, 50)))  # wrong shape, should be (100, 100)

    # check for different size (i.e. deviation from what was defined initially)
    with pytest.raises(ValueError):
        model.reservoir_layer.set_weights(np.ones((200, 200)))  # wrong shape, should be (100, 100)

    # now check for types
    with pytest.raises(TypeError):
        model.reservoir_layer.set_weights([1.0, 2.0, 3.0]) 


if __name__ == "__main__":
    pytest.main()
