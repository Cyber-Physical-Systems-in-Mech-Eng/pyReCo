import pytest
import numpy as np
from pyreco.utils_networks import (
    compute_density,
    get_num_nodes,
    compute_spec_rad,
    set_spec_rad,
)

__author__ = "Manish Yadav"
__copyright__ = "Manish Yadav"
__license__ = "MIT"


def test_compute_density():
    # Test with a 3x3 matrix with 3 links
    network = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
    expected_density = 3 / 9  # 3 links in a 3x3 matrix
    assert compute_density(network) == pytest.approx(expected_density)

    # Test with a 4x4 matrix with 4 links
    network = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0]])
    expected_density = 4 / 16  # 4 links in a 4x4 matrix
    assert compute_density(network) == pytest.approx(expected_density)

    # Test with an empty matrix
    network = np.zeros((3, 3))
    expected_density = 0 / 9  # 0 links in a 3x3 matrix
    assert compute_density(network) == pytest.approx(expected_density)

    # Test with a full matrix
    network = np.ones((2, 2))
    expected_density = 4 / 4  # 4 links in a 2x2 matrix
    assert compute_density(network) == pytest.approx(expected_density)

    # Test with a non-square matrix (should raise ValueError)
    network = np.array([[0, 1, 0],
                        [0, 0, 1]])
    with pytest.raises(ValueError, match="adjacency matrix must be square"):
        compute_density(network)

    # Test with a non-numpy array input (should raise TypeError)
    network = [[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]]
    with pytest.raises(TypeError, match="adjacency matrix must be numpy.ndarray"):
        compute_density(network)


def test_get_num_nodes():
    # Test with a 3x3 matrix
    network = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
    assert get_num_nodes(network) == 3

    # Test with a 4x4 matrix
    network = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0]])
    assert get_num_nodes(network) == 4

    # Test with a non-square matrix (should raise ValueError)
    network = np.array([[0, 1, 0],
                        [0, 0, 1]])
    with pytest.raises(ValueError, match="adjacency matrix must be square"):
        get_num_nodes(network)

    # Test with a non-numpy array input (should raise TypeError)
    network = [[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]]
    with pytest.raises(TypeError, match="adjacency matrix must be numpy.ndarray"):
        get_num_nodes(network)


def test_compute_spec_rad():
    # Test with a 2x2 matrix
    network = np.array([[0, 1],
                        [1, 0]])
    expected_spec_rad = 1.0  # The spectral radius of this matrix is 1
    assert compute_spec_rad(network) == pytest.approx(expected_spec_rad)

    # Test with a 3x3 matrix
    network = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
    expected_spec_rad = 1.0  # The spectral radius of this matrix is 1
    assert compute_spec_rad(network) == pytest.approx(expected_spec_rad)

    # Test with an empty matrix
    network = np.zeros((3, 3))
    expected_spec_rad = 0.0  # The spectral radius of a zero matrix is 0
    assert compute_spec_rad(network) == pytest.approx(expected_spec_rad)

    # Test with a non-square matrix (should raise ValueError)
    network = np.array([[0, 1, 0],
                        [0, 0, 1]])
    with pytest.raises(ValueError, match="adjacency matrix must be square"):
        compute_spec_rad(network)

    # Test with a non-numpy array input (should raise TypeError)
    network = [[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]]
    with pytest.raises(TypeError, match="adjacency matrix must be numpy.ndarray"):
        compute_spec_rad(network)


def test_set_spec_rad():
    # Test with a 2x2 matrix
    network = np.array([[0, 1],
                        [1, 0]])
    new_spec_rad = 2.0
    with pytest.warns(Warning, match="a spectral radius larger than 1 is unusual!"):
        modified_network = set_spec_rad(network, new_spec_rad)
        assert compute_spec_rad(modified_network) == pytest.approx(new_spec_rad)

    # Test with a 3x3 matrix
    network = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
    new_spec_rad = 1.5
    with pytest.warns(Warning, match="a spectral radius larger than 1 is unusual!"):
        modified_network = set_spec_rad(network, new_spec_rad)
        assert compute_spec_rad(modified_network) == pytest.approx(new_spec_rad)

    # Test with an empty matrix (should raise ValueError)
    network = np.zeros((3, 3))
    new_spec_rad = 1.0
    with pytest.raises(ValueError, match="adjacency matrix must have at least one link"):
        set_spec_rad(network, new_spec_rad)

    # Test with a non-square matrix (should raise ValueError)
    network = np.array([[0, 1, 0],
                        [0, 0, 1]])
    new_spec_rad = 1.0
    with pytest.raises(ValueError, match="adjacency matrix must be square"):
        set_spec_rad(network, new_spec_rad)

    # Test with a non-numpy array input (should raise TypeError)
    network = [[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]]
    new_spec_rad = 1.0
    with pytest.raises(TypeError, match="adjacency matrix must be numpy.ndarray"):
        set_spec_rad(network, new_spec_rad)
