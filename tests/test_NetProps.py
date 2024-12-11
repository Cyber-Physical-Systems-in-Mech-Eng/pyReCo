import pytest
# from pyReCo.skeleton import fib, main
import numpy as np
from utils_networks import compute_density


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
    with pytest.raises(ValueError):
        compute_density(network)

    # Test with a non-numpy array input (should raise TypeError)
    network = [[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]]
    with pytest.raises(TypeError):
        compute_density(network)


