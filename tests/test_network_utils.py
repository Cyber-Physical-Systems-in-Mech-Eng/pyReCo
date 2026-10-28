"""
Adjacency-matrix utilities: density, number of nodes, spectral radius and its rescaling, checked on
small matrices with known values. The spectral radius of generated graphs is in test_spectral_radius.py.
"""

import numpy as np
import pytest

from pyreco.utils_networks import compute_density, compute_spec_rad, get_num_nodes, set_spec_rad

CYCLE_3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # directed ring: 3 of 9 entries, rho = 1
CYCLE_4 = np.roll(np.eye(4, dtype=int), 1, axis=1)  # directed ring of 4 nodes
SWAP_2 = np.array([[0, 1], [1, 0]])  # rho = 1

ALL_FUNCTIONS = [compute_density, get_num_nodes, compute_spec_rad, lambda a: set_spec_rad(a, 1.0)]


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_non_square_matrices_are_rejected(function):
    with pytest.raises(ValueError, match="adjacency matrix must be square"):
        function(np.array([[0, 1, 0], [0, 0, 1]]))


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_only_numpy_arrays_are_accepted(function):
    with pytest.raises(TypeError, match="adjacency matrix must be numpy.ndarray"):
        function(CYCLE_3.tolist())


@pytest.mark.parametrize(
    "network, expected",
    [
        (CYCLE_3, 3 / 9),  # density = number of non-zero entries / n^2
        (CYCLE_4, 4 / 16),
        (np.zeros((3, 3)), 0.0),
        (np.ones((2, 2)), 1.0),
    ],
)
def test_compute_density(network, expected):
    assert compute_density(network) == pytest.approx(expected)


@pytest.mark.parametrize("network, expected", [(CYCLE_3, 3), (CYCLE_4, 4)])
def test_get_num_nodes(network, expected):
    assert get_num_nodes(network) == expected


@pytest.mark.parametrize(
    "network, expected",
    [
        (SWAP_2, 1.0),  # eigenvalues +-1
        (CYCLE_3, 1.0),  # the eigenvalues are the cube roots of unity
        (np.zeros((3, 3)), 0.0),
    ],
)
def test_compute_spec_rad(network, expected):
    assert compute_spec_rad(network) == pytest.approx(expected)


@pytest.mark.parametrize("network, target", [(SWAP_2, 2.0), (CYCLE_3, 1.5)])
def test_set_spec_rad_rescales_to_the_target(network, target):
    """A / rho(A) * target has spectral radius target; a value above 1 warns (unstable reservoirs)."""
    with pytest.warns(Warning, match="a spectral radius larger than 1 is unusual!"):
        rescaled = set_spec_rad(network, target)
    assert compute_spec_rad(rescaled) == pytest.approx(target)


def test_set_spec_rad_needs_a_link():
    with pytest.raises(ValueError, match="adjacency matrix must have at least one link"):
        set_spec_rad(np.zeros((3, 3)), 1.0)
