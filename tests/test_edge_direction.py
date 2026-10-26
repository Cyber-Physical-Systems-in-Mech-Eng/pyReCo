"""
Tests that the reservoir dynamics follow the same edge convention as the graph
analysis: weights[u, v] is the edge u -> v (u sends, v receives).
"""

import numpy as np
import networkx as nx

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer


N = 30  # large enough that random ER construction is numerically safe


def _pad(small: np.ndarray) -> np.ndarray:
    out = np.zeros((N, N) if small.ndim == 2 else N)
    out[tuple(slice(0, s) for s in small.shape)] = small
    return out


def _model(weights: np.ndarray, r0: np.ndarray) -> RC:
    """Compiled RC with hand-set weights (padded to N nodes), zero input, alpha=1."""
    n_nodes = N
    weights, r0 = _pad(weights), _pad(r0)
    model = RC()
    model.add(InputLayer(input_shape=(1, 1)))
    model.add(RandomReservoirLayer(nodes=n_nodes, density=0.3))
    model.add(ReadoutLayer(output_shape=(1, 1)))
    model.compile()

    model.reservoir_layer.weights = weights.astype(float)
    model.reservoir_layer.leakage_rate = 1.0
    model.reservoir_layer.initial_res_states = r0.astype(float)
    model.input_layer.weights = np.zeros((n_nodes, 1))
    return model


def test_edge_u_to_v_drives_node_v():
    # single edge 0 -> 1 in the networkx sense, node 0 active
    A = np.zeros((3, 3))
    A[0, 1] = 1.0
    model = _model(A, r0=np.array([1.0, 0.0, 0.0]))

    states = model.compute_reservoir_state(np.zeros((1, 1, 1)))

    assert states[0, 1, 1] != 0.0  # receiver 1 is driven
    assert states[0, 1, 0] == 0.0  # sender 0 gets nothing back
    assert states[0, 1, 2] == 0.0


def test_edge_u_to_v_does_not_drive_node_u():
    # same edge, but only the receiver is active: nothing may flow backwards
    A = np.zeros((3, 3))
    A[0, 1] = 1.0
    model = _model(A, r0=np.array([0.0, 1.0, 0.0]))

    states = model.compute_reservoir_state(np.zeros((1, 1, 1)))

    assert np.allclose(states[0, 1], 0.0)


def test_removing_edge_cuts_exactly_that_signal_path():
    # directed 3-cycle 0 -> 1 -> 2 -> 0
    A = np.zeros((3, 3))
    A[0, 1] = A[1, 2] = A[2, 0] = 1.0
    r0 = np.array([1.0, 0.0, 0.0])
    x = np.zeros((1, 2, 1))

    before = _model(A.copy(), r0).compute_reservoir_state(x)
    assert before[0, 1, 1] != 0.0  # 0 -> 1 active
    assert before[0, 2, 2] != 0.0  # 1 -> 2 active in the second step

    pruned = _model(A.copy(), r0)
    pruned.remove_reservoir_edges([(0, 1)])
    after = pruned.compute_reservoir_state(x)

    assert np.allclose(after[0, 1:], 0.0)  # path 0 -> 1 -> 2 is cut


def test_dynamics_agree_with_networkx_graph_view():
    # the nodes that receive activity from node s are exactly the successors of s
    ref = RC()
    ref.add(InputLayer(input_shape=(1, 1)))
    ref.add(RandomReservoirLayer(nodes=N, density=0.1))
    ref.add(ReadoutLayer(output_shape=(1, 1)))
    ref.compile()

    A = ref.reservoir_layer.weights.copy()
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    source = 0
    r0 = np.zeros(N)
    r0[source] = 1.0
    model = _model(A, r0)

    states = model.compute_reservoir_state(np.zeros((1, 1, 1)))
    driven = set(np.nonzero(states[0, 1])[0])

    assert driven == set(G.successors(source))
