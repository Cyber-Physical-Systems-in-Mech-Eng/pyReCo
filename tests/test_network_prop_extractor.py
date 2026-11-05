"""
NetworkQuantifier and NodePropExtractor, tested on small networks whose properties are known by hand.

The main network has 4 nodes and the edges 0 -> 1, 1 -> 2, 2 -> 0, 2 -> 3 (a directed 3-cycle with a tail):

    0 --> 1 --> 2 --> 3
    ^___________|

Degrees (in, out): node 0 (1, 1), node 1 (1, 1), node 2 (1, 2), node 3 (1, 0).
"""

import networkx as nx
import numpy as np
import pytest

from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.network_prop_extractor import NetworkQuantifier, NodePropExtractor
from pyreco.node_analyzer import NodeAnalyzer

# A[u, v] is the weight of the edge u -> v. The weights of the cycle multiply to 2 * (-1) * 0.5 = -1.
CYCLE_WITH_TAIL = np.array(
    [
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.5, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)

# two nodes that are connected in both directions
TWO_WAY = np.array([[0.0, 0.5], [0.5, 0.0]])


# ------------------------------------------------------------------- network properties


def test_network_properties_of_the_cycle_with_tail():
    props = NetworkQuantifier().extract_properties(CYCLE_WITH_TAIL)
    assert set(props) == {"density", "spectral_radius", "in_degree_av", "out_degree_av", "clustering_coefficient"}
    assert props["density"] == pytest.approx(4 / 12)  # 4 edges of N (N - 1) = 12 possible ones
    assert props["in_degree_av"] == pytest.approx(1.0)  # 4 edges / 4 nodes
    assert props["out_degree_av"] == pytest.approx(1.0)
    # the cycle has the eigenvalues of the cube roots of the product of its weights (-1): modulus 1
    assert props["spectral_radius"] == pytest.approx(1.0)


def test_network_properties_of_the_two_way_pair():
    props = NetworkQuantifier().extract_properties(TWO_WAY)
    assert props["density"] == pytest.approx(1.0)  # both possible edges exist
    assert props["in_degree_av"] == pytest.approx(1.0)
    assert props["spectral_radius"] == pytest.approx(0.5)  # eigenvalues +-0.5


def test_the_spectral_radius_scales_with_the_weights():
    assert NetworkQuantifier(["spectral_radius"]).extract_properties(3.0 * TWO_WAY)["spectral_radius"] == pytest.approx(1.5)


def test_selected_properties_only_and_graph_input():
    quantifier = NetworkQuantifier(["density", "spectral_radius"])
    from_matrix = quantifier.extract_properties(CYCLE_WITH_TAIL)
    from_graph = quantifier.extract_properties(nx.from_numpy_array(CYCLE_WITH_TAIL, create_using=nx.DiGraph))
    assert list(from_matrix) == ["density", "spectral_radius"]
    assert from_matrix == pytest.approx(from_graph)


def test_unknown_network_properties_are_skipped_with_a_warning():
    with pytest.warns(UserWarning, match="not recognized network properties"):
        quantifier = NetworkQuantifier(["density", "diameter"])
    assert list(quantifier.extract_properties(TWO_WAY)) == ["density"]


def test_agrees_with_the_graph_analyzer_of_the_pruning():
    """The analyzer that the pruners use and this module compute the same numbers."""
    mine = NetworkQuantifier(["density", "spectral_radius", "clustering_coefficient"]).extract_properties(CYCLE_WITH_TAIL)
    theirs = GraphAnalyzer(["density", "spectral_radius", "clustering_coefficient"]).extract_properties(CYCLE_WITH_TAIL)
    assert mine == pytest.approx(theirs)


# ------------------------------------------------------------------- node properties


def test_node_degrees_of_the_cycle_with_tail():
    props = NodePropExtractor().extract_properties(CYCLE_WITH_TAIL)
    assert props["in_degree"].tolist() == [1, 1, 1, 1]
    assert props["out_degree"].tolist() == [1, 1, 2, 0]
    assert props["degree"].tolist() == [2, 2, 3, 1]  # in-degree + out-degree


def test_betweenness_of_the_cycle_with_tail():
    """
    The shortest paths that pass through a node, over (N - 1)(N - 2) = 6:
    node 0 lies on 2 -> 0 -> 1; node 1 on 0 -> 1 -> 2 and 0 -> 1 -> 2 -> 3; node 2 on 0 -> ... -> 3,
    1 -> 2 -> 0 and 1 -> 2 -> 3; node 3 is the end of every path through it.
    """
    betweenness = NodePropExtractor(["betweenness_centrality"]).extract_properties(CYCLE_WITH_TAIL)["betweenness_centrality"]
    assert betweenness == pytest.approx([1 / 6, 2 / 6, 3 / 6, 0.0])


def test_clustering_of_the_cycle_with_tail():
    """
    The triangle 0, 1, 2 (directed cycle). Node 0 and 1 have two neighbors that are connected: 1 of 2 possible
    ordered pairs; node 2 has three neighbors (0, 1, 3) and the one connected pair (0, 1) of 6; node 3 has one.
    """
    clustering = NodePropExtractor(["clustering_coefficient"]).extract_properties(CYCLE_WITH_TAIL)["clustering_coefficient"]
    assert clustering == pytest.approx([1 / 2, 1 / 2, 1 / 6, 0.0])
    # the network average of NetworkQuantifier is their mean
    average = NetworkQuantifier(["clustering_coefficient"]).extract_properties(CYCLE_WITH_TAIL)["clustering_coefficient"]
    assert average == pytest.approx(clustering.mean())


def test_pagerank_is_a_distribution_and_follows_the_structure():
    pagerank = NodePropExtractor(["pagerank"]).extract_properties(CYCLE_WITH_TAIL)["pagerank"]
    assert pagerank.sum() == pytest.approx(1.0)
    assert np.all(pagerank > 0)
    # node 0 and node 3 both receive exactly half of the rank of node 2 (it has two out-edges) and nothing else
    assert pagerank[0] == pytest.approx(pagerank[3])
    # the structure counts, not the weights: negating all weights changes nothing
    assert NodePropExtractor(["pagerank"]).extract_properties(-CYCLE_WITH_TAIL)["pagerank"] == pytest.approx(pagerank)


def test_two_way_pair_is_symmetric():
    props = NodePropExtractor().extract_properties(TWO_WAY)
    for name, values in props.items():
        assert values.shape == (2,), name
        assert values[0] == pytest.approx(values[1]), name
    assert props["pagerank"] == pytest.approx([0.5, 0.5])
    assert props["in_degree"].tolist() == [1, 1]


def test_node_properties_have_one_value_per_node_in_index_order():
    props = NodePropExtractor().extract_properties(CYCLE_WITH_TAIL)
    assert list(props) == ["degree", "in_degree", "out_degree", "clustering_coefficient", "betweenness_centrality", "pagerank"]
    assert all(values.shape == (4,) for values in props.values())


def test_agrees_with_the_node_analyzer_of_the_pruning():
    mine = NodePropExtractor().extract_properties(CYCLE_WITH_TAIL)
    for node in range(4):
        theirs = NodeAnalyzer().extract_properties(CYCLE_WITH_TAIL, node=node)
        assert {name: values[node] for name, values in mine.items()} == pytest.approx(theirs)


def test_unknown_node_properties_are_skipped_with_a_warning():
    with pytest.warns(UserWarning, match="not recognized node properties"):
        extractor = NodePropExtractor(["degree", "eccentricity"])
    assert list(extractor.extract_properties(TWO_WAY)) == ["degree"]


# ------------------------------------------------------------------- matrix fast paths


@pytest.mark.parametrize("n", [2, 3, 8, 30])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
def test_matrix_and_graph_input_give_the_same_properties(n, p):
    """The network properties of a matrix are computed without networkx; the numbers are the ones of the DiGraph."""
    from pyreco.utils_networks import (
        extract_av_in_degree, extract_av_out_degree, extract_clustering_coefficient,
        extract_density, extract_spectral_radius,
    )

    rng = np.random.default_rng(n)
    for _ in range(5):
        A = (rng.random((n, n)) < p) * rng.normal(size=(n, n))  # includes self-loops and signed weights
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        for fun in (extract_density, extract_spectral_radius, extract_av_in_degree, extract_av_out_degree,
                    extract_clustering_coefficient):
            assert fun(A) == pytest.approx(fun(G), abs=1e-10), fun.__name__
