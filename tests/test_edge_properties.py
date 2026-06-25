import pytest
import numpy as np
import networkx as nx

from pyreco.utils_networks import (
    convert_to_nx_graph,
    extract_edge_weight,
    extract_edge_is_reciprocal,
    extract_edge_in_scc,
    extract_edge_betweenness,
    extract_edge_source_out_degree,
    extract_edge_target_in_degree,
    extract_edge_source_betweenness,
    extract_edge_target_betweenness,
    precompute_edge_metrics,
)
from pyreco.edge_analyzer import (
    EdgeAnalyzer,
    available_extractors,
    map_extractor_names,
)


# Fixtures, to reuse for testing

# Directed graph
# Cycle 0->1->2->0 (one strongly connected component)
# Reciprocal edge 1->0
# Bridge 2->3 into downstream node with no outgoing edges (its own SCC)
@pytest.fixture
def digraph():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=2.5)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    G.add_edge(1, 0)
    G.add_edge(2, 3)
    return G


# Same graph as above in numpy array form
@pytest.fixture
def adjacency_matrix():
    mat = np.zeros((4, 4))
    mat[0, 1] = 1
    mat[1, 2] = 1
    mat[2, 0] = 1
    mat[1, 0] = 1
    mat[2, 3] = 1
    return mat


# Testing convert_to_nx_graph function

class TestConvertToNxGraph:

    def test_numpy_array_converted_to_digraph(self, adjacency_matrix):
        # Converted numpy adjacency matrix into an equivalent nx.DiGraph
        g = convert_to_nx_graph(adjacency_matrix)
        assert isinstance(g, nx.DiGraph)
        assert set(g.edges()) == {(0, 1), (1, 2), (2, 0), (1, 0), (2, 3)}

    def test_nx_graph_passthrough(self, digraph):
        # nx graph input is returned unchanged (same object, no copy)
        g = convert_to_nx_graph(digraph)
        assert g is digraph

    def test_invalid_type_raises(self):
        # Neither numpy array nor nx graph must raise value error
        with pytest.raises(ValueError, match="numpy array or a NetworkX graph"):
            convert_to_nx_graph([(0, 1), (1, 2)])


# Testing extract_edge_weight function

class TestExtractEdgeWeight:

    def test_explicit_weight(self, digraph):
        # Edge with weight attribute returns that value
        assert extract_edge_weight(digraph, (0, 1)) == pytest.approx(2.5)

    def test_default_weight(self, digraph):
        # Edge with no weight attribute defaults to 1.0
        assert extract_edge_weight(digraph, (1, 2)) == pytest.approx(1.0)

    def test_works_with_numpy_array(self, adjacency_matrix):
        # Same test for numpy adjacency matrix
        assert extract_edge_weight(adjacency_matrix, (0, 1)) == pytest.approx(1.0)


# Test extract_edge_is_reciprocal function

class TestExtractEdgeIsReciprocal:

    def test_reciprocal_edge(self, digraph):
        # Edges that exist in both directions are flagged as reciprocal
        assert extract_edge_is_reciprocal(digraph, (0, 1)) == 1
        assert extract_edge_is_reciprocal(digraph, (1, 0)) == 1

    def test_non_reciprocal_edge(self, digraph):
        # Edges with no reverse counterpart are not reciprocal
        assert extract_edge_is_reciprocal(digraph, (1, 2)) == 0
        assert extract_edge_is_reciprocal(digraph, (2, 3)) == 0


# Tes extract_edge_in_scc function

class TestExtractEdgeInScc:

    def test_edge_within_scc(self, digraph):
        # Edges whose endpoints are mutually reachable (0-1-2 cycle) are
        #   flagged as being within same strongly connected component
        assert extract_edge_in_scc(digraph, (0, 1)) == 1
        assert extract_edge_in_scc(digraph, (1, 2)) == 1
        assert extract_edge_in_scc(digraph, (2, 0)) == 1

    def test_edge_across_sccs(self, digraph):
        # Bridge edge into node 3 not SCC part
        assert extract_edge_in_scc(digraph, (2, 3)) == 0


# Test extract_edge_betweenness function

class TestExtractEdgeBetweenness:

    def test_matches_networkx_directly(self, digraph):
        # Edge betweenness centrality matches networkx's own computation
        #   for every edge in the graph
        expected = nx.edge_betweenness_centrality(digraph)
        for edge in digraph.edges():
            assert extract_edge_betweenness(digraph, edge) == \
                pytest.approx(expected[edge])

    def test_missing_edge_defaults_to_zero(self, digraph):
        # Edge that doesn't exist defaults to 0.0 betweenness
        assert extract_edge_betweenness(digraph, (3, 0)) == pytest.approx(0.0)


# Test extract_edge_source_out_degree / extract_edge_target_in_degree function

class TestExtractEdgeDegrees:

    def test_source_out_degree(self, digraph):
        # Returns out-degree of edge's source node
        assert extract_edge_source_out_degree(digraph, (0, 1)) == digraph.out_degree(0)
        assert extract_edge_source_out_degree(digraph, (2, 3)) == digraph.out_degree(2)

    def test_target_in_degree(self, digraph):
        # Returns in-degree of edge's target node
        assert extract_edge_target_in_degree(digraph, (0, 1)) == digraph.in_degree(1)
        assert extract_edge_target_in_degree(digraph, (2, 3)) == digraph.in_degree(3)


# Test extract_edge_source_betweenness / extract_edge_target_betweenness function

class TestExtractEdgeNodeBetweenness:

    def test_source_and_target_betweenness_match_networkx(self, digraph):
        # Source/target node betweenness centrality (not edge betweenness)
        #   matches networkx's own per-node computation for every edge
        expected = nx.betweenness_centrality(digraph)
        for u, v in digraph.edges():
            assert extract_edge_source_betweenness(digraph, (u, v)) == \
                pytest.approx(expected[u])
            assert extract_edge_target_betweenness(digraph, (u, v)) == \
                pytest.approx(expected[v])


# Test precompute_edge_metrics function

class TestPrecomputeEdgeMetrics:

    def test_returns_expected_keys(self, digraph):
        # Precomputed cache always has these four keys
        #   "graph", "node_betweenness", "edge_betweenness", "scc_map"
        cache = precompute_edge_metrics(digraph)
        assert set(cache.keys()) == {
            "graph", "node_betweenness", "edge_betweenness", "scc_map"
        }

    def test_graph_is_digraph(self, digraph):
        # Cached graph is normalized to an nx.DiGraph
        cache = precompute_edge_metrics(digraph)
        assert isinstance(cache["graph"], nx.DiGraph)

    def test_node_betweenness_matches_networkx(self, digraph):
        # Cached node betweenness matches networkx's own computation
        cache = precompute_edge_metrics(digraph)
        assert cache["node_betweenness"] == \
            pytest.approx(nx.betweenness_centrality(digraph))

    def test_edge_betweenness_matches_networkx(self, digraph):
        # Cached edge betweenness matches networkx's own computation
        cache = precompute_edge_metrics(digraph)
        expected = nx.edge_betweenness_centrality(digraph)
        assert cache["edge_betweenness"].keys() == expected.keys()
        for edge in expected:
            assert cache["edge_betweenness"][edge] == pytest.approx(expected[edge])

    def test_scc_map_groups_nodes_correctly(self, digraph):
        # Nodes in the same strongly connected component share component id.
        #   nodes in different components get different ids
        cache = precompute_edge_metrics(digraph)
        scc_map = cache["scc_map"]
        # 0, 1, 2 are mutually reachable -> same component id
        assert scc_map[0] == scc_map[1] == scc_map[2]
        # 3 cannot reach back to the cycle -> separate component
        assert scc_map[3] != scc_map[0]


# Test available_extractors / map_extractor_names function

class TestAvailableExtractors:

    def test_expected_keys(self):
        # available_extractors() advertises exactly these edge property names
        extractors = available_extractors()
        assert set(extractors.keys()) == {
            "weight", "is_reciprocal", "in_scc", "betweenness",
            "source_out_degree", "target_in_degree",
            "source_betweenness", "target_betweenness",
        }

    def test_values_are_callable(self):
        # Every entry maps to an actual callable extractor function
        for fn in available_extractors().values():
            assert callable(fn)


class TestMapExtractorNames:

    def test_known_properties(self):
        # Recognized property names are mapped to their extractor functions
        extractor_dict, extractor_funs = map_extractor_names(["weight", "betweenness"])
        assert list(extractor_dict.keys()) == ["weight", "betweenness"]
        assert len(extractor_funs) == 2

    def test_unknown_property_is_dropped(self, capsys):
        # An unrecognized property name is silently dropped (with a printed warning)
        extractor_dict, extractor_funs = map_extractor_names(["weight",
                                                              "not_a_property"])
        assert "not_a_property" not in extractor_dict
        assert len(extractor_funs) == 1
        captured = capsys.readouterr()
        assert "not_a_property" in captured.out


# Test EdgeAnalyzer.__init__ function

class TestEdgeAnalyzerInit:

    def test_default_quantities_include_all(self):
        # No properties specified -> every available extractor is used
        analyzer = EdgeAnalyzer()
        assert analyzer.quantities == list(available_extractors().keys())

    def test_custom_quantities(self):
        # Custom properties list determines which extractors get used
        analyzer = EdgeAnalyzer(quantities=["weight", "betweenness"])
        assert analyzer.quantities == ["weight", "betweenness"]
        assert set(analyzer.extractors.keys()) == {"weight", "betweenness"}


# Test EdgeAnalyzer.extract_properties functions

class TestEdgeAnalyzerExtractProperties:

    def test_returns_requested_quantities(self, digraph):
        # Only requested properties are returned, with correct values
        analyzer = EdgeAnalyzer(quantities=["weight", "is_reciprocal", "in_scc"])
        props = analyzer.extract_properties(digraph, (0, 1))
        assert set(props.keys()) == {"weight", "is_reciprocal", "in_scc"}
        assert props["weight"] == pytest.approx(2.5)
        assert props["is_reciprocal"] == 1
        assert props["in_scc"] == 1

    def test_matches_standalone_extractors(self, digraph):
        # EdgeAnalyzer's output for every property matches calling
        #   corresponding standalone utils_networks function directly
        analyzer = EdgeAnalyzer()
        edge = (2, 3)
        props = analyzer.extract_properties(digraph, edge)
        assert props["weight"] == pytest.approx(extract_edge_weight(digraph, edge))
        assert props["is_reciprocal"] == extract_edge_is_reciprocal(digraph, edge)
        assert props["in_scc"] == extract_edge_in_scc(digraph, edge)
        assert props["betweenness"] == \
            pytest.approx(extract_edge_betweenness(digraph, edge))
        assert props["source_out_degree"] == \
            extract_edge_source_out_degree(digraph, edge)
        assert props["target_in_degree"] == \
            extract_edge_target_in_degree(digraph, edge)
        assert props["source_betweenness"] == \
            pytest.approx(extract_edge_source_betweenness(digraph, edge))
        assert props["target_betweenness"] == \
            pytest.approx(extract_edge_target_betweenness(digraph, edge))

    def test_non_tuple_edge_raises(self, digraph):
        # Non-tuple edge argument (e.g. a list) must raise value error
        analyzer = EdgeAnalyzer()
        with pytest.raises(ValueError, match="edge must be a"):
            analyzer.extract_properties(digraph, [0, 1])

    def test_wrong_length_tuple_raises(self, digraph):
        # Tuple that isn't (u, v) must raise value error
        analyzer = EdgeAnalyzer()
        with pytest.raises(ValueError, match="edge must be a"):
            analyzer.extract_properties(digraph, (0, 1, 2))


# Test EdgeAnalyzer.extract_properties_batch function

class TestEdgeAnalyzerExtractPropertiesBatch:

    def test_same_order_as_input_edges(self, digraph):
        # Batch results are returned in same order as input edges
        analyzer = EdgeAnalyzer(quantities=["weight"])
        edges = [(0, 1), (1, 2), (2, 3)]
        results = analyzer.extract_properties_batch(digraph, edges)
        assert len(results) == len(edges)
        assert results[0]["weight"] == pytest.approx(2.5)
        assert results[1]["weight"] == pytest.approx(1.0)

    def test_respects_requested_quantities(self, digraph):
        # Batch extraction only returns properties analyzer was
        #   configured with
        analyzer = EdgeAnalyzer(quantities=["betweenness", "in_scc"])
        results = analyzer.extract_properties_batch(digraph, [(0, 1)])
        assert set(results[0].keys()) == {"betweenness", "in_scc"}

    def test_matches_non_batch_extraction(self, digraph):
        # Batched (cached, single graph pass) path coresponds to
        #   calling extract_properties() individually for every edge
        analyzer = EdgeAnalyzer()
        edges = list(digraph.edges())
        batch_results = analyzer.extract_properties_batch(digraph, edges)
        for edge, batch_props in zip(edges, batch_results):
            single_props = analyzer.extract_properties(digraph, edge)
            for key in single_props:
                assert batch_props[key] == pytest.approx(single_props[key])

    def test_works_with_numpy_array(self, adjacency_matrix):
        # Batch extraction also works when graph is numpy adjacency matrix
        analyzer = EdgeAnalyzer(quantities=["weight", "source_out_degree"])
        results = analyzer.extract_properties_batch(adjacency_matrix, [(0, 1), (2, 3)])
        assert len(results) == 2


# Test EdgeAnalyzer.list_properties function

class TestEdgeAnalyzerListProperties:

    def test_matches_available_extractors(self):
        # list_properties() reports available_extractors() keys
        analyzer = EdgeAnalyzer()
        assert analyzer.list_properties() == list(available_extractors().keys())
