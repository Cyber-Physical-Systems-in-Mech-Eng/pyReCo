import pytest
import numpy as np
import networkx as nx
from pyreco.edge_selector import EdgeSelector


# Fixtures, to reuse for testing

# Directed networkx graph
@pytest.fixture
def nx_graph():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    return G

# Graph in form of numpy array
@pytest.fixture
def np_graph():
    return np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]])


# EdgeSelector object wirh networkx graph
@pytest.fixture
def selector_nx(nx_graph):
    return EdgeSelector(graph=nx_graph)


# EdgeSelector object wirh numpy array graph
@pytest.fixture
def selector_np(np_graph):
    return EdgeSelector(graph=np_graph)


# Testing __init__ function

class TestInit:

    def test_valid_nx_graph(self, nx_graph):
        # Valid directed nx graph is accepted and its edges are counted correctly
        sel = EdgeSelector(graph=nx_graph)
        assert sel.num_total_edges == 4

    def test_valid_np_array(self, np_graph):
        # Valid numpy adjacency matrix is accepted and its edges are counted correctly
        sel = EdgeSelector(graph=np_graph)
        assert sel.num_total_edges == 4

    def test_graph_none_raises(self):
        # graph=None must raise ValueError
        with pytest.raises(ValueError):
            EdgeSelector(graph=None)

    def test_graph_invalid_type_raises(self):
        # Unsupported graph type (e.g. a string) must raise TypeError
        with pytest.raises(TypeError):
            EdgeSelector(graph="not a graph")

    def test_unknown_strategy_raises(self, nx_graph):
        # Unrecognized strategy name must raise NotImplementedError
        with pytest.raises(NotImplementedError):
            EdgeSelector(graph=nx_graph, strategy="nonexistent")

    def test_default_strategy(self, selector_nx):
        # Default strategy resolves to the bound _random_uniform_wo_repl method
        assert selector_nx.strategy == selector_nx._random_uniform_wo_repl

    def test_initial_selected_edges_empty(self, selector_nx):
        # selected_edges starts out empty, before select_edges() is ever called
        assert selector_nx.selected_edges == []

    def test_initial_num_select_edges_none(self, selector_nx):
        # num_select_edges starts out as None, before select_edges() is ever called
        assert selector_nx.num_select_edges is None

    def test_initial_fraction_none(self, selector_nx):
        # fraction starts out as None, before select_edges() is ever called
        assert selector_nx.fraction is None

    def test_directed_attribute_stored(self, nx_graph):
        # directed flag passed in is stored as-is on the instance
        sel = EdgeSelector(graph=nx_graph, directed=True)
        assert sel.directed is True

    def test_graph_attribute_stored(self, nx_graph):
        # graph object passed in is stored on the instance (same reference)
        sel = EdgeSelector(graph=nx_graph)
        assert sel.graph is nx_graph


# Testing _extract_edges function

class TestExtractEdges:

    def test_nx_graph_edges(self, nx_graph):
        # All edges of an nx graph are extracted correctly
        sel = EdgeSelector(graph=nx_graph)
        assert set(sel.edge_indices) == {(0, 1), (1, 2), (2, 3), (3, 4)}

    def test_np_array_edges(self, np_graph):
        # Undirected numpy array -< only upper-triangle entries are extracted as edges
        sel = EdgeSelector(graph=np_graph, directed=False)
        assert (0, 1) in sel.edge_indices
        assert (1, 2) in sel.edge_indices
        assert (1, 0) not in sel.edge_indices  # Upper triangle only

    def test_empty_nx_graph(self):
        # A graph with no edges yields an empty edge list
        G = nx.DiGraph()
        sel = EdgeSelector(graph=G)  # no edges added
        assert sel.edge_indices == []
        assert sel.num_total_edges == 0

    def test_empty_np_array(self):
        # An all-zero adjacency matrix yields an empty edge list
        arr = np.zeros((3, 3))
        sel = EdgeSelector(graph=arr)
        assert sel.edge_indices == []
        assert sel.num_total_edges == 0

    def test_np_array_directed_keeps_both_directions(self):
        # (0, 1) and (1, 0) are both present (reciprocal pair), (1, 2) is one-way
        # Matrix is asymmetric overall, so directed=True is the correct reading
        # Verifies directed=True keeps every nonzero entry, not just the
        #   upper triangle -> it does NOT apply the undirected filter
        mat = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 0, 0]])
        sel = EdgeSelector(graph=mat, directed=True)
        assert (0, 1) in sel.edge_indices
        assert (1, 0) in sel.edge_indices
        assert (1, 2) in sel.edge_indices
        assert sel.num_total_edges == 3

    def test_np_array_undirected_drops_reverse_direction(self):
        # Same matrix as above, but undirected -> only the upper-triangle
        #   entry of the (0, 1)/(1, 0) reciprocal pair should survive
        mat = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 0, 0]])
        with pytest.warns(UserWarning):
            sel = EdgeSelector(graph=mat, directed=False)
        assert (0, 1) in sel.edge_indices
        assert (1, 0) not in sel.edge_indices
        assert (1, 2) in sel.edge_indices


# Testing _validate_graph function

class TestValidateGraph:

    def test_none_raises_value_error(self):
        # graph=None must raise ValueError
        with pytest.raises(ValueError):
            EdgeSelector(graph=None)

    def test_string_raises_type_error(self):
        # String is not a valid graph type -> TypeError
        with pytest.raises(TypeError):
            EdgeSelector(graph="not a graph")

    def test_list_raises_type_error(self):
        # List of tuples is not a valid graph type -> TypeError
        with pytest.raises(TypeError):
            EdgeSelector(graph=[(0, 1), (1, 2)])

    def test_valid_nx_does_not_raise(self, nx_graph):
        # Valid nx graph passes validation without raising
        EdgeSelector(graph=nx_graph)  

    def test_valid_np_does_not_raise(self, np_graph):
        # Valid numpy array passes validation without raising
        EdgeSelector(graph=np_graph)  

    def test_directed_true_with_undirected_nx_raises(self):
        # directed=True with an undirected nx.Graph is a flag mismatch
        G = nx.Graph()
        G.add_edges_from([(0, 1)])
        with pytest.raises(ValueError):
            EdgeSelector(graph=G, directed=True)

    def test_directed_false_with_digraph_raises(self):
        # directed=False with a directed nx.DiGraph is a flag mismatch
        G = nx.DiGraph()
        G.add_edges_from([(0, 1)])
        with pytest.raises(ValueError):
            EdgeSelector(graph=G, directed=False)

    def test_symmetric_directed_np_warns(self):
        # directed=True with a symmetric matrix is suspicious (looks undirected)
        arr = np.array([[0, 1], [1, 0]])
        with pytest.warns(UserWarning):
            EdgeSelector(graph=arr, directed=True)

    def test_asymmetric_undirected_np_warns(self):
        # directed=False with an asymmetric matrix is suspicious (looks directed)
        arr = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])
        with pytest.warns(UserWarning):
            EdgeSelector(graph=arr, directed=False)


# Tetsing _validate_strategy function

class TestValidateStrategy:

    def test_unknown_strategy_raises(self, nx_graph):
        # Unrecognized strategy name must raise NotImplementedError
        with pytest.raises(NotImplementedError):
            EdgeSelector(graph=nx_graph, strategy="unknown")

    def test_valid_strategy_does_not_raise(self, nx_graph):
        # Recognized strategy name passes validation without raising
        EdgeSelector(graph=nx_graph, strategy="random_uniform_wo_repl")


# Testing select_edges function

class TestSelectEdges:

    def test_select_by_num(self, selector_nx):
        # Selecting by an exact count returns that many edges
        result = selector_nx.select_edges(num=2)
        assert len(result) == 2

    def test_select_by_fraction(self, selector_nx):
        # Selecting by fraction returns the expected proportion of edges
        result = selector_nx.select_edges(fraction=0.5)
        assert len(result) == 2

    def test_select_all_edges(self, selector_nx):
        # num equal to the total edge count selects every edge
        result = selector_nx.select_edges(num=4)
        assert len(result) == 4

    def test_result_is_subset_of_edges(self, selector_nx):
        # Selected edges are always drawn from edge_indices
        result = selector_nx.select_edges(num=2)
        assert all(edge in selector_nx.edge_indices for edge in result)

    def test_no_duplicates(self, selector_nx):
        # Sampling without replacement never returns the same edge twice
        result = selector_nx.select_edges(num=3)
        assert len(result) == len(set(result))

    def test_stores_result_on_instance(self, selector_nx):
        # Returned selection is also stored as self.selected_edges
        result = selector_nx.select_edges(num=2)
        assert selector_nx.selected_edges == result

    def test_updates_num_select_edges(self, selector_nx):
        # Selecting by num updates the num_select_edges attribute to match
        selector_nx.select_edges(num=2)
        assert selector_nx.num_select_edges == 2

    def test_fraction_attribute_set_when_selecting_by_num(self, selector_nx):
        # Selecting by num should also back-fill the fraction attribute
        # nx_graph fixture has 4 total edges
        selector_nx.select_edges(num=2)
        assert selector_nx.fraction == pytest.approx(0.5)

    def test_num_select_edges_attribute_set_when_selecting_by_fraction(self, selector_nx):
        # Selecting by fraction should also back-fill num_select_edges
        selector_nx.select_edges(fraction=0.5)
        assert selector_nx.num_select_edges == 2

    def test_fraction_one_selects_all_edges(self, selector_nx):
        # fraction=1.0 (the inclusive upper bound) selects every edge
        result = selector_nx.select_edges(fraction=1.0)
        assert len(result) == selector_nx.num_total_edges

    def test_tiny_fraction_still_selects_at_least_one_edge(self):
        # 5 nodes, fully connected directed graph -> 20 edges
        # round(20 * 0.01) = 0, without the max(1, ...) floor  would
        #   propose zero candidates
        G = nx.DiGraph()
        nodes = range(5)
        G.add_edges_from([(i, j) for i in nodes for j in nodes if i != j])
        sel = EdgeSelector(graph=G)
        result = sel.select_edges(fraction=0.01)
        assert len(result) == 1
        assert sel.num_select_edges == 1

    def test_neither_arg_raises(self, selector_nx):
        # Calling select_edges() with neither fraction nor num must raise value error
        with pytest.raises(ValueError):
            selector_nx.select_edges()

    def test_both_args_raises(self, selector_nx):
        # Calling select_edges() with both fraction and num must raise value error
        with pytest.raises(ValueError):
            selector_nx.select_edges(fraction=0.5, num=2)

    def test_num_zero_raises(self, selector_nx):
        # num=0 is outside the valid [1, num_total_edges] range
        with pytest.raises(ValueError):
            selector_nx.select_edges(num=0)

    def test_num_too_large_raises(self, selector_nx):
        # num greater than the total edge count is out of range
        with pytest.raises(ValueError):
            selector_nx.select_edges(num=999)

    def test_fraction_zero_raises(self, selector_nx):
        # fraction=0.0 is outside the valid (0, 1] range
        with pytest.raises(ValueError):
            selector_nx.select_edges(fraction=0.0)

    def test_fraction_above_one_raises(self, selector_nx):
        # fraction > 1.0 is outside the valid (0, 1] range
        with pytest.raises(ValueError):
            selector_nx.select_edges(fraction=1.5)

    def test_num_not_integer_raises(self, selector_nx):
        # num must be an int, not a float
        with pytest.raises(TypeError):
            selector_nx.select_edges(num=2.5)

    def test_fraction_not_float_raises(self, selector_nx):
        # fraction must be a float, not an int
        with pytest.raises(TypeError):
            selector_nx.select_edges(fraction=1)

    def test_works_with_np_graph(self, selector_np):
        # select_edges() also works when the graph is a numpy array
        result = selector_np.select_edges(num=2)
        assert len(result) == 2
        assert all(edge in selector_np.edge_indices for edge in result)
