import pytest
import networkx as nx
import numpy as np
from pyreco.node_selector import NodeSelector


def test_node_selector_initialization():
    # Test valid initialization with total_nodes
    selector = NodeSelector(strategy="random_uniform_wo_repl", total_nodes=10)
    assert selector.num_total_nodes == 10

    # Test valid initialization with graph
    G = nx.erdos_renyi_graph(10, 0.5)
    selector = NodeSelector(strategy="random_uniform_wo_repl", graph=G)
    assert selector.num_total_nodes == G.number_of_nodes()

    # Test invalid initialization with both total_nodes and graph
    with pytest.raises(ValueError):
        NodeSelector(
            strategy="random_uniform_wo_repl",
            total_nodes=10,
            graph=G,
        )

    # Test invalid initialization with neither total_nodes nor graph
    with pytest.raises(ValueError):
        NodeSelector(strategy="random_uniform_wo_repl")

    # Test invalid total_nodes type
    with pytest.raises(TypeError):
        NodeSelector(strategy="random_uniform_wo_repl", total_nodes="10")

    # Test invalid total_nodes value
    with pytest.raises(ValueError):
        NodeSelector(strategy="random_uniform_wo_repl", total_nodes=-1)

    # Test invalid graph type
    with pytest.raises(TypeError):
        NodeSelector(strategy="random_uniform_wo_repl", graph=np.ones(10, 10))

    # Test invalid strategy
    with pytest.raises(NotImplementedError):
        NodeSelector(strategy="invalid_strategy", total_nodes=10)


def test_node_selection():
    # Test node selection with fraction
    G = nx.erdos_renyi_graph(10, 0.5)
    selector = NodeSelector(strategy="random_uniform_wo_repl", graph=G)
    selected_nodes = selector.select_nodes(fraction=0.5)
    assert len(selected_nodes) == 5
    assert len(set(selected_nodes)) == 5  # Ensure no duplicates

    # Test node selection with num
    selected_nodes = selector.select_nodes(num=3)
    assert len(selected_nodes) == 3
    assert len(set(selected_nodes)) == 3  # Ensure no duplicates

    # Test invalid fraction type
    with pytest.raises(TypeError):
        selector.select_nodes(fraction="0.5")

    # Test invalid fraction value
    with pytest.raises(ValueError):
        selector.select_nodes(fraction=1.5)

    # Test invalid num type
    with pytest.raises(TypeError):
        selector.select_nodes(num="3")

    # Test invalid num value
    with pytest.raises(ValueError):
        selector.select_nodes(num=-1)


if __name__ == "__main__":
    pytest.main()
