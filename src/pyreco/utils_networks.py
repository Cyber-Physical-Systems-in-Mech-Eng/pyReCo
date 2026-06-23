"""
Helper routines for networks
"""

from typing import Union
import networkx as nx
import numpy as np
import warnings


def gen_ER_graph(
    nodes: int, density: float, spec_rad: float = 0.9, directed: bool = True, seed=None
):
    """Generate an Erdős-Rényi random graph with specified properties.

    Bug Fix Documentation:
    ---------------------
    Previous Bug:
        The line `G.remove_nodes_from(list(nx.isolates(G)))` removed isolated nodes from the graph
        before converting to a numpy array. This caused the final matrix to sometimes be smaller
        than the specified size (e.g., 29x29 instead of 30x30) when isolated nodes were present.
        This led to dimension mismatches in reservoir computations where other matrices expected
        the full size.

    Solution:
        Instead of removing isolated nodes, we now connect them to maintain the specified
        network size. This ensures consistency between the reservoir weight matrix and
        other matrices in the computation.

    Parameters:
        nodes (int): Number of nodes in the graph
        density (float): Desired connection density (0 to 1)
        spec_rad (float): Desired spectral radius
        directed (bool): Whether to create a directed graph
        seed (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Adjacency matrix with shape (nodes, nodes)
    """
    # use networkx to generate a random graph
    G = nx.erdos_renyi_graph(nodes, density, seed=seed, directed=directed)

    # Instead of removing isolated nodes (old buggy behavior):
    # G.remove_nodes_from(list(nx.isolates(G)))

    # New: Connect isolated nodes to maintain matrix size
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        non_isolated = list(set(G.nodes()) - set(isolated_nodes))
        for node in isolated_nodes:
            if non_isolated:
                target = np.random.choice(non_isolated)
                G.add_edge(node, target)
                if directed:
                    G.add_edge(target, node)

    GNet = nx.to_numpy_array(G)
    curr_spec_rad = max(abs(np.linalg.eigvals(GNet)))
    graph = GNet * spec_rad / curr_spec_rad

    return graph


def compute_density(network: np.ndarray) -> float:
    if not isinstance(network, np.ndarray):
        raise TypeError("adjacency matrix must be numpy.ndarray")
    if network.shape[0] != network.shape[1]:
        raise ValueError("adjacency matrix must be square")
    N = len(network)
    num_links = np.sum(network.flatten() > 0)
    return num_links / (N**2)


def get_num_nodes(network: np.ndarray) -> int:
    if not isinstance(network, np.ndarray):
        raise TypeError("adjacency matrix must be numpy.ndarray")
    if network.shape[0] != network.shape[1]:
        raise ValueError("adjacency matrix must be square")
    return network.shape[0]


def compute_spec_rad(network: np.ndarray) -> float:
    if not isinstance(network, np.ndarray):
        raise TypeError("adjacency matrix must be numpy.ndarray")
    if network.shape[0] != network.shape[1]:
        raise ValueError("adjacency matrix must be square")
    return np.max(np.abs(np.linalg.eigvals(network)))

def set_spec_rad(network: np.ndarray, spec_radius: float) -> np.ndarray:
    if not isinstance(network, np.ndarray):
        raise TypeError("adjacency matrix must be numpy.ndarray")
    if network.shape[0] != network.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if np.count_nonzero(network) == 0:
        raise ValueError("adjacency matrix must have at least one link")
    if spec_radius <= 0:
        raise ValueError("spectral radius must be larger than zero")
    if spec_radius > 1.0:
        warnings.warn("a spectral radius larger than 1 is unusual!", Warning)
    current_spectral_radius = compute_spec_rad(network)
    if current_spectral_radius < 10 ** (-9):
        print("spectral radius smaller than 10^-9!")
        current_spectral_radius = 10 ** (-6)
    scaling_factor = spec_radius / current_spectral_radius
    return network * scaling_factor


def is_zero_col_and_row(x: np.ndarray, idx: int) -> bool:
    # returns zero if adjacency matrix x carries only zeros in column and row of index idx (i.e. missing node)

    is_zero_column = np.all(x[:, idx] == 0)
    is_zero_row = np.all(x[idx, :] == 0)

    if is_zero_column and is_zero_row:
        return True
    else:
        return False


def remove_nodes_from_graph(graph: np.ndarray, nodes: list):
    # removes a node from the graph given as np adjacency matrix

    # TODO: allow the user to input a networkx graph instead of a numpy array

    if not isinstance(nodes, list):
        raise TypeError("Nodes must be provided as a list of indices.")

    num_nodes = get_num_nodes(graph)
    if np.max(nodes) > num_nodes:
        raise ValueError("Node index exceeds the number of nodes in the graph.")

    if np.min(nodes) < 0:
        raise ValueError("Node index must be positive.")

    if not isinstance(graph, np.ndarray):
        raise TypeError("Adjacency matrix must be numpy.ndarray")

    if not all(isinstance(x, int) for x in nodes):
        raise ValueError("All entries in the node list must be integers.")

    # remove nodes from the network using networkx

    # 1. create a graph from the weights matrix
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)

    # 2. remove the specified nodes from the graph
    G.remove_nodes_from(nodes)

    # 3. convert the graph back to a NumPy array
    graph_trunc = nx.to_numpy_array(G)

    if graph_trunc.shape[0] != graph.shape[0] - len(nodes):
        raise ValueError("sth wrong!")

    return graph_trunc


def rename_nodes_after_removal(original_nodes: list, removed_nodes: list):
    # removes the nodes from the original list of nodes and renames the remaining nodes

    # removing node 5 from a 6-node graph with original_nodes=[1,2]
    # -> wrongly produced [0,1] instead of [1,2].
    # old_to_new = {}
    # new_index = 0
    # for old_index in np.unique(original_nodes):
    #     if old_index not in removed_nodes:
    #         old_to_new[old_index] = new_index
    #         new_index += 1
    # updated_nodes = [
    #     old_to_new[node] for node in original_nodes if node not in removed_nodes
    # ]

    # Surviving node's new index = old index minus the number of removed nodes that had
    #   smaller index
    removed_set = set(removed_nodes)
    updated_nodes = [
        node - sum(1 for r in removed_nodes if r < node)
        for node in original_nodes
        if node not in removed_set
    ]

    return updated_nodes


def gen_init_states(num_nodes: int, method: str = "random"):
    # returns an array of length <num_nodes>
    # creates the entries based on different sampling methods
    # when not setting specific values, the range is normalized to abs(1)

    if method == "random":
        init_states = np.random.random(num_nodes)
    elif method == "random_normal":
        init_states = np.random.randn(num_nodes)
    elif method == "ones":
        init_states = np.ones(num_nodes)
    elif method == "zeros":
        init_states = np.zeros(num_nodes)
    else:
        raise (
            ValueError(
                f"Sampling method {method} is unknown for generating initial reservoir states"
            )
        )

    # normalize to max. absolute value of 1
    if method != "zeros":
        init_states = init_states / np.max(np.abs(init_states))

    return init_states


def convert_to_nx_graph(graph: np.ndarray) -> nx.Graph:
    """
    Convert a numpy array to a NetworkX graph.
    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
    Returns:
        nx.Graph: The NetworkX graph.
    """
    if isinstance(graph, np.ndarray):
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    elif not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("adjacency_matrix must be a numpy array or a NetworkX graph.")
    return graph


def extract_density(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    """
    Extract the density of a graph from its adjacency matrix.
    Args:
        adjacency_matrix (np.ndarray, nx.Graph, nx.DiGraph): The adjacency matrix of the graph.
    Returns:
        float: The density of the graph.
    """
    graph = convert_to_nx_graph(graph)

    return nx.density(graph)


def extract_spectral_radius(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:

    graph = convert_to_nx_graph(graph)

    # Get the adjacency matrix of the graph
    adjacency_matrix = nx.to_numpy_array(graph)

    # Compute the eigenvalues of the adjacency matrix
    eigenvalues = np.linalg.eigvals(adjacency_matrix)

    # Compute the spectral radius (maximum absolute eigenvalue)
    spectral_radius = np.max(np.abs(eigenvalues))

    return spectral_radius


def extract_av_in_degree(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    graph = convert_to_nx_graph(graph)
    in_degrees = graph.in_degree()
    avg_in_degree = np.mean(list(dict(in_degrees).values()))
    return avg_in_degree


def extract_av_out_degree(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    graph = convert_to_nx_graph(graph)
    out_degrees = graph.out_degree()
    avg_out_degree = np.mean(list(dict(out_degrees).values()))
    return avg_out_degree


def extract_clustering_coefficient(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph]
) -> float:
    graph = convert_to_nx_graph(graph)
    return nx.average_clustering(graph)


def extract_node_degree(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    graph = convert_to_nx_graph(graph)
    return graph.degree[node]


def extract_node_in_degree(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    graph = convert_to_nx_graph(graph)
    return graph.in_degree[node]


def extract_node_out_degree(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    graph = convert_to_nx_graph(graph)
    return graph.out_degree[node]


def extract_node_clustering_coefficient(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    graph = convert_to_nx_graph(graph)

    return nx.clustering(graph, node)


def extract_node_betweenness_centrality(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    graph = convert_to_nx_graph(graph)
    return nx.betweenness_centrality(graph)[node]


def extract_node_pagerank(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    graph = convert_to_nx_graph(graph)
    return nx.pagerank(graph)[node]


def extract_edge_weight(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                        edge: tuple) -> float:
    """
    Extracts the weight of an edge.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph containing the edge.
    edge : tuple
        Edge (u, v) to extract the weight for.

    Returns
    -------
    float
        Edge weight, or 1.0 if edge has no explicit "weight"
        attribute.
    """
    graph = convert_to_nx_graph(graph)
    u, v = edge
    return graph[u][v].get("weight", 1.0)


def extract_edge_is_reciprocal(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                               edge: tuple) -> int:
    """
    Checks whether an edge is reciprocated.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph containing the edge.
    edge : tuple
        Edge (u, v) to check.

    Returns
    -------
    int
        1 if reverse edge (v, u) also exists in graph, 0 otherwise.
    """
    graph = convert_to_nx_graph(graph)
    u, v = edge
    return int(graph.has_edge(v, u))


def extract_edge_in_scc(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                        edge: tuple) -> int:
    """
    Check whether an edge's endpoints lie in the same strongly connected component.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        The graph containing the edge.
    edge : tuple
        The (u, v) edge to check.

    Returns
    -------
    int
        1 if both endpoints belong to the same strongly connected component,
        0 otherwise.
    """
    # Strongly connected component (SCC) is a set of nodes in a directed
    #   graph that are mutually reachable from one another
    graph = convert_to_nx_graph(graph)
    # u, v essentially are the nodes we want to look for
    u, v = edge
    # nx.strongly_connected_components(graph) returns set of nodes in same strongly
    #   connected component
    # Create dict mapping node to a SCC ID (making tracebable what SCC it belongs to)
    node_to_scc = {
        node: comp_id
        for comp_id, comp in enumerate(nx.strongly_connected_components(graph))
        for node in comp
    }
    # Compare if nodes are in the same SCC
    return int(node_to_scc[u] == node_to_scc[v])


def extract_edge_betweenness(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                             edge: tuple) -> float:
    """
    Extracts the edge betweenness centrality of an edge.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph containing the edge.
    edge : tuple
        Edge (u, v) to extract betweenness centrality for.

    Returns
    -------
    float
        Edge's betweenness centrality, or 0.0 if edge is not found
        (e.g. due to a direction mismatch).
    """
    graph = convert_to_nx_graph(graph)
    # nx.edge_betweenness_centrality(graph) returns dict with values for all graph edges
    return nx.edge_betweenness_centrality(graph).get(edge, 0.0)


def extract_edge_source_out_degree(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                                   edge: tuple) -> int:
    """
    Extracts the out-degree of an edge's source node.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph containing the edge.
    edge : tuple
        Edge (u, v) whose source node's out-degree is extracted.

    Returns
    -------
    int
        Out-degree of node u.
    """
    graph = convert_to_nx_graph(graph)
    u, _ = edge
    return graph.out_degree[u]


def extract_edge_target_in_degree(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                                  edge: tuple) -> int:
    """
    Extracts the in-degree of an edge's target node.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph containing the edge.
    edge : tuple
        Edge(u, v) whose target node's in-degree is extracted.

    Returns
    -------
    int
        In-degree of node v.
    """
    graph = convert_to_nx_graph(graph)
    _, v = edge
    return graph.in_degree[v]


def extract_edge_source_betweenness(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                                    edge: tuple) -> float:
    """
    Extracts the node betweenness centrality of an edge's source node.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph containing the edge.
    edge : tuple
        Edge (u, v) whose source node's betweenness centrality is
        extracted.

    Returns
    -------
    float
        Betweenness centrality of node u.
    """
    graph = convert_to_nx_graph(graph)
    u, _ = edge
    return nx.betweenness_centrality(graph)[u]


def extract_edge_target_betweenness(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                                    edge: tuple) -> float:
    """
    Extracts the node betweenness centrality of an edge's target node.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph containing the edge.
    edge : tuple
        Edge (u, v) whose target node's betweenness centrality is
        extracted.

    Returns
    -------
    float
        Betweenness centrality of node v.
    """
    graph = convert_to_nx_graph(graph)
    _, v = edge
    return nx.betweenness_centrality(graph)[v]


def precompute_edge_metrics(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> dict:
    """
    Precomputes expensive graph-level metrics needed for edge property extraction.

    Calls once per graph state and passes the result to
    'EdgeAnalyzer.extract_properties_batch' to avoid recomputing
    betweenness centrality and SCC membership for every candidate edge.

    Parameters
    ----------
    graph : np.ndarray, nx.Graph, or nx.DiGraph
        Graph to compute metrics for.

    Returns
    -------
    dict
        Dictionary with keys:

        - 'graph' : nx.DiGraph
        - 'node_betweenness' : dict mapping node -> centrality
        - 'edge_betweenness' : dict mapping (u, v) -> centrality
        - 'scc_map' : dict mapping node -> strongly connected component id
    """
    g = convert_to_nx_graph(graph)
    node_betweenness = nx.betweenness_centrality(g)
    edge_betweenness = nx.edge_betweenness_centrality(g)
    scc_map = {
        node: comp_id
        for comp_id, comp in enumerate(nx.strongly_connected_components(g))
        for node in comp
    }
    return {
        "graph": g,
        "node_betweenness": node_betweenness,
        "edge_betweenness": edge_betweenness,
        "scc_map": scc_map,
    }


# Add more network property extraction functions as needed


if __name__ == "__main__":

    # test the node renaming function
    original_nodes = [0, 2, 1, 3, 4, 5, 6, 8, 7]
    removed_nodes = [0, 5, 6]

    updated_nodes = rename_nodes_after_removal(original_nodes, removed_nodes)
    print(updated_nodes)
