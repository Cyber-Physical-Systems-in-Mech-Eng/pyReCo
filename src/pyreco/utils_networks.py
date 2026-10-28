r"""
Helper routines for the reservoir networks.

Networks are weighted adjacency matrices :math:`A \in \mathbb{R}^{N \times N}` in which :math:`A_{uv}` is
the weight of the directed edge :math:`u \to v` (a zero entry is no edge). The module generates random
networks, measures them (density, spectral radius, degrees, centralities) and edits them (removal of nodes).
"""

from typing import Union
import networkx as nx
import numpy as np
import warnings

from pyreco.utils_random import RngLike, make_rng


# number of redraws of a graph without any directed cycle before giving up
MAX_ER_ATTEMPTS = 100

# a drawn graph whose spectral radius is below this fraction of its largest weight has no
# usable cycle structure (see gen_ER_graph)
MIN_SPEC_RAD = 1e-3

# distributions of the weights on the existing edges of a random network
WEIGHT_DISTRIBUTIONS = ("positive", "sign", "uniform", "normal")


def draw_edge_weights(mask: np.ndarray, distribution: str, rng: np.random.Generator):
    r"""
    Weights :math:`w_{uv}` for the edges of a binary adjacency matrix.

    Parameters
    ----------
    mask : ndarray of shape (N, N)
        Non-zero (True) where an edge exists.
    distribution : {"positive", "sign", "uniform", "normal"}
        ``"positive"``: all edges 1 (no random numbers are drawn); ``"sign"``: :math:`\pm 1` with
        equal probability; ``"uniform"``: :math:`\mathcal{U}(-1, 1)`; ``"normal"``: :math:`\mathcal{N}(0, 1)`.
    rng : numpy.random.Generator
        Source of the random numbers.

    Returns
    -------
    ndarray of shape (N, N)
        The weights on the edges, 0 where there is no edge.

    Raises
    ------
    ValueError
        For an unknown distribution.
    """
    if distribution == "positive":
        return mask.astype(float)
    if distribution == "sign":
        weights = rng.choice([-1.0, 1.0], size=mask.shape)
    elif distribution == "uniform":
        weights = rng.uniform(-1.0, 1.0, size=mask.shape)
    elif distribution == "normal":
        weights = rng.standard_normal(size=mask.shape)
    else:
        raise ValueError(
            f"unknown weight_distribution {distribution!r}, use one of {WEIGHT_DISTRIBUTIONS}."
        )
    return np.where(mask, weights, 0.0)


def gen_ER_graph(
    nodes: int,
    density: float,
    spec_rad: float = 0.9,
    directed: bool = True,
    seed: RngLike = None,
    weight_distribution: str = "positive",
):
    r"""
    Generate an Erdős-Rényi random graph as a weighted adjacency matrix with a given spectral radius.

    Every ordered pair of different nodes :math:`(u, v)` is connected with probability :math:`p`
    (``density``), the weights of the edges are drawn from ``weight_distribution``, and the matrix is
    rescaled to the spectral radius :math:`\rho_\mathrm{target}`,

    .. math::

        A_{uv} = \xi_{uv}\, w_{uv}, \quad \xi_{uv} \sim \mathrm{Bernoulli}(p), \qquad
        A \leftarrow A\, \frac{\rho_\mathrm{target}}{\rho(A)}, \qquad
        \rho(A) = \max_i |\lambda_i(A)| .

    The edges (:math:`\xi`) are drawn first, so the support is the same ER graph for every distribution and
    seed; only the weights :math:`w_{uv}` differ.

    Parameters
    ----------
    nodes : int
        Number of nodes :math:`N`.
    density : float
        Edge probability :math:`p` in [0, 1]. The density of the generated graph varies around it.
    spec_rad : float, default=0.9
        Spectral radius :math:`\rho_\mathrm{target}` of the returned matrix.
    directed : bool, default=True
        Whether to create a directed graph.
    seed : int, Generator or None
        Seed for reproducibility. It controls the graph, the weights and the repair of isolated nodes;
        the global random state is not used.
    weight_distribution : {"positive", "sign", "uniform", "normal"}, default="positive"
        Distribution of the weights :math:`w_{uv}`, see ``WEIGHT_DISTRIBUTIONS``: all 1 (a binary
        matrix, no random numbers are drawn), :math:`\pm 1` with equal probability,
        :math:`\mathcal{U}(-1, 1)` or :math:`\mathcal{N}(0, 1)`.

    Returns
    -------
    ndarray of shape (nodes, nodes)
        The adjacency matrix; ``A[u, v]`` is the weight of the edge :math:`u \to v`.

    Raises
    ------
    ValueError
        If no graph with a directed cycle could be drawn (e.g. density 0) or the weight distribution is
        unknown.

    Notes
    -----
    Isolated nodes are not removed (that would change the size of the matrix) but connected, so that the
    matrix keeps the requested size.

    Graphs with spectral radius 0 (no directed cycle) cannot be scaled to the requested spectral radius and
    are redrawn, up to ``MAX_ER_ATTEMPTS`` times. The same holds for signed graphs in which the cycles cancel
    (:math:`\rho(A)` below ``MIN_SPEC_RAD`` times the largest weight), which would otherwise be rescaled by
    a huge factor. The first attempt is identical to a plain ER draw, so for graphs that have a cycle the
    seed gives the same graph as without the check. For reservoirs of realistic size redraws practically
    never happen; for very small or sparse graphs the result is an ER graph conditioned on having a
    directed cycle.
    """
    # all randomness comes from this generator, never from the global state
    rng = make_rng(seed)
    if weight_distribution not in WEIGHT_DISTRIBUTIONS:
        raise ValueError(
            f"unknown weight_distribution {weight_distribution!r}, use one of "
            f"{WEIGHT_DISTRIBUTIONS}."
        )

    # A graph without a directed cycle (e.g. a sparse or small graph) has spectral radius
    # 0 and cannot be rescaled to a target spectral radius. Such graphs are redrawn.
    for attempt in range(MAX_ER_ATTEMPTS):
        # use networkx to generate a random graph. In the first attempt an integer seed is
        # handed to networkx as is, otherwise a seed is drawn from the generator.
        if (
            attempt == 0
            and isinstance(seed, (int, np.integer))
            and not isinstance(seed, bool)
        ):
            nx_seed = int(seed)
        else:
            nx_seed = int(rng.integers(2**31 - 1))
        G = nx.erdos_renyi_graph(nodes, density, seed=nx_seed, directed=directed)

        # Instead of removing isolated nodes (old buggy behavior):
        # G.remove_nodes_from(list(nx.isolates(G)))

        # New: Connect isolated nodes to maintain matrix size
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            non_isolated = list(set(G.nodes()) - set(isolated_nodes))
            for node in isolated_nodes:
                if non_isolated:
                    target = rng.choice(non_isolated)
                    G.add_edge(node, target)
                    if directed:
                        G.add_edge(target, node)

        GNet = draw_edge_weights(nx.to_numpy_array(G) > 0, weight_distribution, rng)
        curr_spec_rad = max(abs(np.linalg.eigvals(GNet)))
        # A binary graph has spectral radius >= 1 if it has a cycle. With signed weights the
        # cycles can cancel (e.g. +1 -1) and leave a spectral radius of rounding noise, which
        # would be blown up by the rescaling; such draws are redrawn as well.
        if curr_spec_rad > MIN_SPEC_RAD * np.max(np.abs(GNet), initial=0.0) and not (
            nx.is_directed_acyclic_graph(nx.from_numpy_array(GNet != 0, create_using=nx.DiGraph))
        ):
            return GNet * spec_rad / curr_spec_rad

    raise ValueError(
        f"Could not draw a graph with a directed cycle (spectral radius > 0) for "
        f"nodes={nodes}, density={density} in {MAX_ER_ATTEMPTS} attempts. Increase the "
        f"number of nodes or the density."
    )


def compute_density(network: np.ndarray) -> float:
    r"""
    Density of a network, the fraction of non-zero entries of its adjacency matrix.

    .. math::

        d = \frac{\left| \{ (u, v) : A_{uv} \ne 0 \} \right|}{N^2}

    Weights of either sign count as links.

    Parameters
    ----------
    network : ndarray of shape (N, N)
        Adjacency matrix.

    Returns
    -------
    float
        The density :math:`d \in [0, 1]`.

    Raises
    ------
    TypeError
        If ``network`` is not a numpy array.
    ValueError
        If it is not square.
    """
    if not isinstance(network, np.ndarray):
        raise TypeError("adjacency matrix must be numpy.ndarray")
    if network.shape[0] != network.shape[1]:
        raise ValueError("adjacency matrix must be square")
    N = len(network)
    num_links = np.count_nonzero(network)  # signed weights count as links as well
    return num_links / (N**2)


def get_num_nodes(network: np.ndarray) -> int:
    """
    Number of nodes :math:`N` of a network given as an :math:`N \\times N` adjacency matrix.

    Raises a TypeError for something else than a numpy array and a ValueError for a non-square matrix.
    """
    if not isinstance(network, np.ndarray):
        raise TypeError("adjacency matrix must be numpy.ndarray")
    if network.shape[0] != network.shape[1]:
        raise ValueError("adjacency matrix must be square")
    return network.shape[0]


def compute_spec_rad(network: np.ndarray) -> float:
    r"""
    Spectral radius of a network, the largest absolute eigenvalue of its adjacency matrix.

    .. math::

        \rho(A) = \max_i \left| \lambda_i(A) \right|

    It is 0 for a network without a directed cycle.

    Parameters
    ----------
    network : ndarray of shape (N, N)
        Adjacency matrix.

    Returns
    -------
    float
        The spectral radius :math:`\rho(A)`.

    Raises
    ------
    TypeError
        If ``network`` is not a numpy array.
    ValueError
        If it is not square.
    """
    if not isinstance(network, np.ndarray):
        raise TypeError("adjacency matrix must be numpy.ndarray")
    if network.shape[0] != network.shape[1]:
        raise ValueError("adjacency matrix must be square")
    return np.max(np.abs(np.linalg.eigvals(network)))


def set_spec_rad(network: np.ndarray, spec_radius: float) -> np.ndarray:
    r"""
    Rescale a network to a spectral radius.

    .. math::

        A' = A\, \frac{\rho_\mathrm{new}}{\rho(A)} \quad \Rightarrow \quad \rho(A') = \rho_\mathrm{new}

    Parameters
    ----------
    network : ndarray of shape (N, N)
        Adjacency matrix with at least one directed cycle.
    spec_radius : float
        The new spectral radius :math:`\rho_\mathrm{new} > 0`. A value above 1 warns, since it is unusual
        for reservoirs (the echo state property is not guaranteed).

    Returns
    -------
    ndarray of shape (N, N)
        The rescaled matrix; ``network`` itself is not changed.

    Raises
    ------
    TypeError
        If ``network`` is not a numpy array.
    ValueError
        If it is not square, has no link, or has spectral radius 0 (no directed cycle), or if
        ``spec_radius`` is not positive.
    """
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
    if current_spectral_radius < 1e-12:
        raise ValueError(
            "the network has spectral radius 0 (it contains no directed cycle) and "
            "cannot be rescaled to a target spectral radius"
        )
    scaling_factor = spec_radius / current_spectral_radius
    return network * scaling_factor


def is_zero_col_and_row(x: np.ndarray, idx: int) -> bool:
    """True if column and row ``idx`` of the adjacency matrix ``x`` are zero, i.e. the node is isolated."""

    is_zero_column = np.all(x[:, idx] == 0)
    is_zero_row = np.all(x[idx, :] == 0)

    if is_zero_column and is_zero_row:
        return True
    else:
        return False


def remove_nodes_from_graph(graph: np.ndarray, nodes: list):
    """
    Remove nodes from a network given as an adjacency matrix.

    Parameters
    ----------
    graph : ndarray of shape (N, N)
        Adjacency matrix.
    nodes : list of int
        Indices of the nodes to remove; their rows and columns are deleted and the remaining nodes are
        renumbered.

    Returns
    -------
    ndarray of shape (N - len(nodes), N - len(nodes))
        The adjacency matrix of the remaining network.

    Raises
    ------
    TypeError
        If ``nodes`` is not a list or ``graph`` not an array.
    ValueError
        If an index is negative, too large or not an integer.
    """

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
    r"""
    Drop the removed nodes from a list of node indices and renumber the survivors.

    A surviving node :math:`v` gets the new index :math:`v - |\{ r \in R : r < v \}|`, with :math:`R` the
    set of removed nodes, which is its index in the reduced adjacency matrix.

    Parameters
    ----------
    original_nodes : list of int
        Node indices before the removal.
    removed_nodes : list of int
        Nodes that were removed.

    Returns
    -------
    list of int
        The indices of the remaining nodes in the reduced network.
    """
    removed_set = set(removed_nodes)
    updated_nodes = [
        node - sum(1 for r in removed_nodes if r < node)
        for node in original_nodes
        if node not in removed_set
    ]

    return updated_nodes


def gen_init_states(num_nodes: int, method: str = "random", rng: RngLike = None):
    r"""
    Initial reservoir state :math:`r(0)` of the given length.

    Parameters
    ----------
    num_nodes : int
        Number of nodes :math:`N`.
    method : {"random", "random_normal", "ones", "zeros"}
        Uniform on :math:`[0, 1)`, standard normal, all ones, or all zeros. All but ``"zeros"`` are divided
        by their largest absolute value, so that :math:`\max_u |r_u(0)| = 1`.
    rng : int, Generator or None
        Seed or generator of the random numbers.

    Returns
    -------
    ndarray of shape (num_nodes,)

    Raises
    ------
    ValueError
        For an unknown method.
    """

    rng = make_rng(rng)
    if method == "random":
        init_states = rng.random(num_nodes)
    elif method == "random_normal":
        init_states = rng.standard_normal(num_nodes)
    elif method == "ones":
        init_states = np.ones(num_nodes)
    elif method == "zeros":
        init_states = np.zeros(num_nodes)
    else:
        raise (
            ValueError(
                f"Sampling method {method} is unknown for generating initial reservoir "
                "states"
            )
        )

    # normalize to max. absolute value of 1
    if method != "zeros":
        init_states = init_states / np.max(np.abs(init_states))

    return init_states


def convert_to_nx_graph(graph: np.ndarray) -> nx.Graph:
    """
    Convert an adjacency matrix to a directed NetworkX graph; graphs are passed through.

    Parameters
    ----------
    graph : ndarray, nx.Graph or nx.DiGraph
        Adjacency matrix with ``graph[u, v]`` the weight of the edge ``u -> v``, or a graph.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The graph; the weights are stored as the ``weight`` attribute of the edges.

    Raises
    ------
    ValueError
        If ``graph`` is neither an array nor a NetworkX graph.
    """
    if isinstance(graph, np.ndarray):
        graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    elif not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("adjacency_matrix must be a numpy array or a NetworkX graph.")
    return graph


def extract_density(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    r"""
    Density of a directed graph, the number of edges over the number of possible edges.

    .. math::

        d = \frac{|E|}{N (N - 1)}

    Note the difference to :func:`compute_density`, which divides by :math:`N^2` (self-loops included).

    Parameters
    ----------
    graph : ndarray, nx.Graph or nx.DiGraph
        Adjacency matrix or graph.

    Returns
    -------
    float
        The density of the graph.
    """
    graph = convert_to_nx_graph(graph)

    return nx.density(graph)


def extract_spectral_radius(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    r"""
    Spectral radius :math:`\rho(A) = \max_i |\lambda_i(A)|` of the adjacency matrix of a graph.

    Parameters
    ----------
    graph : ndarray, nx.Graph or nx.DiGraph
        Adjacency matrix or graph.

    Returns
    -------
    float
        The spectral radius.
    """

    graph = convert_to_nx_graph(graph)

    # Get the adjacency matrix of the graph
    adjacency_matrix = nx.to_numpy_array(graph)

    # Compute the eigenvalues of the adjacency matrix
    eigenvalues = np.linalg.eigvals(adjacency_matrix)

    # Compute the spectral radius (maximum absolute eigenvalue)
    spectral_radius = np.max(np.abs(eigenvalues))

    return spectral_radius


def extract_av_in_degree(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    r"""Mean in-degree :math:`\frac{1}{N} \sum_v k^\mathrm{in}_v` (number of incoming edges per node)."""
    graph = convert_to_nx_graph(graph)
    in_degrees = graph.in_degree()
    avg_in_degree = np.mean(list(dict(in_degrees).values()))
    return avg_in_degree


def extract_av_out_degree(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    r"""Mean out-degree :math:`\frac{1}{N} \sum_u k^\mathrm{out}_u` (number of outgoing edges per node)."""
    graph = convert_to_nx_graph(graph)
    out_degrees = graph.out_degree()
    avg_out_degree = np.mean(list(dict(out_degrees).values()))
    return avg_out_degree


def extract_clustering_coefficient(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph]
) -> float:
    r"""Average clustering coefficient :math:`\frac{1}{N} \sum_v C_v` (see :func:`extract_node_clustering_coefficient`)."""
    graph = convert_to_nx_graph(graph)
    return nx.average_clustering(graph)


def extract_node_degree(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    r"""Degree :math:`k_v = k^\mathrm{in}_v + k^\mathrm{out}_v` of a node of a directed graph."""
    graph = convert_to_nx_graph(graph)
    return graph.degree[node]


def extract_node_in_degree(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    r"""In-degree :math:`k^\mathrm{in}_v = |\{ u : u \to v \}|` of a node."""
    graph = convert_to_nx_graph(graph)
    return graph.in_degree[node]


def extract_node_out_degree(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    r"""Out-degree :math:`k^\mathrm{out}_u = |\{ v : u \to v \}|` of a node."""
    graph = convert_to_nx_graph(graph)
    return graph.out_degree[node]


def extract_node_clustering_coefficient(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    r"""
    Clustering coefficient of a node: the fraction of the possible triangles through it that exist.

    For an undirected graph :math:`C_v = 2 T_v / (k_v (k_v - 1))` with :math:`T_v` the number of triangles
    through :math:`v`; for directed graphs the NetworkX definition is used.
    """
    graph = convert_to_nx_graph(graph)

    return nx.clustering(graph, node)


def extract_node_betweenness_centrality(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    r"""
    Betweenness centrality of a node: the share of shortest paths that pass through it.

    .. math::

        c_B(v) = \sum_{s \ne v \ne t} \frac{\sigma_{st}(v)}{\sigma_{st}} \Big/ \big((N-1)(N-2)\big)

    with :math:`\sigma_{st}` the number of shortest paths from :math:`s` to :math:`t` and
    :math:`\sigma_{st}(v)` those through :math:`v` (normalized, directed).
    """
    graph = convert_to_nx_graph(graph)
    return nx.betweenness_centrality(graph)[node]


def extract_node_pagerank(
    graph: Union[np.ndarray, nx.Graph, nx.DiGraph], node: int
) -> float:
    r"""
    PageRank of a node, the stationary distribution of a random walk with restarts on the graph structure.

    .. math::

        p = \frac{1 - \delta}{N}\, \mathbf{1} + \delta\, P^\top p

    with the damping :math:`\delta = 0.85` and the transition matrix :math:`P` of the random walk. The
    structure counts, not the (possibly negative) weights.
    """
    graph = convert_to_nx_graph(graph)
    # the structure counts, not the (possibly negative) weights
    return nx.pagerank(graph, weight=None)[node]


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


def extract_edge_abs_weight(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                            edge: tuple) -> float:
    r"""Magnitude :math:`|w_{uv}|` of the edge weight (1.0 if the edge has no explicit weight)."""
    return abs(extract_edge_weight(graph, edge))


def extract_edge_sign(graph: Union[np.ndarray, nx.Graph, nx.DiGraph],
                      edge: tuple) -> int:
    """Sign of the edge weight: +1 (excitatory) or -1 (inhibitory)."""
    return -1 if extract_edge_weight(graph, edge) < 0 else 1


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
