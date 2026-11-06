"""
Whole-network and per-node properties of an adjacency matrix in one call.

:class:`NetworkQuantifier` returns scalar properties of the network (density, spectral radius, mean
degrees, clustering) and :class:`NodePropExtractor` returns, for every node, properties such as its
degree or its PageRank. Both build on the extractor functions of :mod:`pyreco.utils_networks`. The
analyzers of :class:`pyreco.pruning.NetworkPruner` are :class:`pyreco.edge_analyzer.EdgeAnalyzer`,
:class:`pyreco.node_analyzer.NodeAnalyzer` and :class:`pyreco.graph_analyzer.GraphAnalyzer`; the edge pruning
measures the named features of :mod:`pyreco.features`.
"""

import warnings

import numpy as np

from .utils_networks import (
    convert_to_nx_graph,
    extract_av_in_degree,
    extract_av_out_degree,
    extract_clustering_coefficient,
    extract_density,
    extract_node_betweenness_centrality,
    extract_node_clustering_coefficient,
    extract_node_degree,
    extract_node_in_degree,
    extract_node_out_degree,
    extract_node_pagerank,
    extract_spectral_radius,
)

NETWORK_PROPERTIES = {
    "density": extract_density,
    "spectral_radius": extract_spectral_radius,
    "in_degree_av": extract_av_in_degree,
    "out_degree_av": extract_av_out_degree,
    "clustering_coefficient": extract_clustering_coefficient,
}

NODE_PROPERTIES = {
    "degree": extract_node_degree,
    "in_degree": extract_node_in_degree,
    "out_degree": extract_node_out_degree,
    "clustering_coefficient": extract_node_clustering_coefficient,
    "betweenness_centrality": extract_node_betweenness_centrality,
    "pagerank": extract_node_pagerank,
}


class NetworkQuantifier:
    r"""
    Scalar properties of a network, given its adjacency matrix :math:`A` (``A[u, v]`` is the edge
    :math:`u \to v`).

    The available properties are the density :math:`|E| / (N (N - 1))`, the spectral radius
    :math:`\rho(A) = \max_i |\lambda_i(A)|`, the mean in-degree :math:`\frac{1}{N} \sum_v k^\mathrm{in}_v`, the mean
    out-degree :math:`\frac{1}{N} \sum_u k^\mathrm{out}_u` and the average clustering coefficient
    (``"density"``, ``"spectral_radius"``, ``"in_degree_av"``, ``"out_degree_av"``,
    ``"clustering_coefficient"``).

    Parameters
    ----------
    quantities : list of str, optional
        The properties to extract. By default all of them. Unknown names are skipped with a warning.
    """

    def __init__(self, quantities=None):
        quantities = list(NETWORK_PROPERTIES) if quantities is None else list(quantities)
        self.quantities = _known(quantities, NETWORK_PROPERTIES, "network")
        self.extractors = {name: NETWORK_PROPERTIES[name] for name in self.quantities}

    def extract_properties(self, adjacency_matrix) -> dict:
        """
        Extract the selected properties.

        Parameters
        ----------
        adjacency_matrix : ndarray, nx.Graph or nx.DiGraph
            The network.

        Returns
        -------
        dict
            Property name -> value.
        """
        return {name: fun(adjacency_matrix) for name, fun in self.extractors.items()}


class NodePropExtractor:
    r"""
    Properties of every node of a network, given its adjacency matrix :math:`A` (``A[u, v]`` is the edge
    :math:`u \to v`).

    The available properties are the degree :math:`k_v = k^\mathrm{in}_v + k^\mathrm{out}_v`, the in- and the
    out-degree, the clustering coefficient, the betweenness centrality and the PageRank (``"degree"``,
    ``"in_degree"``, ``"out_degree"``, ``"clustering_coefficient"``, ``"betweenness_centrality"``,
    ``"pagerank"``; see :mod:`pyreco.utils_networks` for the definitions).

    Parameters
    ----------
    properties : list of str, optional
        The properties to extract. By default all of them. Unknown names are skipped with a warning.
    """

    def __init__(self, properties=None):
        properties = list(NODE_PROPERTIES) if properties is None else list(properties)
        self.properties = _known(properties, NODE_PROPERTIES, "node")
        self.extractors = {name: NODE_PROPERTIES[name] for name in self.properties}

    def extract_properties(self, adjacency_matrix, states=None) -> dict:
        """
        Extract the selected properties for all nodes.

        Parameters
        ----------
        adjacency_matrix : ndarray, nx.Graph or nx.DiGraph
            The network.
        states : ndarray, optional
            Not used (reserved for state-dependent properties).

        Returns
        -------
        dict
            Property name -> ndarray of shape (N,) with the value of every node, in the order of the node
            indices.
        """
        graph = convert_to_nx_graph(adjacency_matrix)
        nodes = list(graph.nodes)
        return {
            name: np.array([fun(graph, node) for node in nodes], dtype=float)
            for name, fun in self.extractors.items()
        }


def _known(names, available, kind):
    """The names that are available; the others are reported in one warning."""
    unknown = [name for name in names if name not in available]
    if unknown:
        warnings.warn(f"not recognized {kind} properties, skipped: {unknown}", UserWarning, stacklevel=3)
    return [name for name in names if name in available]
