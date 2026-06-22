from typing import Union
import numpy as np
import networkx as nx

from pyreco.utils_networks import (
    convert_to_nx_graph,
    precompute_edge_metrics,
    extract_edge_weight,
    extract_edge_is_reciprocal,
    extract_edge_in_scc,
    extract_edge_betweenness,
    extract_edge_source_out_degree,
    extract_edge_target_in_degree,
    extract_edge_source_betweenness,
    extract_edge_target_betweenness,
)


def available_extractors():
    """
    Returns a dictionary mapping edge property names to their extractor functions.
    Each function has signature f(graph, edge) -> scalar, where edge is a (u, v) tuple.
    Make changes here to add more edge properties.

    Returns
    -------
    dict
        mapping property name -> extractor function
    """
    return {
        "weight": extract_edge_weight,
        "is_reciprocal": extract_edge_is_reciprocal,
        "in_scc": extract_edge_in_scc,
        "betweenness": extract_edge_betweenness,
        "source_out_degree": extract_edge_source_out_degree,
        "target_in_degree": extract_edge_target_in_degree,
        "source_betweenness": extract_edge_source_betweenness,
        "target_betweenness": extract_edge_target_betweenness,
    }


def map_extractor_names(prop_names: list):
    """
    Return extractors for the requested property names.

    Returns
    -------
    dict
        Dict mapping property name to extractor function
    list
        List of extractor functions in same order
    """
    # Define dictionary mapping network property names to their corresponding
    #   extractor functions
    all_extractors = available_extractors()
    extractor_dict = {}
    extractor_funs = []
    for prop in prop_names:
        if prop not in all_extractors:
            print(f"Warning: {prop} is not a recognized edge property.")
        else:
            extractor_dict[prop] = all_extractors[prop]
            extractor_funs.append(all_extractors[prop])
    return extractor_dict, extractor_funs


class EdgeAnalyzer:
    """
    A class for analyzing properties of a specific edge in a graph,
    analogue to 'NodeAnalyzer for nodes.

    Takes a graph and a specific edge (u, v), returning a flat dict of scalars
    describing that edge and the nodes it connects.

    Note: betweenness-based properties (betweenness, source_betweenness,
    target_betweenness) each trigger a full graph betweenness computation.
    If all three are requested, consider passing a subset via quantities to
    avoid redundant computation on large graphs.
    """

    def __init__(self, quantities=None):
        """
        Initializes the EdgeAnalyzer with specified quantities to extract.

        Parameters
        ----------
        quantities : list, optional
            List of edge properties to extract.
            Defaults to all at the moment available:
            ['abs_weight', 'sign', 'is_reciprocal', 'in_scc', 'betweenness',
             'source_out_degree', 'target_in_degree',
             'source_betweenness', 'target_betweenness'].
        """
        all_extractors = available_extractors().keys()
        self.quantities = quantities or list(all_extractors)
        self.extractors, self.extractor_funs = map_extractor_names(self.quantities)

    def extract_properties(
        self, graph: Union[nx.Graph, nx.DiGraph, np.ndarray], edge: tuple
    ) -> dict:
        """
        Extracts the specified properties for a given edge.

        Parameters
        ----------
        graph : nx.Graph, nx.DiGraph, or np.ndarray
            Graph to analyze.
        edge : tuple
            Edge (u, v) to extract properties for.

        Returns
        -------
        dict
            Dictionary containing the extracted edge properties.
        """
        # Check edge is tuple list with two entries specifiying edge
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("edge must be a (u, v) tuple")

        #  dict for properties
        edge_props = {}
        # Get property name and functions to store in dict
        for extr_name, extr_fun in self.extractors.items():
            edge_props[extr_name] = extr_fun(graph, edge)
        return edge_props

    def extract_properties_batch(
        self, graph: Union[nx.Graph, nx.DiGraph, np.ndarray], edges: list
    ) -> list:
        """
        Extracts properties for a list of edges with expensive graph-level metrics
        (betweenness centrality, SCC) so that they're computed only once.

        Parameters
        ----------
        graph : nx.Graph, nx.DiGraph, or np.ndarray
            Graph in its current state, e.g. before edge removal.
        edges : list
            List of edges, (u, v) tuples, to extract properties for.

        Returns
        -------
        list
            List of dicts containing the extracted edge properties.
            One per edge, in the same order as edges.
        """
        # Convert here already to use networkx functions
        g = convert_to_nx_graph(graph)

        # Pre compute node_betweenness, edge_betweenness and scc_map on one graph
        cache = precompute_edge_metrics(g)

        # Create look up table, basically just functions to look up edge properties
        #   from pre compute
        _cached = {
            "weight":              lambda u, v: g[u][v].get("weight", 1.0),
            "is_reciprocal":       lambda u, v: int(g.has_edge(v, u)),
            "in_scc":              lambda u, v: int(cache["scc_map"][u] == cache["scc_map"][v]),
            "betweenness":         lambda u, v: cache["edge_betweenness"].get((u, v), 0.0),
            "source_out_degree":   lambda u, v: g.out_degree[u],
            "target_in_degree":    lambda u, v: g.in_degree[v],
            "source_betweenness":  lambda u, v: cache["node_betweenness"][u],
            "target_betweenness":  lambda u, v: cache["node_betweenness"][v],
        }

        results = []
        for edge in edges:
            # Go through edges
            u, v = int(edge[0]), int(edge[1])
            # Look up values for edge for specified properties
            props = {name: _cached[name](u, v) for name in self.quantities if name in _cached}
            # Append properties for edge
            results.append(props)

        # Return list with edge properties for all edges
        return results

    def list_properties(self):
        """
        Returns list of available edge properties.

        Returns
        -------
        list
            Available edge property names.
        """
        return list(available_extractors().keys())


if __name__ == "__main__":

    G = nx.erdos_renyi_graph(10, 0.5, directed=True)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.randn()

    edge = list(G.edges())[0]
    print(f"Properties for edge {edge}:")

    analyzer = EdgeAnalyzer()
    props = analyzer.extract_properties(G, edge)
    for k, v in props.items():
        print(f"  {k}: {v}")
