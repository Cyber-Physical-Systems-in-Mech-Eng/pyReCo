from typing import Union
import numpy as np
import networkx as nx

from pyreco.utils_networks import (
    extract_node_degree,
    extract_node_in_degree,
    extract_node_out_degree,
    extract_node_clustering_coefficient,
    extract_node_betweenness_centrality,
    extract_node_pagerank,
)


def available_extractors():
    """
    Return a dictionary mapping network property names to their corresponding extractor functions. Make changes here if you want to add more network properties.

    Returns:
        dict: A dictionary mapping network property names to their corresponding extractor functions.
    """

    # Define a dictionary mapping network property names to their corresponding extractor functions
    # as implemented in utils_networks.py

    all_extractors = {
        "degree": extract_node_degree,
        "in_degree": extract_node_in_degree,
        "out_degree": extract_node_out_degree,
        "clustering_coefficient": extract_node_clustering_coefficient,
        "betweenness_centrality": extract_node_betweenness_centrality,
        "pagerank": extract_node_pagerank,
    }

    return all_extractors


def map_extractor_names(prop_names: str):
    """
    Return a dictionary mapping network property names to their corresponding extractor functions for the given property names.

    Returns:
        dict: A dictionary mapping network property names to their corresponding extractor functions.
    """

    # Define a dictionary mapping network property names to their corresponding extractor functions
    # as implemented in utils_networks.py

    all_extractors = available_extractors()

    # now return those extractors that are in prop_names
    extractor_dict = {}
    extractor_funs = []
    for prop in prop_names:
        if prop not in all_extractors:
            print(f"Warning: {prop} is not a recognized node property.")
            prop_names.remove(prop)
        else:
            extractor_dict[prop] = all_extractors[prop]
            extractor_funs.append(all_extractors[prop])
    return extractor_dict, extractor_funs


class NodeAnalyzer:
    """
    A class for analyzing node properties in a graph.
    """

    def __init__(self, quantities=None):
        """
        Initialize the NodeAnalyzer with specified quantities to extract.

        Args:
            quantities (list, optional): List of network properties to extract.
                Defaults to ['degree', 'in_degree', 'out_degree', 'clustering_coefficient', 'betweenness_centrality', 'pagerank'].
        """
        all_extractors = available_extractors().keys()
        self.quantities = quantities or list(all_extractors)

        # map given property names to their corresponding extractor functions
        self.extractors, self.extractor_funs = map_extractor_names(self.quantities)

    def extract_properties(
        self, graph: Union[nx.Graph, nx.DiGraph, np.ndarray], node: int
    ) -> dict:
        """
        Extract the specified network properties from the given graph at the given node(s).

        Args:
            graph (nx.Graph, nx.DiGraph, np.ndarray): The graph to analyze.

            nodes (int): The node to analyze.

        Returns:
            dict: A dictionary containing the extracted network properties.
        """

        if not isinstance(node, int):
            raise ValueError(
                "node must be an integer, we currently support only one node at a time"
            )

        graph_props = {}
        for extr_name, extr_fun in self.extractors.items():
            graph_props[extr_name] = extr_fun(graph=graph, node=node)
        return graph_props

    def list_properties(self):
        """
        Return a list of available node properties.

        Returns:
            list: A list of available node properties.
        """
        return available_extractors().keys()


if __name__ == "__main__":

    # Create a sample graph
    G = nx.erdos_renyi_graph(10, 0.5, directed=True)

    # Specify the node for which you want to extract the PageRank
    node = 3

    extractor = NodeAnalyzer()
    graph_props = extractor.extract_properties(graph=G, node=node)
    print(graph_props)
