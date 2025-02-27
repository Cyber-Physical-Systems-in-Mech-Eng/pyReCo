from typing import Union
import numpy as np
import networkx as nx

from pyreco.utils_networks import (
    extract_density,
    extract_spectral_radius,
    extract_av_in_degree,
    extract_av_out_degree,
    extract_clustering_coefficient,
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
        "density": extract_density,
        "spectral_radius": extract_spectral_radius,
        "av_in_degree": extract_av_in_degree,
        "av_out_degree": extract_av_out_degree,
        "clustering_coefficient": extract_clustering_coefficient,
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
            print(f"Warning: {prop} is not a recognized network property.")
            prop_names.remove(prop)
        else:
            extractor_dict[prop] = all_extractors[prop]
            extractor_funs.append(all_extractors[prop])
    return extractor_dict, extractor_funs


class GraphAnalyzer:
    """
    A class for analyzing graph properties and extracting network properties from a graph.
    """

    def __init__(self, quantities=None):
        """
        Initialize the GraphAnalyzer with specified quantities to extract.

        Args:
            quantities (list, optional): List of network properties to extract.
                Defaults to ['density', 'spectral_radius', 'in_degree_av', 'out_degree_av', 'clustering_coefficient'].
        """
        all_extractors = available_extractors().keys()
        self.quantities = quantities or list(all_extractors)

        # map given property names to their corresponding extractor functions
        self.extractors, self.extractor_funs = map_extractor_names(self.quantities)

    def extract_properties(
        self, graph: Union[nx.Graph, nx.DiGraph, np.ndarray]
    ) -> dict:
        """
        Extract the specified network properties from the given graph.

        Args:
            graph (nx.Graph, nx.DiGraph, np.ndarray): The graph to analyze.

        Returns:
            dict: A dictionary containing the extracted network properties.
        """

        network_props = {}
        for extr_name, extr_fun in self.extractors.items():
            network_props[extr_name] = extr_fun(graph)
        return network_props

    def list_properties(self):
        """
        Return a list of available network properties.

        Returns:
            list: A list of available network properties.
        """
        return available_extractors().keys()

    # def extract_node_properties(self, graph):
    #     """
    #     Extract the specified node properties from the given graph.

    #     Args:
    #         graph (networkx.Graph): The graph to analyze.

    #     Returns:
    #         dict: A dictionary containing the extracted node properties.
    #     """
    #     node_props = {}
    #     node_props["degree"] = extract_node_degree(graph)
    #     node_props["in_degree"] = extract_node_in_degree(graph)
    #     node_props["out_degree"] = extract_node_out_degree(graph)
    #     node_props["clustering_coefficient"] = extract_node_clustering_coefficient(
    #         graph
    #     )
    #     node_props["betweenness_centrality"] = extract_node_betweenness_centrality(
    #         graph
    #     )
    #     node_props["pagerank"] = extract_node_pagerank(graph)
    #     return node_props
