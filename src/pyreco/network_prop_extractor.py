import numpy as np
from .utils_networks import (
    extract_density, extract_spectral_radius, extract_in_degree_av,
    extract_out_degree_av, extract_clustering_coefficient,
    extract_node_degree, extract_node_in_degree, extract_node_out_degree,
    extract_node_clustering_coefficient, extract_node_betweenness_centrality,
    extract_node_pagerank
)

class NetworkQuantifier:
    """
    A class for extracting and quantifying network properties from an adjacency matrix.

    This class provides methods to extract various network-level properties such as
    density, spectral radius, average in-degree, average out-degree, and clustering coefficient.
    """

    def __init__(self, quantities=None):
        """
        Initialize the NetworkQuantifier with specified quantities to extract.

        Args:
            quantities (list, optional): List of network properties to extract.
                Defaults to ['density', 'spectral_radius', 'in_degree_av', 'out_degree_av', 'clustering_coefficient'].
        """
        self.quantities = quantities or ['density', 'spectral_radius', 'in_degree_av', 'out_degree_av', 'clustering_coefficient']
        self.extractors = {
            'density': extract_density,
            'spectral_radius': extract_spectral_radius,
            'in_degree_av': extract_in_degree_av,  
            'out_degree_av': extract_out_degree_av,
            'clustering_coefficient': extract_clustering_coefficient
        }

    def extract_properties(self, adjacency_matrix):
        """
        Extract the specified network properties from the given adjacency matrix.

        Args:
            adjacency_matrix (numpy.ndarray): The adjacency matrix of the network.

        Returns:
            dict: A dictionary containing the extracted network properties.
        """
        network_props = {}
        for quantity in self.quantities:
            if quantity in self.extractors:
                network_props[quantity] = self.extractors[quantity](adjacency_matrix)
            else:
                print(f"Warning: {quantity} is not a recognized network property.")
        return network_props

class NodePropExtractor:
    """
    A class for extracting and quantifying node-level properties from an adjacency matrix.

    This class provides methods to extract various node-level properties such as
    degree, in-degree, out-degree, clustering coefficient, betweenness centrality, and PageRank.
    """

    def __init__(self, properties=None):
        """
        Initialize the NodePropExtractor with specified properties to extract.

        Args:
            properties (list, optional): List of node properties to extract.
                Defaults to ['degree', 'in_degree', 'out_degree', 'clustering_coefficient', 'betweenness_centrality', 'pagerank'].
        """
        self.properties = properties or ['degree', 'in_degree', 'out_degree', 'clustering_coefficient', 'betweenness_centrality', 'pagerank']
        self.extractors = {
            'degree': extract_node_degree,
            'in_degree': extract_node_in_degree,
            'out_degree': extract_node_out_degree,
            'clustering_coefficient': extract_node_clustering_coefficient,
            'betweenness_centrality': extract_node_betweenness_centrality,
            'pagerank': extract_node_pagerank,
        }

    def extract_properties(self, adjacency_matrix, states=None):
        """
        Extract the specified node properties from the given adjacency matrix.

        Args:
            adjacency_matrix (numpy.ndarray): The adjacency matrix of the network.
            states (numpy.ndarray, optional): Node states, if applicable. Defaults to None.

        Returns:
            dict: A dictionary containing the extracted node properties.
        """
        node_props = {}
        for property in self.properties:
            if property in self.extractors:
                    node_props[property] = self.extractors[property](adjacency_matrix)
            else:
                print(f"Warning: {property} is not a recognized node property.")
        return node_props
