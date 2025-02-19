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

        Parameters
        ----------
        quantities : list, optional
            A list of network properties to extract. If not provided, defaults to 
            ['density', 'spectral_radius', 'in_degree_av', 'out_degree_av', 'clustering_coefficient'].
            The properties are used to determine which network characteristics will be computed.

        Notes
        -----
        - The `quantities` argument is a list of strings that correspond to network properties. If no 
        argument is passed, the default set of quantities is used.
        - The `extractors` dictionary maps the property names to their corresponding extraction functions.
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

        Parameters
        ----------
        adjacency_matrix : numpy.ndarray
            The adjacency matrix of the network, which represents the connectivity between nodes.

        Returns
        -------
        dict
            A dictionary containing the extracted network properties, where keys are the property names 
            (e.g., 'density', 'spectral_radius') and values are the computed property values.

        Notes
        -----
        - The method iterates over the list of quantities specified during initialization and applies 
        the corresponding extraction function from the `extractors` dictionary.
        - If a quantity is not recognized, a warning is printed.
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

        Parameters
        ----------
        properties : list, optional
            A list of node properties to extract. Defaults to 
            ['degree', 'in_degree', 'out_degree', 'clustering_coefficient', 
            'betweenness_centrality', 'pagerank'].

        Notes
        -----
        - The `properties` list allows users to specify which node attributes to extract from the network.
        - If no properties are provided, the default properties are used, which include various centrality 
        and connectivity measures for the nodes in the network.
        - The `extractors` dictionary maps the property names to their corresponding extraction functions.
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

        Parameters
        ----------
        adjacency_matrix : numpy.ndarray
            The adjacency matrix of the network, where each element represents the connection 
            between two nodes.
            
        states : numpy.ndarray, optional
            Node states, if applicable. Defaults to None. If provided, the states may be used
            in the extraction process for specific properties (e.g., centrality measures).

        Returns
        -------
        dict
            A dictionary containing the extracted node properties, where the keys are property names 
            and the values are the corresponding computed values.

        Notes
        -----
        - The method iterates over the list of properties specified in the `properties` attribute of the 
        class and calls the corresponding extraction function from the `extractors` dictionary.
        - If a property is not recognized, a warning is printed.
        """
        node_props = {}
        for property in self.properties:
            if property in self.extractors:
                    node_props[property] = self.extractors[property](adjacency_matrix)
            else:
                print(f"Warning: {property} is not a recognized node property.")
        return node_props
