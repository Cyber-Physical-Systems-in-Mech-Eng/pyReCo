import random
import networkx as nx
import numpy as np
from typing import Union


class EdgeSelector:
    """
    A class to select edges from a graph based on specific criteria.

    This class provides functionality to select a subset of edges from a total number of edges
    using different strategies. Currently, only the "random without replacement" strategy is implemented.

    Attributes:
    - num_total_edges (int): The total number of edges in the graph.
    - num_select_edges (int): The number of edges to select.
    - fraction (float): The fraction of edges to select.
    - strategy (str): The strategy used for edge selection.
    - selected_edges (list): The list of selected edges.
    """

    def __init__(
        self,
        strategy: str = "random_uniform_wo_repl",
        graph: nx.Graph | np.ndarray = None,
    ):
        """
        Initializes the EdgeSelector object.

        Parameters:
        - graph (nx.Graph, optional): A NetworkX graph object. Graph must be provided #TODO graph is weight matrix right?
        #TODO adjacency matrix could also be passed
        - strategy (str, optional): The strategy used for edge selection. Currently implements "random_uniform_wo_repl".

        Raises:
        - TypeError: If graph is not a NetworkX graph.

        ToDo: let the method also accept adjacency matrices (np.ndarray)
        """

        # Sanity checks
        if graph is not None:
            if not isinstance(graph, nx.Graph) and not isinstance(graph, np.ndarray):
                raise TypeError("graph must be a networkx graph or np.ndarray")
        else:
            raise ValueError("Graph must be provided")

        if strategy != "random_uniform_wo_repl":
            raise NotImplementedError(
                "Only random w/o replacement ('random_uniform_wo_repl') strategy is implemented"
            )

        # Prunable edges in graph
        if isinstance(graph, nx.Graph):
            edge_indices = list(graph.edges()) #TODO rethink when awake if node connections are equivalent to indices but I think so
            graph_shape = graph.number_of_nodes()  # total nodes
        elif isinstance(graph, np.ndarray):
            rows, cols = np.where(graph != 0)  # where entries are not zero
            edge_indices = list(zip(rows, cols))
            graph_shape = graph.shape

        # Assign values to attributes
        self.graph = graph
        #print(self.graph)
        self.edge_indices = edge_indices
        self.num_total_edges: int = len(self.edge_indices)
        self.graph_shape = graph_shape
        self.num_select_edges: int = 0
        self.fraction: float = 0.0
        self.strategy: str = strategy
        self.selected_edges: list = []

    def select_edges(
        self,
        fraction: float = None,
        num: int = None,
    ):
        """
        Selects a specified number of edges from the graph either by fraction or by exact number.

        Parameters:
        - fraction (float, optional): The fraction of the total edges to select. Must be between 0 and 1.
        - num (int, optional): The exact number of edges to select. Must be a positive integer.

        Raises:
        - ValueError: If neither or both of fraction and num are provided.
        - TypeError: If num is not an integer.

        Returns:
        - list: A list of selected edges identifiers.
        """

        # potentially implemement more advanced selectors that inherit form the base class for degree-based selection or others.

        # Sanity checks

        if fraction is not None and not isinstance(fraction, float):
            raise TypeError("fraction must be a float in the range (0, 1]")

        if (num is not None) and (not isinstance(num, int)):
            raise TypeError("num must be an integer")

        if (num is not None) and ((num > self.num_total_edges) or (num <= 0)):
            raise ValueError(
                "number of edges to select must be maximum number of total edges, and larger than 0"
            )

        if (fraction is None) and (num is None):
            raise ValueError(
                "Either <fraction> of edges to select or <num> number of edges must be provided"
            )

        if (fraction is not None) and (num is not None):
            raise ValueError(
                "Either <fraction> of edges to select or <num> number of edges must be provided, not both"
            )

        if (num is None) and (fraction is not None):
            if fraction > 1.0 or fraction <= 0.0:
                raise ValueError("fraction must be larger than 0 and smaller than 1")

        # Assign values to class attributes
        if (fraction is None) and (num > 0):
            self.num_select_edges = num
            self.fraction = num / self.num_total_edges
        elif (fraction is not None) and (num is None):
            self.num_select_edges = round(self.num_total_edges * fraction)
            self.fraction = fraction

        # Finally pick the edges according to the strategy
        if self.strategy == "random_uniform_wo_repl":  #TODO maybe store actual methods elsewhere?
            # random uniform WITHOUT replacement

            self.selected_edges = random.sample(
                self.edge_indices, self.num_select_edges
            )

            if isinstance(self.graph_shape, int): #TODO understand why graph shape is checked
                # input was list, output will be list
                return self.selected_edges

            elif isinstance(self.graph_shape, tuple) or isinstance(
                self.graph_shape, list
            ):  #TODO understand why graph shape is checked
                #selected_graph = np.zeros(self.graph_shape).flatten()
                #selected_graph[self.selected_edges] = 1
                #self.selected_edges = np.reshape(selected_graph, self.graph_shape)

                return self.selected_edges

            else:
                raise ValueError("The graph shape/type is not supported")
        else:
            raise NotImplementedError(
                "Only random w/o replacement ('random_uniform_wo_repl') strategy is implemented"
            )


if __name__ == "__main__":

    # Create a sample graph
    G = nx.erdos_renyi_graph(10, 0.5)
    # Graphs edges
    print(f"Possible edges: {G.edges()}")
    # Select random edges
    selector = EdgeSelector(strategy="random_uniform_wo_repl", graph=G)
    random_edges = selector.select_edges(num=4)
    print(f"Randomly selected edges: {random_edges}")
