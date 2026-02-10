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
        total_edges: int = None,
        graph: nx.Graph | np.ndarray = None,
    ):
        """
        Initializes the EdgeSelector object.

        Parameters:
        - total_edges (int, optional): The total number of edges in the graph. Must be a positive integer.
        - graph (nx.Graph, optional): A NetworkX graph object. Either total_edges or graph must be provided, not both.
        #TODO See if passing of total_edges even makes sense and adjust in rest of code accordingly
        - strategy (str, optional): The strategy used for node selection. Currently implements "random_uniform_wo_repl".

        Raises:
        - ValueError: If both total_edges and graph are provided, or if neither is provided.
        - TypeError: If total_edges is not an integer or if graph is not a NetworkX graph.
        - ValueError: If total_edges is not a positive integer.

        ToDo: let the method also accept adjacency matrices (np.ndarray)
        """

        # Sanity checks
        if total_edges is not None and graph is not None:
            raise ValueError("Specify either total_edges or graph, not both")

        if total_edges is not None:
            if not isinstance(total_edges, int):
                raise TypeError("total_edges must be a positive integer")
            elif total_edges <= 0:
                raise ValueError("total_edges must be a positive integer")
            graph_shape = total_edges
        elif graph is not None:
            if not isinstance(graph, nx.Graph) and not isinstance(graph, np.ndarray):
                raise TypeError("graph must be a networkx graph or np.ndarray")
            # TODO work put this adjustment
            if isinstance(graph, nx.Graph):
                total_edges = graph.number_of_edges()
                total_nodes = graph.number_of_nodes()
                graph_shape = total_nodes
            elif isinstance(graph, np.ndarray):
                total_edges = np.count_nonzero(graph) #TODO check if I need to half for undirected graph
                graph_shape = graph.shape
        else:
            raise ValueError("Either total_edges or graph must be provided")

        if strategy != "random_uniform_wo_repl":
            raise NotImplementedError(
                "Only random w/o replacement ('random_uniform_wo_repl') strategy is implemented"
            )

        # Assign values to attributes
        self.num_total_edges: int = total_edges
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

        # Finally pick the node according to the strategy
        if self.strategy == "random_uniform_wo_repl":
            # random uniform WITHOUT replacement

            self.selected_edges = random.sample(
                range(0, self.num_total_edgess), self.num_select_edges
            )

            if isinstance(self.graph_shape, int):
                # input was list, output will be list
                return self.selected_edges

            elif isinstance(self.graph_shape, tuple) or isinstance(
                self.graph_shape, list
            ):
                selected_graph = np.zeros(self.graph_shape).flatten()
                selected_graph[self.selected_edges] = 1
                self.selected_edges = np.reshape(selected_graph, self.graph_shape)

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

    # Select random edges
    selector = EdgeSelector(strategy="random_uniform_wo_repl", graph=G)
    random_edges = selector.select_edges(num=4)
    print(f"Randomly selected edges: {random_edges}")
