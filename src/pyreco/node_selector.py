import random
import networkx as nx


class NodeSelector:
    """
    A class to select nodes from a graph based on specific criteria.

    This class provides functionality to select a subset of nodes from a total number of nodes
    using different strategies. Currently, only the "random without replacement" strategy is implemented.

    Attributes:
    - num_total_nodes (int): The total number of nodes in the graph.
    - num_select_nodes (int): The number of nodes to select.
    - fraction (float): The fraction of nodes to select.
    - strategy (str): The strategy used for node selection.
    - selected_nodes (list): The list of selected nodes.
    """

    def __init__(
        self,
        total_nodes: int = None,
        graph: nx.Graph = None,
        strategy: str = "random_wo_replacement",
    ):
        """
        Initializes the NetworkPruner object.

        Parameters:
        - total_nodes (int, optional): The total number of nodes in the graph. Must be a positive integer.
        - graph (nx.Graph, optional): A NetworkX graph object. Either total_nodes or graph must be provided, not both.
        - strategy (str, optional): The strategy used for node selection. Defaults to "random_wo_replacement".

        Raises:
        - ValueError: If both total_nodes and graph are provided, or if neither is provided.
        - TypeError: If total_nodes is not an integer or if graph is not a NetworkX graph.
        - ValueError: If total_nodes is not a positive integer.
        """

        # Sanity checks
        if total_nodes is not None and graph is not None:
            raise ValueError("Specify either total_nodes or graph, not both")

        if total_nodes is not None:
            if not isinstance(total_nodes, int):
                raise TypeError("total_nodes must be a positive integer")
            elif total_nodes <= 0:
                raise ValueError("num_nodes must be a positive integer")
        elif graph is not None:
            if not isinstance(graph, nx.Graph):
                raise TypeError("graph must be a networkx graph")
            total_nodes = graph.number_of_nodes()
        else:
            raise ValueError("Either total_nodes or graph must be provided")

        if strategy != "random_wo_replacement":
            raise NotImplementedError(
                "Only 'random w/o replacement' strategy is implemented"
            )

        # Assign values to attributes
        self.num_total_nodes: int = total_nodes
        self.num_select_nodes: int = 0
        self.fraction: float = 0.0
        self.strategy: str = strategy
        self.selected_nodes: list = []

    def select_node(
        self,
        fraction: float = None,
        num: int = None,
    ):
        """
        Selects a specified number of nodes from the graph either by fraction or by exact number.

        Parameters:
        - fraction (float, optional): The fraction of the total nodes to select. Must be between 0 and 1.
        - num (int, optional): The exact number of nodes to select. Must be a positive integer.

        Raises:
        - ValueError: If neither or both of fraction and num are provided.
        - TypeError: If num is not an integer.

        Returns:
        - list: A list of selected node identifiers.
        """

        # potentially implemement more advanced selectors that inherit form the base class for degree-based selection or others.

        # Sanity checks
        if (fraction is None) and (num is None):
            raise ValueError(
                "Either <fraction> of nodes to select or <num> number of nodes must be provided"
            )

        if (fraction is not None) and (num is not None):
            raise ValueError(
                "Either <fraction> of nodes to select or <num> number of nodes must be provided, not both"
            )

        if (num is not None) and (not isinstance(num, int)):
            raise TypeError("num must be an integer")

        if (num is not None) and ((num >= self.num_total_nodes) or (num <= 0)):
            raise ValueError(
                "number of nodes to select must be smaller than number of total nodes, and larger than 0"
            )

        if fraction is not None and not isinstance(fraction, float):
            raise TypeError("fraction must be a float in the range (0, 1)")

        if fraction >= 1.0 or fraction <= 0.0:
            raise ValueError("fraction must be larger than 0 and smaller than 1")

        # Assign values to class attributs
        if (fraction is None) and (num > 0):
            self.num_select_nodes = num
            self.fraction = num / self.num_total_nodes
        elif (fraction is not None) and (num is None):
            self.num_select_nodes = int(self.num_total_nodes * fraction)
            self.fraction = fraction

        # Finally pick the node according to the strategy
        if self.strategy == "random_wo_replacement":
            self.selected_nodes = random.sample(
                range(0, self.num_total_nodes), self.num_select_nodes
            )

            return self.selected_nodes
