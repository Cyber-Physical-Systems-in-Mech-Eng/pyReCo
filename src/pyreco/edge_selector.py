import random
import networkx as nx
import numpy as np
import warnings


class EdgeSelector:
    '''
    A class to select edges from a graph based on specific criteria,
    analogue to 'NodeSelector'.

    Parameters
    ----------
    graph : nx.Graph or np.ndarray
        NetworkX graph or adjacency matrix to select edges from.
    strategy : str, optional
        Strategy used for edge selection.
        Default is "random_uniform_wo_repl".

    Attributes
    ----------
    graph : nx.Graph or np.ndarray
        Input graph.
    directed : bool, optional
        Whether the graph is directed. Only used for np.ndarray inputs.
        Default is True.
    edge_indices : list of tuple
        List of all edge indices in the graph.
    num_total_edges : int
        Total number of edges in the graph.
    num_select_edges : int
        Number of edges to select. Set after calling select_edges.
    fraction : float
        Fraction of edges to select. Set after calling select_edges.
    strategy : callable
        Method corresponding to the selected strategy.
    selected_edges : list
        List of selected edges. Set after calling select_edges.

    Raises
    ------
    TypeError
        If graph is not a nx.Graph or np.ndarray.
    ValueError
        If graph is None.
    NotImplementedError
        If the requested strategy is not implemented.
    '''

    # Selection strategies implemented
    # TODO think about putting these in classes and have abstract base class to
    #       define them
    STRATEGIES = {
        'random_uniform_wo_repl': '_random_uniform_wo_repl',

    }

    def __init__(
        self,
        graph: nx.Graph | np.ndarray = None,
        strategy: str = 'random_uniform_wo_repl',
        directed: bool = True,
    ):
        # Sanity checks for passed graph and selection strategy
        self._validate_graph(graph, directed)
        self._validate_strategy(strategy)

        # Get all edges of the graph
        edge_indices = self._extract_edges(graph, directed)

        # Assign values to attributes
        self.graph = graph
        self.directed = directed
        self.edge_indices = edge_indices
        self.num_total_edges: int = len(self.edge_indices)
        self.num_select_edges: int = None
        self.fraction: float = None
        self.strategy = getattr(self, self.STRATEGIES[strategy])
        self.selected_edges: list = []

    def select_edges(
        self,
        fraction: float = None,
        num: int = None,
    ):
        '''
        Select a subset of edges from the graph.

        Parameters
        ----------
        fraction : float, optional
            Fraction of total edges to select. Must be in (0, 1].
            Mutually exclusive with num.
        num : int, optional
            Exact number of edges to select. Must be a positive integer
            no greater than the total number of edges.
            Mutually exclusive with fraction.

        Returns
        -------
        list of tuple
            List of selected edge indices as (source, target) tuples.

        Raises
        ------
        ValueError
            If neither or both of fraction and num are provided.
        TypeError
            If num is not an integer, or fraction is not a float.
        ValueError
            If num is not in [1, num_total_edges], or fraction not in (0, 1].
        '''

        # Sanity checks for passed fraction and num of edges
        self._validate_selection_args(fraction, num)

        # Calculate number of edges to select or fraction
        # max(1, ...) to ensure at least one edge is always proposed, preventing
        #   fraction * small edge counts from rounding down to 0 (enables pruning to 0)
        self.num_select_edges = num if num is not None else \
            max(1, round(self.num_total_edges * fraction))
        self.fraction = fraction if fraction is not None else num / self.num_total_edges

        self.selected_edges = self.strategy()

        return self.selected_edges

    def _random_uniform_wo_repl(self):
        '''
        Select num_select_edges edges by uniform random sampling without replacement.

        Returns
        -------
        list of tuple
            Randomly sampled edge indices as (source, target) tuples.
        '''

        return random.sample(self.edge_indices, self.num_select_edges)

    def _validate_graph(self, graph, directed):
        '''
        Validate the graph passed to the class.

        Parameters
        ----------
        graph : any
            Graph object to validate.
        directed : bool
            Whether graph is directed. Used to verify consistency with
            nx.Graph/nx.DiGraph types, and to warn about symmetric np.ndarray inputs.

        Raises
        ------
        TypeError
            If graph is not a nx.Graph or np.ndarray.
        ValueError
            If graph is None.
            If directed=True but an undirected nx.Graph is passed.
            If directed=False but a directed nx.DiGraph is passed.

        Warns
        -----
        UserWarning
            If directed=True and np.ndarray appears symmetric.
            If directed=False and np.ndarray appears asymmetric.
        '''
        if graph is not None:
            if not isinstance(graph, nx.Graph) and not isinstance(graph, np.ndarray):
                raise TypeError('graph must be a networkx graph or np.ndarray')

            if isinstance(graph, nx.Graph):
                if directed and not isinstance(graph, nx.DiGraph):
                    raise ValueError('directed=True but graph is undirected nx.Graph, '
                                     'use nx.DiGraph instead')
                if not directed and isinstance(graph, nx.DiGraph):
                    raise ValueError('directed=False but graph is directed nx.DiGraph, '
                                     'use nx.Graph instead')

            # directed=True but matrix is symmetric
            if directed and isinstance(graph, np.ndarray) and np.array_equal(graph,
                                                                             graph.T):
                warnings.warn('directed graph appears symmetric, verify this is '
                              'intended')

            # directed=False but matrix is asymmetric
            if not directed and isinstance(graph, np.ndarray) and not \
                    np.array_equal(graph, graph.T):
                warnings.warn('directed=False but graph appears asymmetric, verify this'
                              ' is intended')

        else:
            raise ValueError('Graph must be provided')

    def _validate_strategy(self, strategy):
        '''
        Validate the strategy passed to the class.

        Parameters
        ----------
        strategy : str
            Strategy name to validate.

        Raises
        ------
        NotImplementedError
            If strategy is not a key in STRATEGIES.
        '''
        if strategy not in self.STRATEGIES:
            raise NotImplementedError(
                f"Unknown strategy '{strategy}'. "
                "Available strategies: {list(self.STRATEGIES)}"
            )

    def _validate_selection_args(self, fraction, num):
        '''
        Validate the arguments passed to select_edges.

        Parameters
        ----------
        fraction : float or None
            Fraction of edges to select.
        num : int or None
            Exact number of edges to select.

        Raises
        ------
        ValueError
            If neither or both of fraction and num are provided.
            If num is not in [1, num_total_edges].
            If fraction is not in (0, 1].
        TypeError
            If num is not an integer.
            If fraction is not a float.
        '''

        # Check that either fraction or num is provided
        if fraction is None and num is None:
            raise ValueError('Provide either fraction or num, not neither')

        if fraction is not None and num is not None:
            raise ValueError('Provide either fraction or num, not both')

        # Check that selected num is an integer and not larger than total amount of
        #   edges in graph
        if num is not None:
            if not isinstance(num, int):
                raise TypeError('num must be an integer')
            if not (0 < num <= self.num_total_edges):
                raise ValueError(f'num must be between 1 and {self.num_total_edges}')

        # Check that selected fraction is a float and larger than zero but not larger
        #   than 1
        if fraction is not None:
            if not isinstance(fraction, float):
                raise TypeError('fraction must be a float')
            if not (0.0 < fraction <= 1.0):
                raise ValueError('fraction must be in (0, 1]')

    def _extract_edges(self, graph: nx.Graph | np.ndarray, directed) -> list[tuple]:
        '''
        Extract edge indices and shape information from a graph.

        Parameters
        ----------
        graph : nx.Graph or np.ndarray
            Graph to extract edges from.

        Returns
        -------
        edge_indices : list of tuple
            List of (source, target) tuples representing all edges.
        '''
        # Get the edges in the graph and the shape of the graph
        # Prunable edges in graph
        if isinstance(graph, nx.Graph):
            edge_indices = list(graph.edges())
            return edge_indices
        elif isinstance(graph, np.ndarray):
            rows, cols = np.where(graph != 0)  # where entries are not zero
            edge_indices = list(zip(rows, cols))
            if not directed:
                edge_indices = [(r, c) for r, c in edge_indices if r < c]
            return edge_indices


if __name__ == "__main__":

    # Create a sample graph
    G = nx.erdos_renyi_graph(10, 0.5, directed=True)
    print(G.edges())        # all edges
    print(G.number_of_edges())  # total edge count
    print(G.number_of_nodes())  # total node count
    # Graphs edges
    print(f'Possible edges: {G.edges()}')
    # Select random edges
    selector = EdgeSelector(strategy="random_uniform_wo_repl", graph=G)
    random_edges = selector.select_edges(num=4)
    print(f'Randomly selected edges: {random_edges}')

    # Create a sample graph
    G = nx.erdos_renyi_graph(10, 0.5, directed=True)
    # Graphs edges
    print(f'Possible edges: {G.edges()}')
    # Select random edges
    selector = EdgeSelector(strategy="random_uniform_wo_repl", graph=G)
    random_edges = selector.select_edges(fraction=0.5)
    print(f'Randomly selected edges: {random_edges}')
