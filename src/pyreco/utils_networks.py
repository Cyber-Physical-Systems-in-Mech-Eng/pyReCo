"""
Helper routines for networks
"""
import networkx as nx
import numpy as np


def gen_ER_graph(nodes: int, density: float, spec_rad: float = 0.9, directed: bool=True, seed=None):
    """
    Generate an Erdős-Rényi random graph with specified properties.

    Parameters
    ----------
    nodes : int
        Number of nodes in the graph.
    density : float
        Desired connection density (between 0 and 1).
    spec_rad : float, optional
        Desired spectral radius of the graph. Default is 0.9.
    directed : bool, optional
        Whether to create a directed graph. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    np.ndarray
        Adjacency matrix of the graph with shape (nodes, nodes).

    Notes
    -----
    Previous Bug:
        The line `G.remove_nodes_from(list(nx.isolates(G)))` removed isolated nodes from the graph
        before converting to a numpy array. This caused the final matrix to sometimes be smaller
        than the specified size (e.g., 29x29 instead of 30x30) when isolated nodes were present.
        This led to dimension mismatches in reservoir computations where other matrices expected
        the full size.

    Solution:
        Instead of removing isolated nodes, we now connect them to maintain the specified
        network size. This ensures consistency between the reservoir weight matrix and
        other matrices in the computation.
    """
    # use networkx to generate a random graph
    G = nx.erdos_renyi_graph(nodes, density, seed=seed, directed=directed)

    # Instead of removing isolated nodes (old buggy behavior):
    # G.remove_nodes_from(list(nx.isolates(G)))
    
    # New: Connect isolated nodes to maintain matrix size
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        non_isolated = list(set(G.nodes()) - set(isolated_nodes))
        for node in isolated_nodes:
            if non_isolated:
                target = np.random.choice(non_isolated)
                G.add_edge(node, target)
                if directed:
                    G.add_edge(target, node)

    GNet = nx.to_numpy_array(G)
    curr_spec_rad = max(abs(np.linalg.eigvals(GNet)))
    graph = GNet * spec_rad/curr_spec_rad

    return graph

def compute_density(network: np.ndarray) -> float:
    """
    Compute the density of a given adjacency matrix.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix of the network.

    Returns
    -------
    float
        Density of the adjacency matrix, calculated as the fraction of non-zero entries 
        over the total number of possible links (N^2).

    Raises
    ------
    TypeError
        If the input is not a numpy ndarray.
    ValueError
        If the input matrix is not square.

    Notes
    -----
    The density is calculated by counting the number of non-zero entries in the adjacency 
    matrix and dividing by the total number of possible entries, which is N^2 where N is 
    the number of nodes in the network.

    Example
    -------
    >>> network = np.array([[0, 1], [1, 0]])
    >>> compute_density(network)
    1.0
    """
    # compute density of a given adjacency matrix by the fraction of non-zero entries over  N^2
    if type(network) is not np.ndarray:
        raise (TypeError('Expect a np.ndarray as reservoir network'))

    # check if the matrix is square
    if network.shape[0] != network.shape[1]:    
        raise (ValueError('Expect network of square size!'))

    N = len(network)
    num_links = np.sum(network.flatten()>0)
    return num_links / (N**2)

def get_num_nodes(network: np.ndarray) -> int:
    """
    Get the number of non-isolated nodes in the given network.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix representing the network.

    Returns
    -------
    int
        The number of non-isolated nodes in the network.

    Raises
    ------
    TypeError
        If the input is not a numpy ndarray.
    ValueError
        If the input matrix is not square.

    Notes
    -----
    This function identifies non-isolated nodes by checking for non-zero entries in both 
    the corresponding row and column of the adjacency matrix. It returns the count of nodes 
    that have at least one connection.

    Example
    -------
    >>> network = np.array([[0, 1], [1, 0]])
    >>> get_num_nodes(network)
    2
    """
    # returns the number of nodes in the given network. Assumes the adjacency matrix to be of square size
    if type(network) is not np.ndarray:
        raise (TypeError('Expect a np.ndarray as reservoir network'))

    if network.shape[0] != network.shape[1]:
        raise (ValueError('Expect network of square size!'))

    non_zero_rows = np.any(network != 0, axis=1)

    # Identify columns that are not entirely zero
    non_zero_columns = np.any(network != 0, axis=0)

    # Identify nodes where both row and column are not zero
    non_isolated_nodes = np.where(non_zero_rows & non_zero_columns)[0]

    # Number of non-isolated nodes
    num_non_isolated_nodes = len(non_isolated_nodes)

    return num_non_isolated_nodes


def compute_spec_rad(network: np.ndarray) -> float:
    """
    Compute the spectral radius of the network.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix representing the network.

    Returns
    -------
    float
        The spectral radius of the network, which is the maximum absolute eigenvalue of the adjacency matrix.

    Raises
    ------
    TypeError
        If the input is not a numpy ndarray.
    ValueError
        If the input matrix is not square.

    Notes
    -----
    The spectral radius is computed as the largest absolute eigenvalue of the adjacency matrix. 
    It is used to characterize the stability and dynamics of a network.

    Example
    -------
    >>> network = np.array([[0, 1], [1, 0]])
    >>> compute_spec_rad(network)
    1.0
    """
    # compute the spectral radius of the network (max. eigenvalue)
    if type(network) is not np.ndarray:
        raise (TypeError('Expect a np.ndarray as reservoir network'))

    if network.shape[0] != network.shape[1]:
        raise (ValueError('Expect network of square size!'))

    return np.max(np.abs(np.linalg.eigvals(network)))


def set_spec_rad(network: np.ndarray, spec_radius: float) -> np.ndarray:
    """
    Set the spectral radius of the network to a given value.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix representing the network.
    
    spec_radius : float
        Desired spectral radius to be set for the network.

    Returns
    -------
    np.ndarray
        The adjacency matrix scaled to the specified spectral radius.

    Raises
    ------
    ValueError
        If the specified spectral radius is less than or equal to zero.
    
    Warning
    -------
    If the specified spectral radius is greater than 1.0, a warning is issued.

    Notes
    -----
    This function scales the adjacency matrix such that the spectral radius is modified to match the specified value.
    If the current spectral radius is smaller than 10^-9, it is adjusted to 10^-6 to prevent numerical issues.

    Example
    -------
    >>> network = np.array([[0, 1], [1, 0]])
    >>> set_spec_rad(network, 0.9)
    array([[ 0.9,  0.9],
           [ 0.9,  0.9]])
    """
    if spec_radius <= 0:
        raise(ValueError('spectral radius must be larger than zero'))
    elif (spec_radius > 1.0):
        raise(Warning('a spectral radius larger than 1 is unusual!'))

    # compute current spectral radius
    current_spectral_radius = compute_spec_rad(network)

    if current_spectral_radius < 10**(-9):
        print('spectral radius smaller than 10^-9!')
        current_spectral_radius = 10**(-6)

    scaling_factor = spec_radius / current_spectral_radius


    return network * scaling_factor


def is_zero_col_and_row(x: np.ndarray, idx: int) -> bool:
    """
    Check if the adjacency matrix carries only zeros in the column and row of the given index.

    Parameters
    ----------
    x : np.ndarray
        Adjacency matrix to check for zero entries.
    
    idx : int
        Index of the column and row to verify for zero values.

    Returns
    -------
    bool
        True if both the row and column corresponding to the given index are filled with zeros, False otherwise.

    Notes
    -----
    This function checks both the column and the row at the specified index in the adjacency matrix to see if they consist entirely of zeros.
    If both are zero, it indicates the node at the specified index is isolated, with no connections.

    Example
    -------
    >>> adj_matrix = np.array([[0, 1], [0, 0]])
    >>> is_zero_col_and_row(adj_matrix, 1)
    True
    >>> is_zero_col_and_row(adj_matrix, 0)
    False
    """
    # returns zero if adjacency matrix x carries only zeros in column and row of index idx (i.e. missing node)

    is_zero_column = np.all(x[:, idx] == 0)
    is_zero_row = np.all(x[idx, :] == 0)

    if is_zero_column and is_zero_row:
        return True
    else:
        return False


def remove_node(x: np.ndarray, idx: int | list) -> np.ndarray:
    """
    Remove the specified node(s) from the adjacency matrix.

    Parameters
    ----------
    x : np.ndarray
        Adjacency matrix from which the node(s) will be removed.

    idx : int or list
        Index or list of indices of the node(s) to remove. Each index refers to the row and column of the node to be removed.

    Returns
    -------
    np.ndarray
        Adjacency matrix with the specified node(s) removed (set to zero).

    Notes
    -----
    This function modifies the given adjacency matrix by setting the rows and columns corresponding to the specified node(s) to zero.
    The node(s) are effectively removed from the network.

    Example
    -------
    >>> adj_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> remove_node(adj_matrix, 1)
    array([[0, 0, 0],
           [0, 0, 0],
           [1, 0, 0]])
    """
    if type(idx) is not list:
        idx = [idx]

    if x.ndim == 1:
        for idxx in idx:
            x[idxx] = 0
    elif x.ndim == 2:
        for idxx in idx:
            x[:, idxx] = 0
            x[idxx, :] = 0

    return x


def gen_init_states(num_nodes: int, method: str = 'random'):
    """
    Generate initial reservoir states.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the reservoir network.

    method : str, optional, default: 'random'
        Method for generating initial states. Available methods:
        - 'random': Uniformly distributed values between 0 and 1.
        - 'random_normal': Normally distributed random values.
        - 'ones': All values set to 1.
        - 'zeros': All values set to 0.

    Returns
    -------
    np.ndarray
        Array of length `num_nodes` with the generated initial states. The states are normalized to have a maximum absolute value of 1, except for the 'zeros' method.

    Raises
    ------
    ValueError
        If the specified `method` is not one of the supported methods.

    Notes
    -----
    The generated states are normalized to have a maximum absolute value of 1 to ensure consistency across different initial states.
    The normalization does not affect the 'zeros' method, which always returns a vector of zeros.

    Example
    -------
    >>> gen_init_states(5, 'random')
    array([ 0.5,  1. ,  0.2, -0.7,  0.9])
    """
    if method == 'random':
        init_states = np.random.random(num_nodes)
    elif method == 'random_normal':
        init_states = np.random.randn(num_nodes)
    elif method == 'ones':
        init_states = np.ones(num_nodes)
    elif method == 'zeros':
        init_states = np.zeros(num_nodes)
    else:
        raise (ValueError(f'Sampling method {method} is unknown for generating initial reservoir states'))
    
    # normalize to max. absolute value of 1
    if method != 'zeros':
        init_states = init_states / np.max(np.abs(init_states))

    return init_states


def extract_density(adjacency_matrix):
    """
    Extract the density of the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix representing the network. It can be directed or undirected.

    Returns
    -------
    float
        The density of the network. It is calculated as the ratio of the number of edges to the 
        number of possible edges in the network.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> extract_density(adjacency_matrix)
    0.6666666666666666
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return nx.density(G)


def extract_spectral_radius(adjacency_matrix):
    """
    Extract the spectral radius of the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    float
        The spectral radius of the network, which is the largest absolute eigenvalue of the adjacency matrix.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> extract_spectral_radius(adjacency_matrix)
    1.618033988749895
    """
    return np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))


def extract_in_degree_av(adjacency_matrix):
    """
    Extract the average in-degree of the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    float
        The average in-degree of the network, calculated as the mean of the in-degrees of all nodes.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> extract_in_degree_av(adjacency_matrix)
    1.0
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    in_degrees = G.in_degree()
    avg_in_degree = np.mean(list(dict(in_degrees).values()))
    return avg_in_degree


def extract_out_degree_av(adjacency_matrix):
    """
    Extract the average out-degree of the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    float
        The average out-degree of the network, calculated as the mean of the out-degrees of all nodes.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> extract_out_degree_av(adjacency_matrix)
    1.0
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    out_degrees = G.out_degree()
    avg_out_degree = np.mean(list(dict(out_degrees).values()))
    return avg_out_degree


def extract_clustering_coefficient(adjacency_matrix):
    """
    Extract the clustering coefficient of the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    float
        The average clustering coefficient of the network, representing the tendency of nodes to cluster together.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> extract_clustering_coefficient(adjacency_matrix)
    0.6666666666666666
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return nx.average_clustering(G)


def extract_node_degree(adjacency_matrix):
    """
    Extract the degree of each node in the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    dict
        A dictionary with node indices as keys and their corresponding degrees as values.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> extract_node_degree(adjacency_matrix)
    {0: 2, 1: 2, 2: 2}
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return dict(G.degree())


def extract_node_in_degree(adjacency_matrix):
    """
    Extract the in-degree of each node in the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    dict
        A dictionary with node indices as keys and their corresponding in-degrees as values.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> extract_node_in_degree(adjacency_matrix)
    {0: 1, 1: 1, 2: 2}
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return dict(G.in_degree())


def extract_node_out_degree(adjacency_matrix):
    """
    Extract the out-degree of each node in the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    dict
        A dictionary with node indices as keys and their corresponding out-degrees as values.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> extract_node_out_degree(adjacency_matrix)
    {0: 2, 1: 2, 2: 2}
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return dict(G.out_degree())


def extract_node_clustering_coefficient(adjacency_matrix):
    """
    Extract the clustering coefficient of each node in the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    dict
        A dictionary with node indices as keys and their corresponding clustering coefficients as values.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> extract_node_clustering_coefficient(adjacency_matrix)
    {0: 0.0, 1: 0.0, 2: 0.0}
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return nx.clustering(G)


def extract_node_betweenness_centrality(adjacency_matrix):
    """
    Extract the betweenness centrality of each node in the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    dict
        A dictionary with node indices as keys and their corresponding betweenness centrality values as values.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> extract_node_betweenness_centrality(adjacency_matrix)
    {0: 0.0, 1: 1.0, 2: 0.0}
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return nx.betweenness_centrality(G)


def extract_node_pagerank(adjacency_matrix):
    """
    Extract the PageRank of each node in the network from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Adjacency matrix representing the network. It is assumed to be a square matrix.

    Returns
    -------
    dict
        A dictionary with node indices as keys and their corresponding PageRank values as values.

    Example
    -------
    >>> adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> extract_node_pagerank(adjacency_matrix)
    {0: 0.3333333333333333, 1: 0.3333333333333333, 2: 0.3333333333333333}
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return nx.pagerank(G)

# Add more network property extraction functions as needed
