r"""
Named features of a reservoir network and of a single edge.

A *feature* is a function that returns one number. Two kinds exist:

* a **network feature** describes a reservoir computer (its reservoir network, or the activity of its nodes
  on data): ``f(ctx) -> float``, with ``ctx`` a :class:`NetworkContext`;
* an **edge feature** describes one edge :math:`u \to v` of a reservoir computer: ``f(ctx, u, v) -> float``,
  with ``ctx`` an :class:`EdgeContext`.

Every feature has a unique name. :class:`pyreco.edge_pruning.EdgePruner` evaluates the features that you
ask for (``EdgePruner(features=[...])``) for every candidate of a pruning iteration and stores the numbers
under their names, instead of the candidate reservoirs themselves.

The context offers what a feature may need (the weight matrix, the state of the nodes on the data, a networkx
graph) and computes each of these only when a feature asks for it, once, and shares it between the features.

**Writing your own feature** (also in another package): decorate a function with :func:`network_feature` or
:func:`edge_feature` and give it a name. Then use the name, or the function, in the ``features`` list::

    from pyreco.features import network_feature, edge_feature

    @network_feature("frobenius_norm")
    def frobenius_norm(ctx):
        '''Frobenius norm of the reservoir matrix.'''       # the first line is the description
        return np.linalg.norm(ctx.weights)

    @edge_feature("edge_signal_power")
    def edge_signal_power(ctx, u, v):
        '''Mean square of the signal that the edge delivers.'''
        return ctx.weights[u, v] ** 2 * ctx.node_mean_sq_activity[u]

    EdgePruner(features=["density", "frobenius_norm", "edge_signal_power"])

:func:`available_network_features` and :func:`available_edge_features` list the names and descriptions.
"""

from functools import cached_property
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from pyreco.utils_networks import (
    convert_to_nx_graph,
    extract_av_in_degree,
    extract_av_out_degree,
    extract_clustering_coefficient,
    extract_density,
    extract_spectral_radius,
)

# name -> function, one registry per level
_NETWORK_FEATURES: Dict[str, Callable] = {}
_EDGE_FEATURES: Dict[str, Callable] = {}

#: Features that ``EdgePruner`` computes if ``features`` is not given: cheap, no pass through the data.
DEFAULT_NETWORK_FEATURES = [
    "density",
    "spectral_radius",
    "av_in_degree",
    "av_out_degree",
    "clustering_coefficient",
]
DEFAULT_EDGE_FEATURES = [
    "weight",
    "is_reciprocal",
    "in_scc",
    "betweenness",
    "source_out_degree",
    "target_in_degree",
]


# --------------------------------------------------------------------------------------------- contexts


class NetworkContext:
    """
    What a network feature may look at: one reservoir computer and the data to run it on.

    Parameters
    ----------
    model : RC
        A fitted reservoir computer.
    x_train : ndarray of shape (n_batch, n_timesteps, n_features)
        The input data on which state-dependent features are measured (use the training inputs).

    Attributes
    ----------
    model : RC
    x_train : ndarray
    weights : ndarray
        Reservoir matrix, ``weights[u, v]`` is the edge from node ``u`` to node ``v``.
    node_mean_sq_activity : ndarray of shape (N,)
        Mean squared activity :math:`\\langle r_v^2 \\rangle` of every node, see :func:`mean_sq_activity`.
        Needs one pass of the model through ``x_train``, done on first use.
    """

    def __init__(self, model, x_train: np.ndarray):
        self.model = model
        self.x_train = x_train

    @property
    def weights(self) -> np.ndarray:
        return self.model.reservoir_layer.weights

    @cached_property
    def node_mean_sq_activity(self) -> np.ndarray:
        model = self.model
        states = model.compute_reservoir_state(model._transform_x(self.x_train))  # (batch, T + 1, N)
        states = states[:, model.num_transients_to_remove + 1:]  # the states that the readout sees
        if states.shape[1] == 0:
            raise ValueError(
                f"x_train has {self.x_train.shape[1]} time steps, but the warm-up is "
                f"{model.num_transients_to_remove}: no state is left to measure the activity."
            )
        return (states ** 2).mean(axis=(0, 1))


class EdgeContext(NetworkContext):
    """
    What an edge feature may look at: a :class:`NetworkContext` of the reservoir computer in which the edge
    exists, shared by all edges of one pruning iteration.

    Attributes
    ----------
    graph : nx.DiGraph
        The reservoir network as a networkx graph.
    scc_map : dict
        Node -> id of its strongly connected component.
    edge_betweenness, node_betweenness : dict
        Betweenness centrality of the edges ``(u, v)`` and of the nodes (unweighted, as in networkx).
    """

    @cached_property
    def graph(self) -> nx.DiGraph:
        return convert_to_nx_graph(self.weights)

    @cached_property
    def scc_map(self) -> dict:
        return {node: i for i, comp in enumerate(nx.strongly_connected_components(self.graph)) for node in comp}

    @cached_property
    def edge_betweenness(self) -> dict:
        return nx.edge_betweenness_centrality(self.graph)

    @cached_property
    def node_betweenness(self) -> dict:
        return nx.betweenness_centrality(self.graph)


# ----------------------------------------------------------------------------------------- registration


def _register(level: str, registry: Dict[str, Callable], name: str):
    if not isinstance(name, str) or not name:
        raise TypeError(f"The name of a feature must be a non-empty string, got {name!r}.")

    def decorator(func: Callable) -> Callable:
        for other in (_NETWORK_FEATURES, _EDGE_FEATURES):
            known = other.get(name)
            # redefining a function with the same name and place (re-running a notebook cell) is fine
            same = known is not None and (known.__module__, known.__qualname__) == (func.__module__, func.__qualname__)
            if known is not None and not same:
                raise ValueError(
                    f"A feature named {name!r} exists already ({known.feature_level} feature "
                    f"{known.__module__}.{known.__qualname__}). Choose another name."
                )
            if same and other is not registry:
                other.pop(name)
        func.feature_name = name
        func.feature_level = level
        registry[name] = func
        return func

    return decorator


def network_feature(name: str):
    """Decorator that registers ``f(ctx) -> float`` as the network feature ``name`` (see the module docstring)."""
    return _register("network", _NETWORK_FEATURES, name)


def edge_feature(name: str):
    """Decorator that registers ``f(ctx, u, v) -> float`` as the edge feature ``name`` (see the module docstring)."""
    return _register("edge", _EDGE_FEATURES, name)


def _describe(registry: Dict[str, Callable]) -> Dict[str, str]:
    return {name: (func.__doc__ or "").strip().split("\n")[0] for name, func in registry.items()}


def available_network_features() -> Dict[str, str]:
    """Names and one-line descriptions of the registered network features."""
    return _describe(_NETWORK_FEATURES)


def available_edge_features() -> Dict[str, str]:
    """Names and one-line descriptions of the registered edge features."""
    return _describe(_EDGE_FEATURES)


def resolve_features(features: Optional[Iterable[Union[str, Callable]]] = None) -> Tuple[Dict, Dict]:
    """
    Turn a list of feature names and decorated functions into the functions to call.

    Parameters
    ----------
    features : list of str or callable, optional
        Registered names, or functions decorated with :func:`network_feature` / :func:`edge_feature`, network
        and edge features mixed. None selects ``DEFAULT_NETWORK_FEATURES`` and ``DEFAULT_EDGE_FEATURES``.
        An empty list selects nothing.

    Returns
    -------
    network, edge : dict
        Name -> function for the network features and for the edge features, in the order given.

    Raises
    ------
    TypeError
        For an entry that is neither a name nor a decorated function, or ``features`` that is not a list.
    ValueError
        For an unknown name or a name that is listed twice.
    """
    if features is None:
        features = DEFAULT_NETWORK_FEATURES + DEFAULT_EDGE_FEATURES
    if isinstance(features, (str, bytes)) or not isinstance(features, Iterable):
        raise TypeError(f"features must be a list of names or decorated functions, got {features!r}.")

    network, edge = {}, {}
    for entry in features:
        if isinstance(entry, str):
            func = _NETWORK_FEATURES.get(entry) or _EDGE_FEATURES.get(entry)
            if func is None:
                raise ValueError(
                    f"Unknown feature {entry!r}. Network features: {sorted(_NETWORK_FEATURES)}. "
                    f"Edge features: {sorted(_EDGE_FEATURES)}."
                )
        elif callable(entry) and hasattr(entry, "feature_level"):
            func = entry
        else:
            raise TypeError(
                f"{entry!r} is neither the name of a feature nor a function decorated with "
                "@network_feature(name) or @edge_feature(name)."
            )
        target = network if func.feature_level == "network" else edge
        if func.feature_name in network or func.feature_name in edge:
            raise ValueError(f"The feature {func.feature_name!r} is listed twice.")
        target[func.feature_name] = func
    return network, edge


# ------------------------------------------------------------------------------------------ evaluation


def _as_float(name: str, value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise TypeError(f"The feature {name!r} must return one number, it returned an array of shape {array.shape}.")
    return float(array.reshape(()))


def evaluate_network_features(network: Dict[str, Callable], ctx: NetworkContext) -> Dict[str, float]:
    """Call the (resolved) network features: ``{name: float}``."""
    return {name: _as_float(name, func(ctx)) for name, func in network.items()}


def evaluate_edge_features(edge: Dict[str, Callable], ctx: EdgeContext, edges: List[tuple]) -> Dict[str, np.ndarray]:
    """Call the (resolved) edge features for every edge: ``{name: array of shape (len(edges),)}``."""
    return {
        name: np.array([_as_float(name, func(ctx, int(u), int(v))) for u, v in edges], dtype=float)
        for name, func in edge.items()
    }


def compute_network_features(model, x_train: np.ndarray, features=None) -> Dict[str, float]:
    """
    Network features of a fitted reservoir computer.

    Parameters
    ----------
    model : RC
        Fitted model.
    x_train : ndarray of shape (n_batch, n_timesteps, n_features)
        Data for the state-dependent features.
    features : list of str or callable, optional
        As in :func:`resolve_features`; only the network features of the list are computed. Default: the
        default network features.

    Returns
    -------
    dict
        Feature name -> float.
    """
    network, _ = resolve_features(DEFAULT_NETWORK_FEATURES if features is None else features)
    return evaluate_network_features(network, NetworkContext(model, x_train))


def compute_edge_features(model, x_train: np.ndarray, edges: List[tuple], features=None) -> Dict[str, np.ndarray]:
    """
    Edge features of edges of a fitted reservoir computer.

    Parameters
    ----------
    model : RC
        Fitted model, the reservoir in which the edges exist.
    x_train : ndarray of shape (n_batch, n_timesteps, n_features)
        Data for the state-dependent features.
    edges : list of tuple
        The edges ``(u, v)``.
    features : list of str or callable, optional
        As in :func:`resolve_features`; only the edge features of the list are computed. Default: the default
        edge features.

    Returns
    -------
    dict
        Feature name -> array with one value per edge, in the order of ``edges``.
    """
    _, edge = resolve_features(DEFAULT_EDGE_FEATURES if features is None else features)
    return evaluate_edge_features(edge, EdgeContext(model, x_train), edges)


# ------------------------------------------------------------------------------------ network features


@network_feature("density")
def _density(ctx):
    """Fraction of the N (N - 1) possible directed edges that exist."""
    return extract_density(ctx.weights)


@network_feature("spectral_radius")
def _spectral_radius(ctx):
    """Largest absolute eigenvalue of the reservoir matrix."""
    return extract_spectral_radius(ctx.weights)


@network_feature("av_in_degree")
def _av_in_degree(ctx):
    """Mean number of incoming edges per node."""
    return extract_av_in_degree(ctx.weights)


@network_feature("av_out_degree")
def _av_out_degree(ctx):
    """Mean number of outgoing edges per node."""
    return extract_av_out_degree(ctx.weights)


@network_feature("clustering_coefficient")
def _clustering_coefficient(ctx):
    """Average directed clustering coefficient (Fagiolo 2007)."""
    return extract_clustering_coefficient(ctx.weights)


@network_feature("mean_sq_activity")
def mean_sq_activity(ctx):
    r"""
    Mean squared activity of the nodes, averaged over the nodes.

    :math:`\frac{1}{N}\sum_v \langle r_v^2 \rangle`, with :math:`\langle\cdot\rangle` the average over the
    samples and over the time steps after the warm-up (the states that the readout sees). The activity is the
    second moment of the node states, not centered. For ``tanh`` it lies in [0, 1): near 0 the nodes are
    barely driven, near 1 they sit in saturation. Measured on ``x_train``, so it needs one pass through the data.
    """
    return ctx.node_mean_sq_activity.mean()


# --------------------------------------------------------------------------------------- edge features


@edge_feature("weight")
def _weight(ctx, u, v):
    """Weight of the edge (signed)."""
    return ctx.weights[u, v]


@edge_feature("abs_weight")
def _abs_weight(ctx, u, v):
    """Absolute weight of the edge."""
    return abs(ctx.weights[u, v])


@edge_feature("sign")
def _sign(ctx, u, v):
    """-1 for a negative weight, +1 otherwise."""
    return -1.0 if ctx.weights[u, v] < 0 else 1.0


@edge_feature("is_reciprocal")
def _is_reciprocal(ctx, u, v):
    """1 if the opposite edge v -> u exists, else 0."""
    return float(ctx.weights[v, u] != 0)


@edge_feature("in_scc")
def _in_scc(ctx, u, v):
    """1 if u and v are in the same strongly connected component (the edge is on a cycle), else 0."""
    return float(ctx.scc_map[u] == ctx.scc_map[v])


@edge_feature("betweenness")
def _betweenness(ctx, u, v):
    """Betweenness centrality of the edge (share of shortest paths that use it)."""
    return ctx.edge_betweenness.get((u, v), 0.0)


@edge_feature("source_out_degree")
def _source_out_degree(ctx, u, v):
    """Number of outgoing edges of the sending node u."""
    return np.count_nonzero(ctx.weights[u])


@edge_feature("target_in_degree")
def _target_in_degree(ctx, u, v):
    """Number of incoming edges of the receiving node v."""
    return np.count_nonzero(ctx.weights[:, v])


@edge_feature("source_betweenness")
def _source_betweenness(ctx, u, v):
    """Betweenness centrality of the sending node u."""
    return ctx.node_betweenness[u]


@edge_feature("target_betweenness")
def _target_betweenness(ctx, u, v):
    """Betweenness centrality of the receiving node v."""
    return ctx.node_betweenness[v]


@edge_feature("source_mean_sq_activity")
def _source_mean_sq_activity(ctx, u, v):
    """Mean squared activity of the sending node u (see ``mean_sq_activity``); needs a pass through the data."""
    return ctx.node_mean_sq_activity[u]


@edge_feature("target_mean_sq_activity")
def _target_mean_sq_activity(ctx, u, v):
    """Mean squared activity of the receiving node v (see ``mean_sq_activity``); needs a pass through the data."""
    return ctx.node_mean_sq_activity[v]
