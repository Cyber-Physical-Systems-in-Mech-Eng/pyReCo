from typing import Union
import numpy as np
import networkx as nx

from pyreco.utils_networks import (
    convert_to_nx_graph,
    precompute_edge_metrics,
    extract_edge_weight,
    extract_edge_abs_weight,
    extract_edge_sign,
    extract_edge_is_reciprocal,
    extract_edge_in_scc,
    extract_edge_betweenness,
    extract_edge_source_out_degree,
    extract_edge_target_in_degree,
    extract_edge_source_betweenness,
    extract_edge_target_betweenness,
)


def available_extractors():
    """
    Returns a dictionary mapping edge property names to their extractor functions.
    Each function has signature f(graph, edge) -> scalar, where edge is a (u, v) tuple.
    Make changes here to add more edge properties.

    Returns
    -------
    dict
        mapping property name -> extractor function
    """
    return {
        "weight": extract_edge_weight,
        "abs_weight": extract_edge_abs_weight,
        "sign": extract_edge_sign,
        "is_reciprocal": extract_edge_is_reciprocal,
        "in_scc": extract_edge_in_scc,
        "betweenness": extract_edge_betweenness,
        "source_out_degree": extract_edge_source_out_degree,
        "target_in_degree": extract_edge_target_in_degree,
        "source_betweenness": extract_edge_source_betweenness,
        "target_betweenness": extract_edge_target_betweenness,
    }


def map_extractor_names(prop_names: list):
    """
    Return extractors for the requested property names.

    Returns
    -------
    dict
        Dict mapping property name to extractor function
    list
        List of extractor functions in same order
    """
    # Define dictionary mapping network property names to their corresponding
    #   extractor functions
    all_extractors = available_extractors()
    extractor_dict = {}
    extractor_funs = []
    for prop in prop_names:
        if prop not in all_extractors:
            print(f"Warning: {prop} is not a recognized edge property.")
        else:
            extractor_dict[prop] = all_extractors[prop]
            extractor_funs.append(all_extractors[prop])
    return extractor_dict, extractor_funs


class EdgeAnalyzer:
    """
    A class for analyzing properties of a specific edge in a graph,
    analogue to 'NodeAnalyzer for nodes.

    Takes a graph and a specific edge (u, v), returning a flat dict of scalars
    describing that edge and the nodes it connects.

    Note: betweenness-based properties (betweenness, source_betweenness,
    target_betweenness) each trigger a full graph betweenness computation.
    If all three are requested, consider passing a subset via quantities to
    avoid redundant computation on large graphs.
    """

    def __init__(self, quantities=None):
        """
        Initializes the EdgeAnalyzer with specified quantities to extract.

        Parameters
        ----------
        quantities : list, optional
            List of edge properties to extract. Defaults to all that are available at the moment
            (see :func:`available_extractors`): ``'abs_weight'``, ``'sign'``, ``'is_reciprocal'``,
            ``'in_scc'``, ``'betweenness'``, ``'source_out_degree'``, ``'target_in_degree'``,
            ``'source_betweenness'`` and ``'target_betweenness'``.
        """
        all_extractors = available_extractors().keys()
        self.quantities = quantities or list(all_extractors)
        self.extractors, self.extractor_funs = map_extractor_names(self.quantities)

    def extract_properties(
        self, graph: Union[nx.Graph, nx.DiGraph, np.ndarray], edge: tuple
    ) -> dict:
        """
        Extracts the specified properties for a given edge.

        Parameters
        ----------
        graph : nx.Graph, nx.DiGraph, or np.ndarray
            Graph to analyze.
        edge : tuple
            Edge (u, v) to extract properties for.

        Returns
        -------
        dict
            Dictionary containing the extracted edge properties.
        """
        # Check edge is tuple list with two entries specifiying edge
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("edge must be a (u, v) tuple")

        #  dict for properties
        edge_props = {}
        # Get property name and functions to store in dict
        for extr_name, extr_fun in self.extractors.items():
            edge_props[extr_name] = extr_fun(graph, edge)
        return edge_props

    def extract_properties_batch(
        self, graph: Union[nx.Graph, nx.DiGraph, np.ndarray], edges: list
    ) -> list:
        """
        Extracts properties for a list of edges with expensive graph-level metrics
        (betweenness centrality, SCC) so that they're computed only once.

        Parameters
        ----------
        graph : nx.Graph, nx.DiGraph, or np.ndarray
            Graph in its current state, e.g. before edge removal.
        edges : list
            List of edges, (u, v) tuples, to extract properties for.

        Returns
        -------
        list
            List of dicts containing the extracted edge properties.
            One per edge, in the same order as edges.
        """
        # Convert here already to use networkx functions
        g = convert_to_nx_graph(graph)

        # Pre compute node_betweenness, edge_betweenness and scc_map on one graph
        cache = precompute_edge_metrics(g)

        # Create look up table, basically just functions to look up edge properties
        #   from pre compute
        _cached = {
            "weight": lambda u, v: g[u][v].get("weight", 1.0),
            "abs_weight": lambda u, v: abs(g[u][v].get("weight", 1.0)),
            "sign": lambda u, v: -1 if g[u][v].get("weight", 1.0) < 0 else 1,
            "is_reciprocal": lambda u, v: int(g.has_edge(v, u)),
            "in_scc": lambda u, v: int(cache["scc_map"][u] == cache["scc_map"][v]),
            "betweenness": lambda u, v: cache["edge_betweenness"].get((u, v), 0.0),
            "source_out_degree": lambda u, v: g.out_degree[u],
            "target_in_degree": lambda u, v: g.in_degree[v],
            "source_betweenness": lambda u, v: cache["node_betweenness"][u],
            "target_betweenness": lambda u, v: cache["node_betweenness"][v],
        }

        results = []
        for edge in edges:
            # Go through edges
            u, v = int(edge[0]), int(edge[1])
            # Look up values for edge for specified properties
            props = {name: _cached[name](u, v) for name in self.quantities if name in
                     _cached}
            # Append properties for edge
            results.append(props)

        # Return list with edge properties for all edges
        return results

    def list_properties(self):
        """
        Returns list of available edge properties.

        Returns
        -------
        list
            Available edge property names.
        """
        return list(available_extractors().keys())


def operating_point_statistics(model, x: np.ndarray, discard: int = None, saturation_threshold: float = 0.1) -> dict:
    r"""
    State-dependent statistics of the nodes and edges of a fitted reservoir computer, measured on data.

    Structural quantifiers such as the in- and out-degree describe the wiring only. Whether an edge
    matters also depends on where the receiving node sits on its activation curve, and the
    fixed random input bias moves the nodes to different places. This function runs the reservoir
    on ``x`` and measures that. It is a read-only diagnostic: the model is not modified.

    **Computation.** Let :math:`\tilde x(t)` be the preprocessed input (as in ``fit``), :math:`A` the
    reservoir matrix with :math:`A_{uv}` the edge :math:`u \to v`, and :math:`\beta b` the input bias.
    The pre-activation of node :math:`v` at time :math:`t` is

    .. math::

        z_v(t) = \sum_u r_u(t)\, A_{uv} + \sum_k \tilde x_k(t)\, W_{\mathrm{in},vk} + \beta\, b_v ,
        \qquad t = d, \dots, T-1 ,

    where :math:`d` is the warm-up. All statistics pool the samples of the batch and the time steps
    :math:`t \ge d`; :math:`\langle\cdot\rangle` denotes this average and :math:`\mathrm{Var}` the variance.
    With the slope :math:`g'(z)` of the activation (:math:`1 - \tanh^2 z`, or :math:`g(1 - g)` for the
    logistic function), the per-node statistics are

    .. math::

        \bar z_v = \langle z_v \rangle , \qquad
        \sigma_v^2 = \mathrm{Var}[z_v] , \qquad
        \bar g_v = \langle g'(z_v) \rangle , \qquad
        s_v = \Big\langle \mathbb{1}\big[\, g'(z_v) < \theta\, g'(z^\star) \,\big] \Big\rangle ,

    with :math:`g'(z^\star)` the maximal slope (1 for tanh, 1/4 for the logistic function) and
    :math:`\theta` the ``saturation_threshold``. The per-edge statistics, for every existing edge
    :math:`u \to v`, are

    .. math::

        V_{uv} = \mathrm{Var}\big[ A_{uv}\, r_u(t) \big] = A_{uv}^2\, \mathrm{Var}[r_u] , \qquad
        W_{uv} = \bar g_v^{\,2}\, V_{uv} .

    **What they tell.**

    * ``mean_preactivation`` :math:`\bar z_v`: the average drive of the node. Far from 0 means that the
      bias (or a strong input) holds the node in the flat part of the activation, where it is hard to
      move; the sign shows on which side. With a zero-mean input, :math:`\bar z_v \approx \beta b_v`
      plus the mean of the recurrent input.
    * ``std_preactivation`` :math:`\sigma_v`: how far the drive swings around its mean. Large compared
      with the width of the activation curve means the node is driven into the nonlinear regime.
    * ``mean_gain`` :math:`\bar g_v`: the average slope, i.e. how strongly the node passes on a change
      of its inputs. Near 1 (tanh): linear regime, input differences are kept. Near 0: saturated, the
      node is insensitive, and so are the edges that lead into it.
    * ``saturation_fraction`` :math:`s_v`: the fraction of time in which the slope is below
      :math:`\theta` times its maximum.
    * ``edge_signal_variance`` :math:`V_{uv}`: how much the signal that edge :math:`u \to v` delivers
      fluctuates. An edge with a tiny variance carries almost no information, whatever its weight is.
    * ``edge_gain_weighted_variance`` :math:`W_{uv}`: the same, scaled with the squared mean slope of the
      receiving node. This approximates how much of the *fluctuation of the node output* the edge can
      be responsible for: an edge into a saturated node scores low even if its signal is strong.

    The edge variances are not additive: the sum over the incoming edges of a node is not
    :math:`\mathrm{Var}[z_v]`, because the signals are correlated and the input contributes as well. The
    values are an indicator of relative importance, not a decomposition of the variance. They
    are measured on the data ``x`` that is passed in, so use the training data for a statement about the
    fitted model, and compare across models only with the same data.

    Parameters
    ----------
    model : RC or ReservoirComputer
        A fitted model (the preprocessing and the bias settings frozen by ``fit`` are used).
    x : ndarray of shape (n_batch, T, n_in)
        Input sequences, for instance the training inputs.
    discard : int, optional
        Warm-up :math:`d`: the number of leading time steps that are excluded. By default the warm-up
        of the model (``discard_transients``).
    saturation_threshold : float, default=0.1
        Threshold :math:`\theta` in (0, 1) for the saturation fraction, relative to the maximal slope.

    Returns
    -------
    dict
        ``"node"``: dict of arrays of shape (N,) with the keys ``mean_preactivation``,
        ``std_preactivation``, ``mean_gain``, ``saturation_fraction`` and ``bias`` (:math:`\beta b_v`, the
        constant drive from the input bias, 0 if switched off).
        ``"edge"``: dict with ``edges`` of shape (E, 2), the (u, v) index pairs of the existing edges
        (the nonzero entries of :math:`A`), and the arrays ``edge_signal_variance`` and
        ``edge_gain_weighted_variance`` of shape (E,), in the same order.

    Raises
    ------
    RuntimeError
        If the model is not fitted.
    TypeError, ValueError
        For input that is not a 3D array with the right number of channels, or if ``discard`` leaves no
        time step, or if ``saturation_threshold`` is not in (0, 1).
    """
    # accept the sklearn-style wrapper as well as the layer-based model
    model = getattr(model, "model", model)

    if getattr(model, "_input_scaling_fit", None) is None:
        raise RuntimeError("Model is not fitted yet. Call fit() first.")
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array.")
    if x.ndim != 3:
        raise ValueError("x must have 3 dimensions (n_batch, n_timesteps, n_states).")
    if not 0 < saturation_threshold < 1:
        raise ValueError(f"saturation_threshold must be in (0, 1), got {saturation_threshold!r}.")
    n_in = model.input_layer.weights.shape[1]
    if x.shape[2] != n_in:
        raise ValueError(f"x has {x.shape[2]} channels, but the model has {n_in} input channels.")
    d = model.num_transients_to_remove if discard is None else int(discard)
    if d < 0 or d >= x.shape[1]:
        raise ValueError(f"discard must be in [0, {x.shape[1] - 1}], got {d}.")

    layer = model.reservoir_layer
    A = layer.weights
    x_tilde = model._transform_x(x)
    states = model.compute_reservoir_state(x_tilde)[:, :-1]  # r(0), ..., r(T-1), the states that drive step t

    # pre-activation z(t) = r(t) A + W_in x~(t) + beta b
    bias = model._bias_contribution()
    z = np.einsum("uv,btu->btv", A, states) + np.einsum("vk,btk->btv", model.input_layer.weights, x_tilde) + bias
    z, r = z[:, d:].reshape(-1, A.shape[0]), states[:, d:].reshape(-1, A.shape[0])

    # slope of the activation, from its output g(z)
    g = layer.activation_fun(z)
    slope = 1.0 - g**2 if layer.activation == "tanh" else g * (1.0 - g)
    max_slope = 1.0 if layer.activation == "tanh" else 0.25
    mean_gain = slope.mean(axis=0)

    node = {
        "mean_preactivation": z.mean(axis=0),
        "std_preactivation": z.std(axis=0),
        "mean_gain": mean_gain,
        "saturation_fraction": (slope < saturation_threshold * max_slope).mean(axis=0),
        "bias": np.broadcast_to(np.asarray(bias, dtype=float), (A.shape[0],)).copy(),
    }

    u, v = np.nonzero(A)
    var_r = r.var(axis=0)
    edge_var = A[u, v] ** 2 * var_r[u]
    edge = {
        "edges": np.column_stack([u, v]),
        "edge_signal_variance": edge_var,
        "edge_gain_weighted_variance": mean_gain[v] ** 2 * edge_var,
    }
    return {"node": node, "edge": edge}


if __name__ == "__main__":

    G = nx.erdos_renyi_graph(10, 0.5, directed=True)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.randn()

    edge = list(G.edges())[0]
    print(f"Properties for edge {edge}:")

    analyzer = EdgeAnalyzer()
    props = analyzer.extract_properties(G, edge)
    for k, v in props.items():
        print(f"  {k}: {v}")
