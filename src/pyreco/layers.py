r"""
Layers of a reservoir computer.

A reservoir computer maps an input sequence :math:`x(t) \in \mathbb{R}^{n_\mathrm{in}}` to an output
sequence :math:`\hat y(t) \in \mathbb{R}^{n_\mathrm{out}}` through a recurrent network of :math:`N` nodes
(the reservoir) with state :math:`r(t) \in \mathbb{R}^N`. The three kinds of layers hold the weights of
the three steps:

.. math::

    r(t+1) &= (1 - a)\, r(t) + a\, g\!\left( r(t)\, A + \tilde x(t)\, W_\mathrm{in}^\top + \beta\, b^\top \right) \\
    \hat y(t) &= r_\mathrm{out}(t+1)\, W_\mathrm{out} + c

* :class:`InputLayer` holds the read-in weights :math:`W_\mathrm{in} \in \mathbb{R}^{N \times n_\mathrm{in}}`
  and the signs :math:`b \in \{-1, +1\}^N` of the input bias, whose gain :math:`\beta \ge 0` is set with
  ``compile(input_bias=...)``. The preprocessed input is :math:`\tilde x = s \odot (x - \mu) / \sigma`
  (z-scoring with the training statistics, followed by the input scaling :math:`s`).
* :class:`ReservoirLayer` (:class:`RandomReservoirLayer`) holds the adjacency matrix
  :math:`A \in \mathbb{R}^{N \times N}`, where :math:`A_{uv}` is the weight of the edge :math:`u \to v`,
  the leakage rate :math:`a \in (0, 1]` and the activation :math:`g` (``tanh`` or the logistic function).
* :class:`ReadoutLayer` holds the readout weights :math:`W_\mathrm{out}` and the intercept :math:`c`
  that are fitted by ridge regression, acting on the states :math:`r_\mathrm{out}` of the read-out nodes.

All layers inherit from the abstract :class:`Layer`.
"""

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx

from pyreco.utils_networks import (
    gen_ER_graph,
    compute_density,
    get_num_nodes,
    compute_spec_rad,
    remove_nodes_from_graph,
    set_spec_rad
)


ACTIVATIONS = ("tanh", "sigmoid")
INPUT_WEIGHT_DISTRIBUTIONS = ("normal", "uniform", "sign")


def _check_number(name, value, low=None, high=None):
    """
    Return `value` as a float if it is a finite number in the interval (low, high]; a missing
    bound is not checked. Booleans are not numbers here.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {value!r}.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    too_low = low is not None and not value > low
    too_high = high is not None and value > high
    if too_low or too_high:
        interval = f"in ({low}, {high}]" if high is not None else f"larger than {low}"
        raise ValueError(f"{name} must be {interval}, got {value!r}.")
    return float(value)


def _check_nodes(nodes):
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)):
        raise TypeError(f"nodes must be an integer, got {nodes!r}.")
    if nodes < 2:
        raise ValueError(
            f"nodes must be at least 2, got {nodes!r}: a reservoir without self-loops needs two "
            "nodes to contain a directed cycle."
        )
    return int(nodes)


# implements the abstract base class
class Layer(ABC):
    """
    Abstract base class of all layers.

    Attributes
    ----------
    weights : ndarray or None
        The weight matrix of the layer (trainable or not).
    name : str
        Name of the layer.
    """

    @abstractmethod
    def __init__(self):
        self.weights = None  # every layer will have some weights (trainable or not)
        self.name: str = "layer"
        self._is_compiled: bool = False  # flag to check if the overall model is compiled
        pass

    @abstractmethod
    def update_layer_properties(self):
        pass


class InputLayer(Layer):
    r"""
    Read-in layer: maps the input channels to the reservoir nodes.

    The read-in matrix :math:`W_\mathrm{in} \in \mathbb{R}^{N \times n_\mathrm{in}}` (``N`` reservoir
    nodes, ``n_in`` input channels, irrespective of whether a time series or a vector is fed in) is
    drawn by ``model.compile()``, which knows the size of the reservoir. Its rows are set to 0 for the
    nodes that do not receive the input (``fraction_input`` of the reservoir layer).

    Parameters
    ----------
    input_shape : tuple of int
        ``(n_timesteps, n_states)`` of the input sequences.
    weight_distribution : {"normal", "uniform", "sign"}, default="normal"
        Distribution of the non-zero entries of :math:`W_\mathrm{in}`:

        * ``"normal"``: :math:`\mathcal{N}(0, 1)`, scaled to a maximum absolute value of 1,
        * ``"uniform"``: :math:`\mathcal{U}(-1, 1)`,
        * ``"sign"``: :math:`\pm 1` with equal probability (equal magnitudes, as in many echo state networks).

    Attributes
    ----------
    weights : ndarray of shape (N, n_states)
        The read-in matrix :math:`W_\mathrm{in}`, set by ``compile()``.
    bias : ndarray of shape (N,) or None
        Signs :math:`b_u \in \{-1, +1\}` of the constant input of every node, set by ``compile()``.
        The strength is the gain :math:`\beta` (``input_bias``) of the model.
    fraction_nonzero_entries : float
        Fraction of non-zero entries of :math:`W_\mathrm{in}`.

    Raises
    ------
    ValueError
        For an unknown ``weight_distribution``.
    """

    def __init__(self, input_shape, weight_distribution: str = "normal"):
        super().__init__()
        if weight_distribution not in INPUT_WEIGHT_DISTRIBUTIONS:
            raise ValueError(
                f"unknown weight_distribution {weight_distribution!r}, use one of "
                f"{INPUT_WEIGHT_DISTRIBUTIONS}."
            )
        self.weight_distribution = weight_distribution
        self.bias = None  # signs of the input bias, shape (n_reservoir_nodes,), set by compile
        self.shape = input_shape
        self.n_time = input_shape[0]
        self.n_states = input_shape[1]
        self.name = "input_layer"
        self.type = "input"

        # some properties of the readin layer
        self.fraction_nonzero_entries: (
            float  # fraction of nonzero entries in the input layer
        )

    def remove_nodes(self, nodes: list):
        r"""
        Delete the rows of :math:`W_\mathrm{in}` (and the bias signs) of reservoir nodes that are removed.

        Parameters
        ----------
        nodes : list of int
            Indices of the reservoir nodes.

        Raises
        ------
        TypeError
            If ``nodes`` is not a list.
        ValueError
            If an index is negative, too large or not an integer.
        """

        if not isinstance(nodes, list):
            raise TypeError("Nodes must be provided as a list of indices.")
        if np.max(nodes) > self.weights.shape[0]:
            raise ValueError(
                "Node index exceeds the number of nodes in the input layer."
            )
        if np.min(nodes) < 0:
            raise ValueError("Node index must be positive.")
        if not all(isinstance(x, int) for x in nodes):
            raise ValueError("All entries in the node list must be integers.")

        # remove nodes from [n_reservoir_nodes, n_states] matrix
        self.weights = np.delete(self.weights, nodes, axis=0)
        if self.bias is not None:
            self.bias = np.delete(self.bias, nodes)

        # update the properties of the input layer
        self.update_layer_properties()

    def update_layer_properties(self):
        """Recompute ``fraction_nonzero_entries`` = (non-zero entries of W_in) / (all entries)."""
        self.fraction_nonzero_entries = (
            np.count_nonzero(self.weights) / self.weights.size
        )


class ReadoutLayer(Layer):
    r"""
    Readout layer: a linear map from (a subset of) the reservoir states to the outputs.

    The readout is fitted by ridge regression on the reservoir states :math:`R` of the read-out nodes,

    .. math::

        \hat y(t) = r_\mathrm{out}(t+1)\, W_\mathrm{out} + c, \qquad
        \min_{W_\mathrm{out}, c} \; \sum_{t} \left\| y(t) - \hat y(t) \right\|^2 + \alpha \left\| W_\mathrm{out} \right\|_F^2 ,

    where the intercept :math:`c` is not penalized (states and targets are centered before the
    weights are fitted) and :math:`\alpha` is the ridge penalty of the optimizer.

    Parameters
    ----------
    output_shape : tuple of int
        ``(n_timesteps, n_states)`` of the target sequences.
    fraction_out : float, default=1.0
        Fraction of the reservoir nodes that are read out, in (0, 1].

    Attributes
    ----------
    readout_nodes : list of int
        Indices of the read-out nodes, chosen by ``compile()``.
    weights : ndarray of shape (n_readout_nodes, n_states)
        Readout weights :math:`W_\mathrm{out}`, set by ``fit()``.
    bias : ndarray of shape (n_states,) or None
        Readout intercept :math:`c`, set by ``fit()``.
    """

    def __init__(self, output_shape, fraction_out=1.0):
        super().__init__()
        self.output_shape: tuple = output_shape
        self.n_time = output_shape[0]
        self.n_states = output_shape[1]
        self.type = "output"
        # fraction of the reservoir nodes that are read out
        self.fraction_out: float = _check_number("fraction_out", fraction_out, low=0, high=1)
        self.name = "readout_layer"

        self.readout_nodes = []  # list of nodes that are linked to output
        self.bias = None  # readout intercept of shape (n_states,), set by fit

    def remove_nodes(self, nodes: list):
        """Not implemented: ``RC.remove_reservoir_nodes`` updates the read-out nodes itself."""

        # # delete entries that equal the nodes to be removed
        # idx_del = np.where(np.isin(self.readout_nodes, nodes))[0]
        # self.readout_nodes = np.delete(self.readout_nodes, idx_del)
        # self.update_layer_properties()
        raise NotImplementedError("This method is not yet implemented.")

    def update_layer_properties(self, n_reservoir_nodes: int = None):
        """
        Update ``fraction_out`` = (number of read-out nodes) / N.

        Parameters
        ----------
        n_reservoir_nodes : int, optional
            Current number of reservoir nodes :math:`N`. If None, ``fraction_out`` is unchanged.
        """
        if n_reservoir_nodes is not None:
            self.fraction_out = len(self.readout_nodes) / n_reservoir_nodes


class FeedbackLayer(Layer):
    r"""
    Feedback layer: like :class:`InputLayer`, but for the connections from the outputs back to the reservoir.

    The weights :math:`W_\mathrm{fb} \in \mathbb{R}^{N \times n_\mathrm{out}}` are created by ``compile()``.

    Parameters
    ----------
    feedback_shape : tuple of int
        ``(n_timesteps, n_states)`` of the fed-back sequences.
    """

    def __init__(self, feedback_shape):
        super().__init__()
        self.shape = feedback_shape
        self.n_time = feedback_shape[0]
        self.n_states = feedback_shape[1]
        self.name = "feedback_layer"

        # some properties of the feedback layer
        self.fraction_nonzero_entries: (
            float  # fraction of nonzero entries in the feedback layer
        )

    def remove_nodes(self, nodes: list):
        """
        Delete the rows of the feedback matrix of reservoir nodes that are removed.

        Parameters
        ----------
        nodes : list of int
            Indices of the reservoir nodes.

        Raises
        ------
        TypeError
            If ``nodes`` is not a list.
        ValueError
            If an index is negative, too large or not an integer.
        """

        if not isinstance(nodes, list):
            raise TypeError("Nodes must be provided as a list of indices.")
        if np.max(nodes) > self.weights.shape[0]:
            raise ValueError(
                "Node index exceeds the number of nodes in the input layer."
            )
        if np.min(nodes) < 0:
            raise ValueError("Node index must be positive.")
        if not all(isinstance(x, int) for x in nodes):
            raise ValueError("All entries in the node list must be integers.")

        # remove nodes from [n_reservoir_nodes, n_states] matrix
        self.weights = np.delete(self.weights, nodes, axis=0)

        # update the properties of the input layer
        self.update_layer_properties()

    def update_layer_properties(self):
        """Recompute ``fraction_nonzero_entries`` = (non-zero entries) / (all entries)."""
        self.fraction_nonzero_entries = (
            np.count_nonzero(self.weights) / self.weights.size
        )

class ReservoirLayer(Layer):  # subclass for the specific reservoir layers
    r"""
    Reservoir layer: the recurrent network of :math:`N` nodes.

    ``weights`` is the adjacency matrix :math:`A` of a directed graph. Edge convention:
    :math:`A_{uv}` is the edge :math:`u \to v` (:math:`u` sends, :math:`v` receives), identical to
    ``networkx.from_numpy_array(A, create_using=nx.DiGraph)``. The state update uses the row-vector form

    .. math::

        r(t+1) = (1 - a)\, r(t) + a\, g\!\left( r(t)\, A + \dots \right),

    so activity flows from the row index to the column index. The leakage rate :math:`a \in (0, 1]`
    sets the time scale of the nodes (:math:`a = 1`: no leak), and the activation :math:`g` is

    .. math::

        g(z) = \tanh(z) \quad \text{or} \quad g(z) = \frac{1}{1 + e^{-z}} .

    Parameters
    ----------
    nodes : int
        Number of nodes :math:`N \ge 2`.
    density : float
        Density of the network, in (0, 1].
    activation : {"tanh", "sigmoid"}
        The activation function :math:`g`.
    leakage_rate : float
        Leakage rate :math:`a`, in (0, 1].
    fraction_input : float
        Fraction of the nodes that receive the input, in (0, 1].
    init_res_sampling : {"zeros", "ones", "random", "random_normal"}
        Initial state :math:`r(0)`, see :class:`RandomReservoirLayer`.

    Attributes
    ----------
    weights : ndarray of shape (N, N)
        The adjacency matrix :math:`A`.
    spec_rad : float
        Spectral radius :math:`\rho(A) = \max_i |\lambda_i(A)|`.
    input_receiving_nodes : list of int
        Nodes that receive the input, chosen by ``compile()``.
    initial_res_states : ndarray of shape (N,)
        The initial state :math:`r(0)`.
    """

    def __init__(
        self,
        nodes,
        density,
        activation,
        leakage_rate,
        fraction_input,
        init_res_sampling,
    ):
        super().__init__()
        self.nodes: int = _check_nodes(nodes)  # number of reservoir nodes
        self.density: float = _check_number("density", density, low=0, high=1)
        self.spec_rad = None
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}.")
        self.activation = activation
        self.leakage_rate = _check_number("leakage_rate", leakage_rate, low=0, high=1)
        self.name = "reservoir_layer"
        self.type = "reservoir"
        # fraction of the nodes that receive the input
        self.fraction_input = _check_number("fraction_input", fraction_input, low=0, high=1)
        self.weights = None  # the adjacency matrix / reservoir network
        self.input_receiving_nodes = None

        # initial reservoir state (will be set later)
        self.initial_res_states = None
        if init_res_sampling not in ("zeros", "ones", "random", "random_normal"):
            raise ValueError(
                f"unknown init_res_sampling '{init_res_sampling}', use 'zeros', 'ones', "
                "'random' or 'random_normal'"
            )
        self.init_res_sampling = init_res_sampling

    def activation_fun(self, x: np.ndarray):
        r"""
        Apply the activation function :math:`g` element-wise.

        Parameters
        ----------
        x : ndarray
            Pre-activation :math:`z`.

        Returns
        -------
        ndarray
            :math:`\tanh(z)` or :math:`1 / (1 + e^{-z})`, depending on ``activation``.
        """
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        else:
            raise (ValueError(f"unknown activation function {self.activation}!"))

    def set_weights(self, network: np.ndarray):
        """
        Replace the adjacency matrix :math:`A` and update density and spectral radius.

        After ``compile()`` the new matrix must have the shape of the existing one.

        Parameters
        ----------
        network : ndarray of shape (N, N)
            The new adjacency matrix.

        Raises
        ------
        TypeError
            If ``network`` is not a numpy array.
        ValueError
            If it is not a square matrix of the current size.
        """
        # Updates all related parameters. 
        # MUST be having the same shape as the existing reservoir network.

        if not isinstance(network, np.ndarray):
            raise TypeError("Network must be a numpy array.")
        
        if network.ndim != 2:
            raise ValueError("Network must be a 2D numpy array.")
        
        if network.shape[0] != network.shape[1]:
            raise ValueError("Network must be square (same number of rows and columns).")
        
        # for now, we only accept matrices that have the same shape as the existing reservoir
        if self._is_compiled and (network.shape != self.weights.shape):
            raise ValueError(f"Network must have the same shape as the existing reservoir ({self.weights.shape}).")

        # overwrite the existing adjacency matrix
        self.weights = network

        # update related reservoir properties (density, spectral radius, etc.)
        self.update_layer_properties()

    def set_initial_state(self, r_init: np.ndarray):
        r"""
        Set the initial state :math:`r(0)` that every sequence starts from.

        Parameters
        ----------
        r_init : ndarray of shape (N,)
            Initial state of the nodes.

        Raises
        ------
        ValueError
            If the number of entries is not :math:`N`.
        """

        if r_init.shape[0] != self.nodes:
            raise (
                ValueError(
                    "initial reservoir state does not match the number of nodes in the reservoir!"
                )
            )
        self.initial_res_states = r_init

    def remove_nodes(self, nodes: list):
        """
        Remove nodes from the reservoir network.

        The rows and columns of the nodes are deleted from the adjacency matrix, the initial state
        shrinks accordingly, and number of nodes, density and spectral radius are updated.

        Parameters
        ----------
        nodes : list of int
            Indices of the nodes to remove.

        Raises
        ------
        TypeError
            If ``nodes`` is not a list.
        ValueError
            If an index is negative or exceeds the number of nodes.
        """
        # remove nodes from the reservoir network

        if not isinstance(nodes, list):
            raise TypeError("Nodes must be provided as a list of indices.")

        if np.max(nodes) > self.nodes:
            raise ValueError("Node index exceeds the number of nodes in the reservoir.")

        if np.min(nodes) < 0:
            raise ValueError("Node index must be positive.")

        # 1. remove nodes from the adjacency matrix
        self.weights = remove_nodes_from_graph(graph=self.weights, nodes=nodes)

        # 2. update reservoir properties
        self.update_layer_properties()

        # 3. update the initial reservoir state
        self.initial_res_states = np.delete(self.initial_res_states, nodes, axis=0)

        # # 4. update the info about input-receiving nodes
        # self.input_receiving_nodes = np.delete(self.input_receiving_nodes, nodes)

    def update_layer_properties(self):
        r"""
        Recompute the properties of the network from ``weights``; call it after any change of the network.

        .. math::

            N = \dim A, \qquad
            d = \frac{\left| \{ (u, v) : A_{uv} \ne 0 \} \right|}{N^2}, \qquad
            \rho(A) = \max_i \left| \lambda_i(A) \right| .
        """
        self.nodes = get_num_nodes(self.weights)
        self.density = compute_density(self.weights)
        self.spec_rad = compute_spec_rad(self.weights)

    def get_spec_rad(self):
        r"""Spectral radius :math:`\rho(A)` of the current network."""
        return self.spec_rad

    def set_spec_rad(self, value):
        r"""
        Rescale the network to a spectral radius, :math:`A \leftarrow A \, \rho_\mathrm{new} / \rho(A)`.

        A failure leaves the layer unchanged.

        Parameters
        ----------
        value : float
            The desired spectral radius :math:`\rho_\mathrm{new} > 0`.

        Raises
        ------
        ValueError
            If the value is not positive, the weights are not set, or the network has spectral
            radius 0 (no directed cycle), which cannot be rescaled.
        """

        # check that the value is positive and that the weights are set
        if value <= 0:
            raise ValueError("Spectral radius must be positive")
        if self.weights is None:
            raise ValueError("Weights must be set before changing spectral radius.")

        # rescale and verify first, so that a failure leaves the layer untouched.
        # Raises a ValueError if the network has spectral radius 0 (no directed cycle).
        _new_weights = set_spec_rad(self.weights, value)
        _new_spec_rad = compute_spec_rad(_new_weights)
        if not np.isclose(value, _new_spec_rad, atol=1e-5):
            raise ValueError(
                f"Failed to set spectral radius within tolerance 1e-5. Expected {value}, got {_new_spec_rad}."
            )
        self.weights = _new_weights
        self.spec_rad = _new_spec_rad

    def get_leakage_rate(self):
        """Leakage rate :math:`a`."""
        return self.leakage_rate

    def set_leakage_rate(self, value):
        """Set the leakage rate :math:`a \\in (0, 1]`; raises a ValueError outside this interval."""
        if not (0 < value <= 1):
            raise ValueError("Leakage rate must be in (0, 1]")
        self.leakage_rate = value

    def get_activation(self):
        """Name of the activation function (``"tanh"`` or ``"sigmoid"``)."""
        return self.activation

    def set_activation(self, value):
        """Set the activation function by name; raises a ValueError for an unknown name."""
        if value not in ACTIVATIONS:
            raise ValueError("Activation must be 'tanh' or 'sigmoid'")
        self.activation = value


class RandomReservoirLayer(ReservoirLayer):
    r"""
    Random (Erdős-Rényi) reservoir network.

    Every ordered pair of different nodes :math:`(u, v)` is connected with probability :math:`p`
    (``density``); the weights of the edges are drawn from ``weight_distribution`` and the matrix is
    rescaled to the requested spectral radius,

    .. math::

        A_{uv} = \xi_{uv}\, w_{uv}, \quad \xi_{uv} \sim \mathrm{Bernoulli}(p), \qquad
        A \leftarrow A \, \frac{\rho_\mathrm{target}}{\rho(A)} ,

    so the realized density differs a little from :math:`p` (it is available as ``density``).
    Isolated nodes are connected again, and graphs without a directed cycle (:math:`\rho(A) = 0`)
    are redrawn, since they cannot be rescaled.

    Parameters
    ----------
    nodes : int
        Number of nodes, at least 2.
    density : float, default=0.1
        Edge probability :math:`p`, in (0, 1].
    activation : {"tanh", "sigmoid"}, default="tanh"
        Activation function :math:`g`.
    leakage_rate : float, default=0.5
        Leakage rate :math:`a`, in (0, 1].
    fraction_input : float, default=0.8
        Fraction of the nodes that receive the input, in (0, 1].
    spec_rad : float, default=0.9
        Spectral radius :math:`\rho_\mathrm{target} > 0` that the network is scaled to.
    init_res_sampling : {"zeros", "ones", "random", "random_normal"}, default="zeros"
        Initial state :math:`r(0)`, the same for every sample: all zeros, all ones, uniform on
        :math:`[-1, 1]` or standard normal (the last two scaled to a maximum absolute value of 1).
        Explicit values can be set after ``compile()`` with :meth:`ReservoirLayer.set_initial_state`.
    seed : int, optional
        Seed of the reservoir graph. It fixes the adjacency matrix completely, including the repair of
        isolated nodes; the global random state is never used. The random choices made when a model
        is compiled (read-in weights, initial state, input and readout nodes) are seeded by the model,
        see ``RC(seed=...)``.
    weight_distribution : {"positive", "sign", "uniform", "normal"}, default="positive"
        Distribution of the weights :math:`w_{uv}` before the rescaling:

        * ``"positive"``: all weights equal 1 (a binary graph),
        * ``"sign"``: :math:`\pm 1` with equal probability,
        * ``"uniform"``: :math:`\mathcal{U}(-1, 1)`,
        * ``"normal"``: :math:`\mathcal{N}(0, 1)`.

        The graph (which edges exist) is the same for all of them. Signed weights are common for
        echo state networks and matter for tasks with nonlinear memory such as NARMA10.

    Raises
    ------
    TypeError, ValueError
        For arguments of the wrong type or outside their range.
    """

    def __init__(
        self,
        nodes,
        density: float = 0.1,
        activation: str = "tanh",
        leakage_rate: float = 0.5,
        fraction_input: float = 0.8,
        spec_rad: float = 0.9,
        init_res_sampling="zeros",
        seed=None,
        weight_distribution: str = "positive",
    ):
        # Call the parent class's __init__ method
        super().__init__(
            nodes=nodes,
            density=density,
            activation=activation,
            leakage_rate=leakage_rate,
            fraction_input=fraction_input,
            init_res_sampling=init_res_sampling,
        )

        # initialize subclass-specific attributes
        self.seed = seed
        self.weight_distribution = weight_distribution
        self.spec_rad = _check_number("spec_rad", spec_rad, low=0)

        # generate a random ER graph using networkx
        self.weights = gen_ER_graph(
            nodes=nodes,
            density=density,
            spec_rad=self.spec_rad,
            directed=True,
            seed=seed,
            weight_distribution=weight_distribution,
        )

        # update the spectral radius and the density to the actual values. The density is
        # the edge probability of the ER graph, so the realized density differs from the
        # requested one (it is available as self.density).
        self.update_layer_properties()

        # check that the graph actually satisfies the spectral radius requested by the user
        if not np.isclose(spec_rad, self.spec_rad, atol=1e-5):
            raise ValueError(
                f"Failed to set spectral radius within tolerance 1e-5. Expected {spec_rad}, got {self.spec_rad}."
            )

    #def update_layer_properties(self):
        #Updates the reservoir properties including the number of nodes, density, and spectral radius.
        #self.nodes = get_num_nodes(self.weights)
        #self.density = compute_density(self.weights)
        # self.spec_rad = compute_spec_rad(self.weights)


# class ReccurrenceLayer(ReservoirLayer):
#     # To Do: accept a random seed
#     def __init__(self, nodes, density, activation: str = 'tanh', leakage_rate: float = 0.2):
#         # Call the parent class's __init__ method
#         super().__init__(nodes, density, activation, leakage_rate)
#
#         # Initialize subclass-specific attributes
#         # https://pyts.readthedocs.io/en/stable/generated/pyts.image.RecurrencePlot.html#pyts.image.RecurrencePlot
#         # https://tocsy.pik-potsdam.de/pyunicorn.php
#
