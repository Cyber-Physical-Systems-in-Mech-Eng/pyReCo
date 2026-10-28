import numpy as np
from abc import ABC
from typing import Union
import networkx as nx
import copy
import multiprocessing
from functools import partial
from pyreco.plotting import visualize_reservoir_network_circle

from pyreco.layers import (
    Layer,
    InputLayer,
    ReservoirLayer,
    ReadoutLayer,
    FeedbackLayer,
)
from pyreco.optimizers import Optimizer, assign_optimizer
from pyreco.metrics import assign_metric
from pyreco.node_selector import NodeSelector
from pyreco.initializer import NetworkInitializer
from pyreco.utils_networks import rename_nodes_after_removal
from pyreco.utils_random import check_seed


# def sample_random_nodes(total_nodes: int, fraction: float):
#     """
#     Select a subset of randomly chosen nodes.

#     Args:
#     total_nodes (int): Total number of available nodes.
#     fraction (float): Fraction of nodes to select.

#     Returns:
#     np.ndarray: Array of randomly selected node indices.
#     """
#     return np.random.choice(
#         total_nodes, size=int(total_nodes * fraction), replace=False
#     )


# def discard_transients_indices(n_batches, n_timesteps, transients):
#     indices_to_remove = []
#     for i in range(n_batches * n_timesteps):
#         t = i % n_timesteps  # Current timestep within the batch
#         if t < transients:
#             indices_to_remove.append(i)
#     return indices_to_remove


class CustomModel(ABC):
    r"""
    Abstract base class of the reservoir computers.

    The syntax is similar to the TensorFlow model API: layers are added with :meth:`add`, the model is
    configured with :meth:`compile`, trained with :meth:`fit` and applied with :meth:`predict`. A model
    has an input layer, a reservoir layer and a readout layer.

    For an input sequence :math:`x(0), \dots, x(T-1)` the reservoir state :math:`r(t) \in \mathbb{R}^N`
    starts at :math:`r(0)` (zero by default) and evolves as

    .. math::

        r(t+1) = (1 - a)\, r(t) + a\, g\!\left( r(t)\, A + \tilde x(t)\, W_\mathrm{in}^\top + \beta\, b^\top \right),
        \qquad
        \tilde x(t) = s \odot \frac{x(t) - \mu}{\sigma} ,

    with the adjacency matrix :math:`A`, leakage rate :math:`a`, activation :math:`g`, read-in matrix
    :math:`W_\mathrm{in}`, input bias :math:`\beta b` (:math:`b \in \{\pm 1\}^N`) and the input
    preprocessing (z-scoring with the training mean :math:`\mu` and standard deviation :math:`\sigma` per
    channel, input scaling :math:`s`). The output is a linear readout of the states :math:`r_\mathrm{out}`
    of the read-out nodes,

    .. math::

        \hat y(j) = r_\mathrm{out}(d + j + 1)\, W_\mathrm{out} + c ,

    where :math:`d` is the warm-up (``discard_transients``): target :math:`j` is paired with the state
    after the inputs :math:`x(0), \dots, x(d + j)`. :math:`W_\mathrm{out}` and :math:`c` minimize the
    ridge-regularized squared error of :class:`pyreco.layers.ReadoutLayer`.

    Parameters
    ----------
    seed : int, optional
        Seed for the random choices made when the model is compiled: the read-in weights, the initial
        reservoir state, the nodes that receive the input, the nodes that are read out and the signs of
        the input bias. The global random state (``np.random``, ``random``) is never used. None draws
        fresh entropy at every ``compile()``. The reservoir graph has its own seed, see
        :class:`pyreco.layers.RandomReservoirLayer`.
    """

    def __init__(self, seed: int = None):
        """Create an empty model with default hyperparameters (see the class documentation)."""
        check_seed(seed)
        self.seed = seed
        self._init_rngs()

        # Initialize layers
        self.input_layer: InputLayer
        self.reservoir_layer: ReservoirLayer
        self.readout_layer: ReadoutLayer
        self.feedback_layer: FeedbackLayer  # optional feedback layer

        # Initialize hyperparameters
        self.metrics = []
        self.metrics_fun = []
        self.optimizer: Optimizer
        self.discard_transients = 0

        # Initialize other attributes
        self.num_trainable_weights: int
        self.num_nodes: int
        self.num_time_in: int
        self.num_time_out: int
        self.num_states_in: int
        self.num_states_out: int

        # Input preprocessing: per-channel z-scoring followed by a gain (input scaling).
        #   x_tilde = input_scaling * (x - input_mean) / input_std
        # The gain is equivalent to scaling the columns of the read-in matrix W_in.
        # The flags below are the user-facing settings; the values actually used for
        # predictions are frozen at fit time (see the *_fit attributes), so that
        # training and inference always apply the same transform.
        self.normalize_inputs: bool = True  # z-score the inputs
        self.input_scaling: Union[float, np.ndarray] = 1.0  # scalar or (d,) vector

        # Input bias: a constant input to every reservoir node, input_bias * b with b a
        # fixed vector of random signs (input_layer.bias, drawn at compile). 0 = none.
        self.input_bias: float = 0.0

        # Statistics learned during fit (z-scoring), shape (d,)
        self.input_mean: np.ndarray = None
        self.input_std: np.ndarray = None

        # Preprocessing settings frozen at fit time
        self._normalize_inputs_fit: bool = None
        self._input_scaling_fit: Union[float, np.ndarray] = None
        self._input_bias_fit: float = None

    def _init_rngs(self):
        """
        (Re)create the random streams of the model from its seed. One independent stream
        per random choice, so that e.g. changing the fraction of readout nodes does not
        change the read-in weights. Called at construction and at every compile(), so
        compiling twice with a fixed seed gives the same model.
        """
        children = np.random.SeedSequence(self.seed).spawn(5)
        (
            self._rng_readin_weights,
            self._rng_init_state,
            self._rng_input_nodes,
            self._rng_readout_nodes,
            self._rng_bias,
        ) = [np.random.default_rng(child) for child in children]

    def add(self, layer: Layer):
        """
        Add a layer to the model; it is assigned to the attribute that belongs to its type.

        Parameters
        ----------
        layer : Layer
            An input, reservoir, readout or feedback layer. A second layer of the same kind replaces
            the first.

        Raises
        ------
        TypeError
            If ``layer`` is not a :class:`pyreco.layers.Layer`.
        """
        # Sanity check for the correct shape of the input argument layer
        if not isinstance(layer, Layer):
            raise TypeError(
                "The layer must be an instance of the Layer class or its subclasses."
            )

        # assign the layer to the correct attribute
        if isinstance(layer, InputLayer):
            self.input_layer = layer
        elif issubclass(type(layer), ReservoirLayer):
            self.reservoir_layer = layer
        elif isinstance(layer, ReadoutLayer):
            self.readout_layer = layer
        elif isinstance(layer, FeedbackLayer):
            self.feedback_layer = layer
        else:
            raise ValueError("Unsupported layer type.")

    # TODO: the following method should be implemented in the CustomModel class
    #   def _set_readin_nodes(self, nodes: Union[list, np.ndarray] = None):

    def compile(
        self,
        optimizer: str = "ridge",
        metrics: Union[list, str] = None,
        discard_transients: int = 0,
        normalize_inputs: bool = None,
        input_scaling: Union[float, np.ndarray] = None,
        input_bias: float = None,
    ):
        r"""
        Configure the model for training and draw its random components.

        Creates the read-in matrix :math:`W_\mathrm{in}` and the bias signs :math:`b`, the initial state
        :math:`r(0)` and chooses the input-receiving and read-out nodes. Compiling twice with a fixed
        seed gives the same model.

        Parameters
        ----------
        optimizer : str or Optimizer, default="ridge"
            Name of the optimizer that finds the readout weights, or an optimizer instance.
        metrics : list of str or str, optional
            Metrics that are computed during :meth:`fit` (default ``["mse"]``).
        discard_transients : int, default=0
            Warm-up :math:`d` of the reservoir. The target at time step :math:`j` is paired with the state
            :math:`r(d + j + 1)`, i.e. the state after the inputs :math:`x(0), \dots, x(d + j)` have been
            applied; the first states are not used because they still depend on the initial state. It must
            not exceed :math:`T - K` for :math:`T` input and :math:`K` target time steps (checked in
            :meth:`fit`): to get a warm-up, provide inputs that are longer than the targets. For a
            sequence-to-scalar task (:math:`K = 1`) use :math:`d = T - 1` to pair the target with the last
            state.
        normalize_inputs : bool, optional
            z-score each input channel to mean 0 and standard deviation 1 (statistics over batch and time
            of the training data). Default: on. None keeps the current setting.
        input_scaling : float or ndarray, optional
            Positive gain :math:`s` applied to the z-scored inputs, a scalar or one value per input channel.
            Default 1 (no additional scaling). None keeps the current setting.
        input_bias : float, optional
            Gain :math:`\beta \ge 0` of a constant input to every reservoir node. Node :math:`u` receives
            :math:`\beta b_u` with :math:`b_u = \pm 1` (drawn once at compile with equal probability),

            .. math::

                r(t+1) = (1 - a)\, r(t) + a\, g\!\left( r(t)\, A + \tilde x(t)\, W_\mathrm{in}^\top + \beta\, b^\top \right).

            Default 0 (no bias). A bias lets the units work away from the symmetric point of ``tanh`` (without
            it, the states are odd functions of the input and cannot represent even functions such as
            :math:`x^2`) and is part of most echo state networks. None keeps the current setting.
            Not used by :class:`AutoRC`.

        Raises
        ------
        TypeError
            If ``discard_transients`` is not an integer.
        ValueError
            If ``discard_transients`` is negative, or a metric or optimizer is unknown.

        Notes
        -----
        The reservoir starts every sample from the same initial state :math:`r(0)`, which is zero by default.
        Choose another sampling method with ``RandomReservoirLayer(init_res_sampling=...)``, or set explicit
        values after ``compile()`` with ``model.reservoir_layer.set_initial_state(r0)``.
        """

        # sanity checks for inputs
        if not isinstance(discard_transients, int):
            raise TypeError("discard_transients must be a positive integer!")
        elif discard_transients < 0:
            raise ValueError("discard_transients must be >=0")

        if metrics is None:
            metrics = ["mse"]

        # input preprocessing settings
        self.set_normalization(normalize_inputs=normalize_inputs)
        if input_scaling is not None:
            self.set_input_scaling(input_scaling)
        if input_bias is not None:
            self.set_input_bias(input_bias)

        # fresh random streams: compiling twice with the same seed gives the same model
        self._init_rngs()

        # check consistency of layers, data shapes etc.
        # TODO: do we have input, reservoir and readout layer?
        # TODO: are all shapes correct on input and output side?
        # TODO: let the user specify the reservoir initialization method

        # set the metrics (like in TensorFlow)
        self._set_metrics(metrics)

        # set the optimizer that will find the readout weights
        self._set_optimizer(optimizer)

        # set number of transients to discard (warmup phase)
        self.discard_transients = int(discard_transients)
        # IDEA (not implemented): discard_transients="auto". The influence of the initial state
        # decays (echo state property), per step by about lambda = 1 - leakage + leakage * spec_rad
        # in the linear regime and faster when the input saturates the units. A warm-up could be
        # chosen from the data: run two different initial states through the training inputs and
        # take the first number of steps after which the relative difference of the two state
        # trajectories stays below a threshold (e.g. 1e-2). Measured for 100 nodes with a unit
        # variance input: 26 steps for leakage 0.5 / spec_rad 0.9, 66 for 0.2 / 0.9, 129 for
        # 0.1 / 0.9.

        # copy some layer properties to the model level for easier access
        self.num_states_in = self.input_layer.n_states
        self.num_states_out = self.readout_layer.n_states
        self.num_nodes = self.reservoir_layer.nodes

        # Sample the input connections: create W_in read-in weight matrix
        self._connect_input_to_reservoir()  # check for dependency injection here!

        # Set initial states of the reservoir (zeros unless the layer asks for another method)
        self._initialize_network()

        # Select readout nodes according to the fraction specified by the user in the readout layer. By default, randomly sample nodes. User can also provide a list of nodes to use for readout.
        self._set_readout_nodes()

        # flag all layers as compiled (required for later manipulation from outside)
        self.input_layer._is_compiled = True
        self.reservoir_layer._is_compiled = True
        self.readout_layer._is_compiled = True

    def AutoRC_compile(
        self,
        optimizer: str = "ridge",
        metrics: Union[list, str] = None,
        discard_transients: int = 0,
        normalize_inputs: bool = None,
        input_scaling: Union[float, np.ndarray] = None,
    ):
        """
        Configure the model for training. Same arguments as `compile`.
        """

        # sanity checks for inputs
        if not isinstance(discard_transients, int):
            raise TypeError("discard_transients must be a positive integer!")
        elif discard_transients < 0:
            raise ValueError("discard_transients must be >=0")

        if metrics is None:
            metrics = ["mse"]

        # input preprocessing settings
        self.set_normalization(normalize_inputs=normalize_inputs)
        if input_scaling is not None:
            self.set_input_scaling(input_scaling)

        # fresh random streams: compiling twice with the same seed gives the same model
        self._init_rngs()

        # check consistency of layers, data shapes etc.
        # TODO: do we have input, reservoir and readout layer?
        # TODO: are all shapes correct on input and output side?
        # TODO: let the user specify the reservoir initialization method

        # set the metrics (like in TensorFlow)
        self._set_metrics(metrics)

        # set the optimizer that will find the readout weights
        self._set_optimizer(optimizer)

        # set number of transients to discard (warmup phase)
        self.discard_transients = int(discard_transients)
        # IDEA (not implemented): discard_transients="auto". The influence of the initial state
        # decays (echo state property), per step by about lambda = 1 - leakage + leakage * spec_rad
        # in the linear regime and faster when the input saturates the units. A warm-up could be
        # chosen from the data: run two different initial states through the training inputs and
        # take the first number of steps after which the relative difference of the two state
        # trajectories stays below a threshold (e.g. 1e-2). Measured for 100 nodes with a unit
        # variance input: 26 steps for leakage 0.5 / spec_rad 0.9, 66 for 0.2 / 0.9, 129 for
        # 0.1 / 0.9.

        # copy some layer properties to the model level for easier access
        self.num_states_in = self.input_layer.n_states
        self.num_states_out = self.readout_layer.n_states
        self.num_nodes = self.reservoir_layer.nodes
        print(f"feedback layer: {self.feedback_layer}")
        if self.feedback_layer is not None:
            self.num_states_fb = self.feedback_layer.n_states   
        # self.num_states_fb = self.feedback_layer.n_states # else 0

        # Sample the input connections: create W_in read-in weight matrix
        self._connect_input_to_reservoir()  # check for dependency injection here!

        ###connect feedback layer to reservoir
        self._connect_feedback_to_reservoir()

        # Set initial states of the reservoir (zeros unless the layer asks for another method)
        self._initialize_network()

        # Select readout nodes according to the fraction specified by the user in the readout layer. By default, randomly sample nodes. User can also provide a list of nodes to use for readout.
        self._set_readout_nodes()

        # flag all layers as compiled (required for later manipulation from outside)
        self.input_layer._is_compiled = True
        self.reservoir_layer._is_compiled = True
        self.readout_layer._is_compiled = True
        self.feedback_layer._is_compiled = True if self.feedback_layer is not None else False

    def fit(self, x: np.ndarray,
            y: np.ndarray,
            visualize=False,
            store_states: bool = False
            ) -> dict:
        r"""
        Train the readout on the reservoir states of the given sequences.

        The inputs are preprocessed as :math:`\tilde x = s \odot (x - \mu) / \sigma` (per-channel z-scoring
        with statistics over batches and time, if enabled; see :meth:`compile` and :meth:`set_normalization`).
        The settings and statistics are frozen here and reused by :meth:`predict`. Targets are used as given,
        so predictions are in the physical units of :math:`y`.

        With the matrix :math:`R` of the read-out states (one row per target time step and sample, paired
        as described in :meth:`compile`) and the targets :math:`Y`, the intercept is the mean difference and
        the weights solve the ridge problem on the centered data,

        .. math::

            W_\mathrm{out} = \arg\min_W \left\| (R - \bar R) W - (Y - \bar Y) \right\|_F^2 + \alpha \left\| W \right\|_F^2 ,
            \qquad
            c = \bar Y - \bar R\, W_\mathrm{out} .

        Parameters
        ----------
        x : ndarray of shape (n_batch, T, n_in)
            Input sequences.
        y : ndarray of shape (n_batch, K, n_out)
            Targets, :math:`K \le T`; the :math:`K` targets belong to the time steps after the warm-up.
        visualize : bool, default=False
            Has no effect (kept for compatibility); plot the network with :meth:`model_visualize`.
        store_states : bool, default=False
            Also return the reservoir states.

        Returns
        -------
        dict
            ``"train_score"`` (the first metric on the training data), ``"train_metrics"`` (all compiled
            metrics) and, with ``store_states=True``, ``"res_states"`` of shape ``(n_batch, T, N)``.

        Raises
        ------
        TypeError, ValueError
            For data that is not a 3D float array, has other sample or channel counts than the model,
            contains NaN or infinite values, is too short for the warm-up, or if no node receives the input
            or is read out.
        RuntimeError
            If the model is not compiled.
        """
        # sanity checks for inputs
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input and target data must be numpy arrays.")
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(
                "Input and target data must have 3 dimensions (n_batch, n_timesteps, n_features)."
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "Input and target data must have the same number of samples (first dimension)."
            )
        if x.shape[1] < y.shape[1]:
            raise ValueError(
                "Input data must have at least as many timesteps as the target data."
            )
        if not isinstance(store_states, bool):
            raise TypeError("store_states must be a boolean.")

        # the model has to be compiled, and the input has to reach the reservoir and the
        # readout has to see it (nodes can be removed after compiling)
        if getattr(self, "input_layer", None) is None or self.input_layer.weights is None:
            raise RuntimeError("The model is not compiled. Call compile() before fit().")
        if not np.any(self.input_layer.weights):
            raise ValueError(
                "No reservoir node receives the input: all read-in weights are zero (were all "
                "nodes that receive the input removed?)."
            )
        if len(self.readout_layer.readout_nodes) == 0:
            raise ValueError(
                "No reservoir node is read out (were all readout nodes removed?)."
            )

        # the data has to match the model and must not contain NaN or infinite values
        # (NaN is easy to produce when targets are shifted by hand)
        self._check_channels(x, "x", self.input_layer.weights.shape[1], "input")
        self._check_channels(y, "y", self.readout_layer.n_states, "target")
        self._check_finite(x, "x")
        self._check_finite(y, "y")

        # Extract shapes of inputs and outputs, expected to be 3D arrays
        # of shape [n_batch, n_timesteps, n_features]
        n_batch, n_time_in = x.shape[0], x.shape[1]
        n_time_out, n_states_out = y.shape[-2], y.shape[-1]

        # Input preprocessing: z-score per channel (statistics over batches and time of
        # the training data), then apply the input scaling. The settings and the
        # statistics are frozen here and reused by predict().
        x_raw = x
        self.input_scaling = self._validate_input_scaling(
            self.input_scaling, n_channels=x.shape[-1]
        )
        self._normalize_inputs_fit = bool(self.normalize_inputs)
        self._input_scaling_fit = self.input_scaling
        self._input_bias_fit = self.input_bias
        if self._normalize_inputs_fit:
            self.input_mean = x.mean(axis=(0, 1))
            self.input_std = x.std(axis=(0, 1))
            self.input_std[self.input_std == 0] = 1.0  # avoid divide-by-zero
        else:
            self.input_mean, self.input_std = None, None
        x = self._transform_x(x)

        # warm-up: the target at time step j is paired with the state r(d + j + 1), d being
        # the number of transients to discard. The states r(0), ..., r(d) are not used, the
        # inputs after the state of the last target are not used either.
        max_warmup = n_time_in - n_time_out
        if self.discard_transients > max_warmup:
            raise ValueError(
                f"discard_transients ({self.discard_transients}) must not exceed the number of "
                f"input time steps minus the number of target time steps "
                f"({n_time_in} - {n_time_out} = {max_warmup}). Provide longer inputs or "
                f"reduce the warm-up."
            )
        self.num_transients_to_remove = self.discard_transients

        # update some class attributes that depend on the training data
        self.num_time_in = n_time_in
        self.num_states_in = x.shape[-1]
        self.num_time_out = n_time_out
        self.num_states_out = n_states_out

        # metrics that score the model performance (the ones given to compile())
        if self.metrics_fun:
            metric_names, metric_funs = self.metrics, self.metrics_fun
        else:
            metric_names, metric_funs = ["mean_squared_error"], [assign_metric("mean_squared_error")]

        # train the model on the given data
        reservoir_states, _ = self._train_model(x=x, y=y)

        # score on the training data, in the original units of x and y
        y_pred = self.predict(x=x_raw)[:, :n_time_out]
        train_metrics = {
            name: float(fun(y, y_pred)) for name, fun in zip(metric_names, metric_funs)
        }
        train_score = train_metrics[metric_names[0]]

        # Update trainable weights count
        self.num_trainable_weights = self.reservoir_layer.weights.size

        # Build history dictionary: train_score is the first metric given to compile(),
        # train_metrics holds all of them
        history = {"train_score": train_score, "train_metrics": train_metrics}

        if store_states:
            history["res_states"] = reservoir_states

        return history

    def model_visualize(self, save=False, file_name=None, file_type=None, Node_colors=None,
                        Edge_Weights=None):
        """
        Visualizes the reservoir network.
        Allows saving in PNG/PDF/JPG?SVG/JPEG formats with a custom filename.
        Allows user-defined node/edge colors.
        """

        # --- DEFULT EDIGE SIZE ---
        if Edge_Weights is None:
            Edge_Weights = 0.7

        # --- DEFAULT COLORS ---
        default_colors = {
            'CWinp': 'black',               # input → reservoir edges
            'CWres_inp': 'lightcoral',      # reservoir: input only
            'CWres_out': 'lightgreen',      # reservoir: output only
            'CWres_both': 'orange',         # reservoir: both in & out
            'CWres_internal': 'lightblue',  # reservoir internal edges
            'CWout': 'black',               # reservoir → output edges
            'Winp': 'blue',                 # input node
            'Wout': 'red',                  # output node
            'CWres': 'grey'                 # reservoir internal connection
        }

        # Merge user colors
        if Node_colors is None:
            Node_colors = default_colors
        else:
            for key in default_colors:
                if key not in Node_colors or Node_colors[key] is None:
                    Node_colors[key] = default_colors[key]

        # Normalize file type
        if file_type is not None:
            file_type = file_type.lower().strip()

        save_path = None
        if save:
            # If no filename is provided → default
            if not file_name and file_name != "":
                file_name = "reservoir_network"
            
            # If no file type → default JPEG
            if not file_type and file_type != "":
                file_type = "jpeg"

            # Validate file type
            if file_type not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
                raise RuntimeError("Error: Unsupported file type. Use png, jpg, jpeg, svg, or pdf.")

            save_path = f"{file_name}.{file_type}"

        # Call visualizer
        try:
            visualize_reservoir_network_circle(
                    G_Net=self.reservoir_layer.weights,
                    W_inp=self.input_layer.weights,
                    W_out=self.readout_layer.weights,
                    n_inputs=self.input_layer.n_states,
                    n_outputs=self.readout_layer.n_states,
                    save_path=save_path,
                    Node_colors=Node_colors,
                    Edge_Weights=Edge_Weights
                )
        except Exception as e:
            print("Visualization failed:", e)

    def predict(self, x: np.ndarray) -> np.ndarray:
        r"""
        Predict the output for each time step after the warm-up (single-step prediction).

        Every sample starts from the initial state :math:`r(0)`; the prediction for step :math:`j` is
        :math:`\hat y(j) = r_\mathrm{out}(d + j + 1)\, W_\mathrm{out} + c` with the warm-up :math:`d`.

        Parameters
        ----------
        x : ndarray of shape (n_batch, T, n_in)
            Input sequences, :math:`T > d`.

        Returns
        -------
        ndarray of shape (n_batch, T - d, n_out)
            The predictions in the physical units of the training targets.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        TypeError, ValueError
            For input that is not a 3D array, has the wrong number of channels, contains NaN or infinite
            values, or is not longer than the warm-up.
        """
        # makes prediction for given input (single-step prediction)
        # expects inputs of shape [n_batch, n_timestep, n_states]
        # returns predictions in shape of [n_batch, n_timestep, n_states]

        # one_shot = True will *not* re-initialize the reservoir from sample to sample. Introduces a dependency on the
        # sequence by which the samples are given

        # TODO: external function that is going to check the input dimensionality
        # and raise an error if shape is not correct

        if self._input_scaling_fit is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        if not isinstance(x, np.ndarray):
            raise TypeError("Input data must be a numpy array.")
        if x.ndim != 3:
            raise ValueError(
                "Input data must have 3 dimensions (n_batch, n_timesteps, n_states)."
            )
        self._check_channels(x, "x", self.input_layer.weights.shape[1], "input")
        self._check_finite(x, "x")
        if x.shape[1] <= self.num_transients_to_remove:
            raise ValueError(
                f"x has {x.shape[1]} time steps, but the warm-up is {self.num_transients_to_remove} "
                f"time steps: there must be at least one more to make a prediction."
            )

        # Same input preprocessing as in fit() (z-scoring and input scaling)
        x = self._transform_x(x)

        # Compute reservoir states. Returns reservoir states of shape
        # [n_batch, n_timesteps+1, n_nodes]
        # (n_timesteps+1 because we also store the initial state)
        reservoir_states = self.compute_reservoir_state(x)

        # discard the warm-up from the reservoir states, incl. the initial reservoir state: one
        # prediction for every input time step after the warm-up, shape (n_batch, n_time - d, n_nodes)
        del_mask = np.arange(0, self.num_transients_to_remove + 1)
        reservoir_states = np.delete(reservoir_states, del_mask, axis=1)

        # Masking non-readout nodes: if the user specified to not use all nodes for output, we can get rid of the non-readout node states
        reservoir_states = reservoir_states[:, :, self.readout_layer.readout_nodes]

        # make predictions y = R * W_out + b, W_out has a shape of [n_out, N]
        y_pred = np.einsum(
            "bik,jk->bij", reservoir_states, self.readout_layer.weights.T
        )
        return y_pred + self.readout_layer.bias
    

    @staticmethod
    def _check_channels(a: np.ndarray, name: str, expected: int, kind: str):
        if a.shape[-1] != expected:
            raise ValueError(
                f"{name} has {a.shape[-1]} {kind} channels (last axis), but the model has "
                f"{expected}. Build the model with matching layers, e.g. "
                f"InputLayer(input_shape=(n_time, {a.shape[-1]})) for a new data set."
            )

    @staticmethod
    def _check_finite(a: np.ndarray, name: str):
        bad = ~np.isfinite(a)
        if bad.any():
            first = tuple(int(i) for i in np.argwhere(bad)[0])
            raise ValueError(
                f"{name} contains {int(bad.sum())} NaN or infinite values (first at index {first})."
                + (
                    " Targets that were shifted by hand need their padding removed, e.g. "
                    "by leaving out the warm-up."
                    if name == "y"
                    else ""
                )
            )

    def _transform_x(self, x: np.ndarray, channels=None) -> np.ndarray:
        """
        Input preprocessing used for training and inference:
            x_tilde = input_scaling * (x - input_mean) / input_std
        (z-scoring only if enabled at fit time). Uses the settings frozen in fit().

        channels (optional): indices of the input channels that the last axis of x
        corresponds to. Used to map a signal that lives in physical units, such as a
        readout prediction that is fed back, into the coordinates of those inputs.
        """
        if self._input_scaling_fit is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        scaling = self._input_scaling_fit
        if channels is not None and np.ndim(scaling) == 1:
            scaling = scaling[channels]
        if self._normalize_inputs_fit:
            mean, std = self.input_mean, self.input_std
            if channels is not None:
                mean, std = mean[channels], std[channels]
            x = (x - mean) / std
        return x * scaling

    @staticmethod
    def _validate_input_scaling(scaling, n_channels: int = None):
        """Input scaling must be positive and finite: a scalar or one value per channel."""
        s = np.asarray(scaling, dtype=float)
        if s.ndim > 1:
            raise ValueError("input_scaling must be a scalar or a 1D array (one value per input channel).")
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("input_scaling must be positive and finite.")
        if s.ndim == 1 and n_channels is not None and s.shape[0] != n_channels:
            raise ValueError(
                f"input_scaling has {s.shape[0]} entries but the input has {n_channels} channels."
            )
        return float(s) if s.ndim == 0 else s

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, metrics: Union[str, list, None] = None
    ) -> list:
        r"""
        Score the predictions for ``x`` against the targets ``y``.

        Prediction :math:`j` is compared with target :math:`j`; targets are given without the warm-up.
        If there are fewer targets than predictions, only the first :math:`K` predictions are compared.

        Parameters
        ----------
        x : ndarray of shape (n_batch, T, n_in)
            Input sequences.
        y : ndarray of shape (n_batch, K, n_out)
            Targets, :math:`K \le T - d`.
        metrics : str or list of str, optional
            Metric names (see :mod:`pyreco.metrics`); default: the metrics given to :meth:`compile`.

        Returns
        -------
        list of float
            The metric values, in the order of ``metrics``.

        Raises
        ------
        ValueError
            If there are more targets than predictions (targets that still contain the warm-up).
        """
        # evaluate metrics on predictions made for input data
        # expects: x of shape [n_batch, n_timesteps, n_states]
        # expects: y of shape [n_batch, n_timesteps_out, n_states_out]
        # depends on self.metrics = metrics from .compile()
        # returns float, if multiple metrics, then in given order (TODO: implement this)

        if (
            metrics is None
        ):  # user did not specify metric, take the one(s) given to .compile()
            metrics = self.metrics
        if type(metrics) is str:  # make sure that we are working with lists of strings
            metrics = [metrics]

        # self.metrics_available = ['mse', 'mae        #
        # eval_metrics = self.metrics + metrics  # combine from .compile and user specified
        # eval_metrics = list(set(eval_metrics))  # removes potential duplicates

        # get metric function handle from the list of metrics specified as str
        metric_funs = [assign_metric(m) for m in metrics]

        # make predictions
        y_pred = self.predict(x=x)

        # prediction j belongs to target j. The predictions cover all input steps after the
        # warm-up, the targets may be shorter (e.g. a single target of a sequence)
        if y.shape[1] > y_pred.shape[1]:
            raise ValueError(
                f"The targets have {y.shape[1]} time steps but there are only "
                f"{y_pred.shape[1]} predictions (inputs: {x.shape[1]} time steps, warm-up: "
                f"{self.discard_transients}). Prediction j belongs to the target at time step j "
                f"after the warm-up, so leave out the first {self.discard_transients} targets "
                f"of a sequence that was given in full."
            )
        n_compare = min(y.shape[1], y_pred.shape[1])
        y, y_pred = y[:, :n_compare, :], y_pred[:, :n_compare, :]

        # get metric values
        metric_values = []
        for _metric_fun in metric_funs:
            metric_values.append(float(_metric_fun(y, y_pred)))

        return metric_values

    def _train_model(self, x: np.ndarray, y: np.ndarray):
        """
        Train the model with a single reservoir initialization.

        Args:
        x (np.ndarray): Input data of shape [n_batch, n_timesteps, n_states]
        y (np.ndarray): Target data of shape [n_batch, n_timesteps, n_states]

        Returns:
        dict: History of the training process
        """
        # extract shapes
        n_batch = x.shape[0]
        n_time_out, n_states_out = y.shape[1], y.shape[2]

        # Compute reservoir states. This is the most time-consuming part of the training process.
        # returns reservoir states of shape [n_batch, n_timesteps+1, n_nodes]
        # (n_timesteps+1 because we also store the initial state)
        reservoir_states = self.compute_reservoir_state(x)

        # discard the requested transients from the reservoir states, incl. initial reservoir state
        del_mask = np.arange(0, self.num_transients_to_remove + 1)
        reservoir_states = np.delete(reservoir_states, del_mask, axis=1)

        # keep the states that are paired with a target (the first n_time_out states after the warm-up)
        reservoir_states = reservoir_states[:, :n_time_out]

        # Masking non-readout nodes: if the user specified to not use all nodes for output, we can get rid of the non-readout node states
        # TODO: check if the readout nodes are in the correct order!!!
        reservoir_states = reservoir_states[:, :, self.readout_layer.readout_nodes]

        # Training: Solve regression problem y = R^T * W_out
        # First reshape reservoir states and targets such that we regress across
        # all batches and time steps
        b = y.reshape(n_batch * n_time_out, n_states_out)
        num_readout = len(self.readout_layer.readout_nodes)
        A = reservoir_states.reshape(n_batch * n_time_out, num_readout)

        # Readout intercept (bias): center the states and the targets before solving, so
        # that the offset of the targets is not forced through the reservoir states and
        # is not shrunk by the ridge penalty. The intercept is recovered afterwards:
        #   y = (R - R_mean) W + y_mean  =  R W + (y_mean - R_mean W)
        A_mean = A.mean(axis=0)
        b_mean = b.mean(axis=0)

        # Solve the regression problem b = Ax to find readout weights
        self.readout_layer.weights = self.optimizer.solve(A=A - A_mean, b=b - b_mean)

        # is there is only a single system state to predict, we need to add that dim
        # TODO: move this to the sanity checks and add an artificial dimension prior to fitting!
        if self.readout_layer.weights.ndim == 1:
            self.readout_layer.weights = np.expand_dims(
                self.readout_layer.weights, axis=-1
            )

        self.readout_layer.bias = b_mean - A_mean @ self.readout_layer.weights

        return reservoir_states, self.readout_layer.weights

    def remove_reservoir_nodes(self, nodes: list):
        """
        Remove reservoir nodes together with their read-in weights, bias signs and readout connections.

        The adjacency matrix loses the rows and columns of the nodes, so :math:`N` decreases. The readout
        has to be trained again with :meth:`fit`.

        Parameters
        ----------
        nodes : list of int
            Indices of the nodes to remove.

        Raises
        ------
        TypeError
            If ``nodes`` is not a list.
        ValueError
            If an index is out of range or all nodes would be removed.
        """

        # print(
        # f"removing nodes {nodes} from the reservoir. You need to retrain the model!"
        # )

        if not isinstance(nodes, list):
            raise TypeError("Nodes must be provided as a list of indices.")

        if np.max(nodes) > self.reservoir_layer.nodes:
            raise ValueError(
                f"Node index {np.max(nodes)} exceeds the number of nodes {self.reservoir_layer.nodes} in the reservoir."
            )

        if np.min(nodes) < 0:
            raise ValueError("Node index must be positive.")

        if len(nodes) >= self.reservoir_layer.nodes:
            raise ValueError("You cannot remove all nodes from the reservoir.")

        # 1. remove nodes from the reservoir layer
        self.reservoir_layer.remove_nodes(nodes)

        # 2. remove nodes from the read-in weights matrix of shape [num_nodes, num_states_in]
        self.input_layer.remove_nodes(nodes)

        # 2.b update input-receiving nodes in reservoir layer
        # This is opposite of how input is defined
        # non_zero_rows = np.all(self.input_layer.weights != 0, axis=1)
        # non_zero_row_indices = np.where(non_zero_rows)[0]
        # self.reservoir_layer.input_receiving_nodes = non_zero_row_indices

        # Remap surviving input-receiving node indices the same way as readout_nodes:
        # each node's new index = old index minus the number of removed nodes below it.
        self.reservoir_layer.input_receiving_nodes = rename_nodes_after_removal(
            original_nodes=list(self.reservoir_layer.input_receiving_nodes),
            removed_nodes=nodes
        )

        # 3. remove nodes from the list of readout-nodes in the readout layer
        # update the indices in the readout.readout_nodes list
        self.readout_layer.readout_nodes = rename_nodes_after_removal(
            original_nodes=self.readout_layer.readout_nodes, removed_nodes=nodes
        )
        self.readout_layer.update_layer_properties(self.reservoir_layer.nodes)

        # TODO: any more attributes to change here?

    def remove_reservoir_edges(self, edges: list):
        """
        Removes specific edges from the reservoir weights by setting them 0.

        Parameters
        ----------
        edges : list of tuple
            List of edges (u, v) tuples representing the edges to remove.

        Raises
        ------
        TypeError
            If 'edges' is not a list, or if the reservoir weights are
            neither a NetworkX graph nor a NumPy array.

        Notes
        -----
        For 'np.ndarray' weights, only 'weights[u, v]' is zeroed, the
        reverse direction 'weights[v, u]' is not adjusted. This is
        correct for directed graphs (the assumed case), but for an undirected graphs,
        leaving  the matrix asymmetric, with the edge still present in the reverse
        direction.
        """
        if not isinstance(edges, list):
            raise TypeError("Edges must be provided as a list of tuples (u, v).")

        # If weights are a NetworkX Graph
        if isinstance(self.reservoir_layer.weights, nx.Graph):
            self.reservoir_layer.weights.remove_edges_from(edges)

        # If weights are a NumPy Array (Adjacency Matrix)
        elif isinstance(self.reservoir_layer.weights, np.ndarray):
            for u, v in edges:
                self.reservoir_layer.weights[u, v] = 0
                # TODO If undirected, also remove the symmetric edge
                #       (we assume directed grpahs for now)
                # self.weights[v, u] = 0
            # keep density and spectral radius of the layer in sync with the weights
            # (the spectral radius is 0 once no directed cycle is left)
            self.reservoir_layer.update_layer_properties()
        # TODO check if input receving node can still access rest of the reservoir
        # TODO maybe make protection rules for pruner regarding input receveiving nodes,
        #       bias to output, etc.?
        else:
            raise TypeError("Reservoir weights must be NetworkX or NumPy array.")

    """
    The setter methods are used to set the parameters of the model.
    """    

    def _set_readin_weights(self, weights: Union[list, np.ndarray]):
        """
        Set the read-in weights matrix.

        Args:
        weights (Union[list, np.ndarray]): Read-in weights matrix.
        """
        # some sanity checks

        # type check
        if not isinstance(weights, np.ndarray) and not isinstance(weights, list):
            raise TypeError("Read-in weights matrix has to be a numpy array or a list.")

        if isinstance(weights, list):  # convert to np.array if list
            weights = np.array(weights)

        # read-in weights matrix has to have the shape [n_nodes, n_states_in]
        if weights.shape != (self.reservoir_layer.nodes, self.num_states_in):
            raise ValueError(
                f"Read-in weights matrix has to have the shape [n_nodes, n_states_in], i.e. {self.reservoir_layer.nodes}, {self.num_states_in}]"
            )

        # set the read-in weights in the input layer
        self.input_layer.weights = weights

    def _set_readout_nodes(self, nodes: Union[list, np.ndarray] = None):
        """
        Sets the nodes that will be linked to the output.

        Args:
        nodes (Union[list, np.ndarray], optional): Specific nodes to use for readout
        provided as indices. If None, randomly sample nodes.
        """
        if nodes is None:
            selector = NodeSelector(
                total_nodes=self.reservoir_layer.nodes,
                strategy="random_uniform_wo_repl",
                rng=self._rng_readout_nodes,
            )
            nodes = selector.select_nodes(fraction=self.readout_layer.fraction_out)
            if len(nodes) == 0:
                raise ValueError(
                    f"fraction_out={self.readout_layer.fraction_out} selects 0 of the "
                    f"{self.reservoir_layer.nodes} reservoir nodes to be read out. Use a larger "
                    "fraction_out or more nodes."
                )
        elif len(nodes) == 0:
            raise ValueError("At least one reservoir node has to be read out.")

        # set the readout nodes in the readout layer
        self.readout_layer.readout_nodes = nodes  # sorted(nodes)


    def _set_optimizer(self, optimizer: Union[str, Optimizer]):
        """
        Sets the optimizer that will find the readout weights.

        Args:
        optimizer (Union[str, Optimizer]): Name of the optimizer or an Optimizer
        instance.
        """
        self.optimizer = assign_optimizer(optimizer)

    def _set_metrics(self, metrics: Union[list, str]):
        """
        Sets the metric(s) for model evaluation.

        Args:
        metrics (Union[list, str]): List of metric names or a single metric name.
        """
        if isinstance(metrics, str):  # only single metric given
            self.metrics = [metrics]
        else:
            self.metrics = metrics  # if metrics is a list of strings.

        # assign the metric functions (callable) according to the metric names
        self.metrics_fun = []  # has been initialized, we reset it here
        for metric in self.metrics:
            self.metrics_fun.append(assign_metric(metric))

    def _set_init_states(self, init_states: Union[list, np.ndarray]):
        """
        Sets the initial states of the reservoir nodes.

        Args:
        init_states (np.ndarray, optional): Array of initial states. If None, sample
        initial states using the specified method.
        """

        # set the initial states in the reservoir layer
        self.reservoir_layer.set_initial_state(r_init=init_states)

    def _initialize_network(self, method: str = None):
        """
        Initialize the reservoir states.

        Args:
        method (str, optional): Method for sampling initial states. Default: the
        init_res_sampling of the reservoir layer (zeros unless chosen differently).
        """
        if method is None:
            method = self.reservoir_layer.init_res_sampling
        num_nodes = self.reservoir_layer.nodes
        initializer = NetworkInitializer(method=method, rng=self._rng_init_state)
        init_states = initializer.gen_initial_states(num_nodes)
        self._set_init_states(init_states=init_states)

    def _connect_input_to_reservoir(self):
        """
        Wire input layer with reservoir layer. Creates a random matrix of shape
        [nodes x n_states_in], i.e. number of reservoir nodes x state dimension of input.
        If no full connection is desired, a fraction of nodes will be selected according to the fraction_input parameter of the reservoir layer.

        """

        # generate random input connection matrix [nodes, n_states_in]
        shape = (self.num_nodes, self.num_states_in)
        distribution = self.input_layer.weight_distribution
        if distribution == "normal":
            net_generator = NetworkInitializer(
                method="random_normal", rng=self._rng_readin_weights
            )
            full_input_weights = net_generator.gen_initial_states(shape=shape)
        elif distribution == "uniform":
            full_input_weights = self._rng_readin_weights.uniform(-1.0, 1.0, size=shape)
        else:  # "sign": equal magnitude, random sign
            full_input_weights = self._rng_readin_weights.choice([-1.0, 1.0], size=shape)

        # signs of the input bias, always drawn so that the input weights and the bias do not
        # depend on whether the bias gain is switched on
        self.input_layer.bias = self._rng_bias.choice([-1.0, 1.0], size=self.num_nodes)

        # select read-in node indices according to the fraction specified by the user
        node_selector = NodeSelector(
            total_nodes=self.num_nodes,
            strategy="random_uniform_wo_repl",
            rng=self._rng_input_nodes,
        )

        # select the fraction of nodes that are input nodes [nodes, n_states_in]
        input_receiving_nodes = node_selector.select_nodes(
            fraction=self.reservoir_layer.fraction_input
        )

        if len(input_receiving_nodes) == 0:
            raise ValueError(
                f"fraction_input={self.reservoir_layer.fraction_input} selects 0 of the "
                f"{self.num_nodes} reservoir nodes to receive the input. Use a larger "
                "fraction_input or more nodes."
            )

        self.reservoir_layer.input_receiving_nodes = input_receiving_nodes
        node_mask = np.zeros_like(full_input_weights)
        node_mask[input_receiving_nodes] = 1

        # set the input layer weight matrix
        self._set_readin_weights(weights=(full_input_weights * node_mask))

    def _connect_feedback_to_reservoir(self):
        """
        Wire feedback layer with reservoir layer. Creates a random matrix of shape
        [nodes x n_states_fb], i.e. number of reservoir nodes x state dimension of feedback.
        If no full connection is desired, a fraction of nodes will be selected according to the fraction_input parameter of the reservoir layer.

        """

        # generate random feedback connection matrix [nodes, n_states_fb]
        # net_generator = NetworkInitializer(method="random_normal")
        # full_feedback_weights = net_generator.gen_initial_states(
        #     shape=(self.num_nodes, self.num_states_feedback)
        # )

        # # To DO: select feedback-receiving node indices according to the fraction specified by the user
        # node_selector = NodeSelector(
        #     total_nodes=self.num_nodes,
        #     strategy="random_uniform_wo_repl",
        # )

        # select the fraction of nodes that are feedback-receiving nodes [nodes, n_states_fb]
        # feedback_receiving_nodes = node_selector.select_nodes(
        #     fraction=self.reservoir_layer.fraction_input
        # )

        # self.reservoir_layer.input_receiving_nodes = input_receiving_nodes
        # node_mask = np.ones_like(full_input_weights)
        # node_mask[input_receiving_nodes] = 0

        # set the input layer weight matrix
        self.feedback_layer.weights = self.input_layer.weights

    def compute_reservoir_state(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the reservoir states for all samples at once.

        Iterates the update

        .. math::

            r(t+1) = (1 - a)\, r(t) + a\, g\!\left( r(t)\, A + \tilde x(t)\, W_\mathrm{in}^\top + \beta\, b^\top \right)

        from :math:`r(0)` for every sample, using the preprocessing and bias settings frozen by
        :meth:`fit` (the current settings before the first fit).

        Parameters
        ----------
        x : ndarray of shape (n_batch, T, n_in)
            Input sequences.

        Returns
        -------
        ndarray of shape (n_batch, T + 1, N)
            The states :math:`r(0), \dots, r(T)`.
        """

        # sanity checks for inputs
        if not isinstance(x, np.ndarray):
            raise TypeError("Input data must be a numpy array.")
        if x.ndim != 3:
            raise ValueError(
                "Input data must have 3 dimensions (n_batch, n_timesteps, n_states)."
            )

        # Extract shapes and parameters
        n_batch, n_time = x.shape[0], x.shape[1]

        # get local variables for easier access
        num_nodes = self.reservoir_layer.nodes
        activation = self.reservoir_layer.activation_fun
        alpha = self.reservoir_layer.leakage_rate  # leakage rate
        A = self.reservoir_layer.weights  # reservoir weight matrix (adjacency matrix)
        W_in = self.input_layer.weights  # read-in weight matrix

        # We will compute the reservoir states for all time steps in the first sample,
        # then reset the reservoir state to the initial values, and proceed with the
        # next sample. This makes sure to have no data leakage between samples.

        # Pre-allocate reservoir state matrix and initialize with initial states
        states = np.zeros((n_batch, n_time + 1, num_nodes))
        states[:, 0] = self.reservoir_layer.initial_res_states

        # vectorized computation of reservoir states over time steps

        # 1. compute dot(W_in, x) for all time steps across all batches
        # (can be done before the loop as it does not depend on the reservoir states)
        input_contrib = np.einsum(
            "ij,btj->bti", W_in, x
        )  # shape [n_batch, n_time, n_nodes]
        input_contrib = input_contrib + self._bias_contribution()

        # 2. now step through time to compute reservoir states
        for t in range(n_time):

            # compute r(t) @ A for all batches. Row-vector convention: A[u, v] is the
            # edge u -> v (sender u, receiver v), consistent with networkx.
            reservoir_contrib = np.einsum("ji,bj->bi", A, states[:, t])

            # now compute r(t+1) = (1-alpha) * r(t) + alpha * g(r(t) @ A + W_in @ x(t))
            states[:, t + 1] = (1 - alpha) * states[:, t] + alpha * activation(
                reservoir_contrib + input_contrib[:, t]
            )

        # flatten reservoir states along batch dimension:
        # [(n_batch * n_timesteps), num_nodes]
        # states[:, 1:].reshape(-1, num_nodes)
        return states

    def AutoRC_predict(self, x: np.ndarray, fb_scale: float, T_run: int, feedback_indices: np.ndarray = None) -> np.ndarray:
        r"""
        Run the AutoRC model for ``T_run`` steps with the previous output fed back into the reservoir.

        Experimental. At step :math:`t` the output is read from the current state,
        :math:`\hat y(t) = r_\mathrm{out}(t)\, W_\mathrm{out} + c`, and the state is updated with

        .. math::

            r(t+1) = (1 - a)\, r(t) + a\, g\!\left( r(t)\, A + \tilde x(t)\, W_\mathrm{in}^\top
            + \phi\; \tilde{\hat y}(t)\, W_\mathrm{fb}^\top \right) ,

        where :math:`\tilde x(t)` is the preprocessed input (zero after the end of the input sequence) and
        :math:`\tilde{\hat y}` the output in the coordinates of the input channels. The feedback term, scaled
        by :math:`\phi` (``fb_scale``), is active from step 500 on (hard-coded); before that the reservoir is
        driven by the input only.

        Parameters
        ----------
        x : ndarray of shape (n_batch, n_timesteps, n_states)
            Input data.
        fb_scale : float
            Scale :math:`\phi` of the fed-back prediction.
        T_run : int
            Number of time steps to run.
        feedback_indices : ndarray, optional
            Indices of the input channels that the outputs replace, i.e. the columns of the feedback weights
            that are used. If not given, all available feedback weights are used.

        Returns
        -------
        states : ndarray of shape (n_batch, T_run + 1, N)
            Reservoir states.
        predictions : ndarray of shape (n_batch, T_run - 1, n_states_out)
            The predictions for the steps 1, ..., T_run - 1.
        """

        # sanity checks for inputs
        if not isinstance(x, np.ndarray):
            raise TypeError("Input data must be a numpy array.")
        if x.ndim != 3:
            raise ValueError(
                "Input data must have 3 dimensions (n_batch, n_timesteps, n_states)."
            )

        # Same input preprocessing as in fit() (z-scoring and input scaling)
        x = self._transform_x(x)

        # Extract shapes and parameters
        n_batch, n_time, n_states = x.shape[0], x.shape[1], x.shape[2]

        # get local variables for easier access
        num_nodes = self.reservoir_layer.nodes
        activation = self.reservoir_layer.activation_fun
        alpha = self.reservoir_layer.leakage_rate  # leakage rate
        A = self.reservoir_layer.weights  # reservoir weight matrix (adjacency matrix)
        W_in = self.input_layer.weights  # read-in weight matrix
        W_fb = self.feedback_layer.weights

        # print("Input shape: ", x.shape)
        # print("W_in shape: ", W_in.shape)
        # print("W_fb shape: ", W_fb.shape)
        # print("is W_in == W_fb? ", np.array_equal(W_in, W_fb))
        # print("W_out shape: ", self.readout_layer.weights.shape)
        # We will compute the reservoir states for all time steps in the first sample,
        # then reset the reservoir state to the initial values, and proceed with the
        # next sample. This makes sure to have no data leakage between samples.

        # Pre-allocate reservoir state matrix and initialize with initial states
        states = np.zeros((n_batch, T_run + 1, num_nodes))
        states[:, 0] = self.reservoir_layer.initial_res_states
        # print("states shape: ", states.shape)
        # vectorized computation of reservoir states over time steps

        # 1. compute dot(W_in, x) for all time steps across all batches
        # (can be done before the loop as it does not depend on the reservoir states)
        input_contrib = np.einsum(
            "ij,btj->bti", W_in, x
        )  # shape [n_batch, n_time, n_nodes]
        print("input_contrib shape: ", input_contrib.shape)


        ### predictions to be used for feedback
        # predicted_states = ####self.AutoRC_predict(x)
        
        ###Initialize feedback contribution same as input x
        y_pred = []#np.zeros((n_batch, n_time, n_states))
        # print("Initialized y_pred shape: ", y_pred.shape)
        
        ###If feedback indices are not given by the user, the model will use all available feedback weights.
        if feedback_indices is None:
            feedback_indices = np.arange(self.feedback_layer.weights.shape[1])

        feedback_contrib = 0# p.einsum("bik,jk->bij", self.feedback_layer.weights, y_pred[:, 0, :])  # shape [n_batch, n_time, n_nodes]
        # print("feedback_contrib shape: ", feedback_contrib.shape)
        print('Input layer weights shape: ', self.input_layer.weights.shape)
        print('Feedback layer weights shape: ', self.feedback_layer.weights.shape)
        print('Readout layer weights shape: ', self.readout_layer.weights.shape)
        # 2. now step through time to compute reservoir states
        
        for t in range(0, T_run):

            ###condition for input contribution
            if t<input_contrib.shape[1]:
                input_contrib_t = input_contrib[:, t]
            else:
                input_contrib_t = 0
                
            
            #############################################          
            y_pred_t = np.einsum("bik,jk->bij", states[:, t: t+1, self.readout_layer.readout_nodes], self.readout_layer.weights.T) + self.readout_layer.bias
            
            ####feedback contribution
            if t >= 500:#input_contrib.shape[1]:
                # the readout predicts in physical units; map the fed-back signal into
                # the coordinates of the inputs it replaces (z-scoring and scaling)
                fb_signal = self._transform_x(y_pred_t[:, 0, :], channels=feedback_indices)
                feedback_contrib = np.einsum("ij,bj->bi", fb_signal, self.feedback_layer.weights[:, feedback_indices]).T
            else:
                feedback_contrib = 0
            # print("feedback_contrib shape am ende: ", feedback_contrib.shape)
            ### store the predictions for the current time step
            y_pred.append(y_pred_t[:,0,:])
            #############################################

            # compute r(t) @ A for all batches (A[u, v] is the edge u -> v)
            reservoir_contrib = np.einsum("ji,bj->bi", A, states[:, t])

            # now compute r(t+1) = (1-alpha) * r(t) + alpha * g(r(t) @ A + W_in @ x(t))
            states[:, t + 1] = (1 - alpha) * states[:, t] + alpha * activation(
                reservoir_contrib + input_contrib_t + fb_scale * feedback_contrib
            )

            ####make predictions for feedback already here
            # Masking non-readout nodes: if the user specified to not use all nodes for output, we can get rid of the non-readout node states
            # and using current t+1 res. state for prediction
            
            # states_t = states[:, t+1 : t+2, self.readout_layer.readout_nodes]

            # print("time step t: ", t)
            # print("Input contribution at t: ", input_contrib[:, t].shape)
            # print("states_t shape after masking: ", states_t.shape)
            # make predictions y = R * W_out, W_out has a shape of [n_out, N]
            

        # flatten reservoir states along batch dimension:
        # [(n_batch * n_timesteps), num_nodes]
        # states[:, 1:].reshape(-1, num_nodes)
        y_pred = np.array(y_pred)[1:,:, :]
        print("Final y_pred shape: ", y_pred.shape)
        return states, y_pred.transpose(1,0,2)

    def fit_evolve(self, X: np.ndarray, y: np.ndarray):
        """Placeholder for an evolving reservoir computer (performance-dependent node addition and removal); not implemented, returns None."""

        history = None
        return history

    # # @abstractmethod
    # def get_params(self, deep=True):
    #     """
    #     Get parameters for scikit-learn compatibility.

    #     Args:
    #     deep (bool): If True, return a deep copy of parameters.

    #     Returns:
    #     dict: Dictionary of model parameters.
    #     """
    #     # needed for scikit-learn compatibility
    #     return {
    #         "input_layer": self.input_layer,
    #         "reservoir_layer": self.reservoir_layer,
    #         "readout_layer": self.readout_layer,
    #     }

    # @abstractmethod
    def save(self, path: str):
        """
        Store the model to disk.

        Args:
        path (str): Path to save the model.
        """
        # store the model to disk
        raise NotImplementedError("Method not implemented yet.")

    def plot(self, path: str):
        """
        Print the model to some figure file.

        Args:
        path (str): Path to save the figure.
        """
        # print the model to some figure file
        raise NotImplementedError("Method not implemented yet.")

    def set_normalization(self, normalize_inputs: bool = None):
        r"""
        Enable or disable z-scoring of the inputs (default: enabled).

        With z-scoring every input channel is shifted and divided by its mean :math:`\mu` and standard
        deviation :math:`\sigma`, computed over all batches and time steps of the training data in
        :meth:`fit`; :math:`\tilde x = (x - \mu) / \sigma`. The same statistics are used in :meth:`predict`.
        Takes effect at the next :meth:`fit`.

        Parameters
        ----------
        normalize_inputs : bool, optional
            None leaves the setting unchanged.

        Raises
        ------
        TypeError
            If the argument is not a boolean.
        """
        if normalize_inputs is not None:
            if not isinstance(normalize_inputs, (bool, np.bool_)):
                raise TypeError("normalize_inputs must be a boolean.")
            self.normalize_inputs = bool(normalize_inputs)

    def set_input_scaling(self, input_scaling: Union[float, np.ndarray]):
        r"""
        Set the input scaling: a positive gain :math:`s` multiplied onto the (z-scored) inputs.

        This is equivalent to scaling the columns of the read-in matrix, :math:`W_\mathrm{in} \leftarrow W_\mathrm{in} \, \mathrm{diag}(s)`.
        Takes effect at the next :meth:`fit`.

        Parameters
        ----------
        input_scaling : float or ndarray
            A scalar or one value per input channel; all values positive and finite.

        Raises
        ------
        ValueError
            For non-positive, non-finite or mis-shaped values.
        """
        self.input_scaling = self._validate_input_scaling(input_scaling)

    def set_input_bias(self, input_bias: float):
        r"""
        Set the gain :math:`\beta \ge 0` of the constant input :math:`\beta b` to the reservoir nodes (0 = no bias).

        Takes effect at the next :meth:`fit`.

        Parameters
        ----------
        input_bias : float
            The gain; the signs :math:`b` are fixed by :meth:`compile`.

        Raises
        ------
        TypeError
            If the value is not a number.
        ValueError
            If it is negative or not finite.
        """
        self.input_bias = self._validate_input_bias(input_bias)

    @staticmethod
    def _validate_input_bias(input_bias) -> float:
        if isinstance(input_bias, bool) or not isinstance(input_bias, (int, float, np.number)):
            raise TypeError("input_bias must be a number.")
        if not np.isfinite(input_bias) or input_bias < 0:
            raise ValueError("input_bias must be finite and >= 0 (0 = no bias).")
        return float(input_bias)

    def _bias_contribution(self):
        """Constant input to the reservoir nodes, shape (n_nodes,): input_bias * b (or 0)."""
        gain = self._input_bias_fit if self._input_bias_fit is not None else self.input_bias
        bias = getattr(self.input_layer, "bias", None)
        if gain == 0 or bias is None:
            return 0.0
        return gain * bias

    def set_hp(self, **kwargs):
        r"""
        Set one or more hyperparameters; they are the arguments that hyperparameter searches vary.

        Parameters
        ----------
        spec_rad : float, optional
            Rescales the reservoir to spectral radius :math:`\rho(A)` (see :meth:`pyreco.layers.ReservoirLayer.set_spec_rad`).
        leakage_rate : float, optional
            Leakage rate :math:`a \in (0, 1]`.
        activation : {"tanh", "sigmoid"}, optional
            Activation function :math:`g`.
        input_scaling : float or ndarray, optional
            Input scaling :math:`s`, applied from the next :meth:`fit`.
        input_bias : float, optional
            Gain :math:`\beta` of the input bias, applied from the next :meth:`fit`.
        alpha : float, optional
            Ridge penalty :math:`\alpha \ge 0` of the readout.

        Raises
        ------
        ValueError
            For an unsupported name or an invalid value.
        """
        supported_hps = {
            "spec_rad", "leakage_rate", "activation", "input_scaling", "input_bias", "alpha"
        }
        unsupported = set(kwargs) - supported_hps
        if unsupported:
            raise ValueError(f"Unsupported hyperparameter(s): {', '.join(unsupported)}")

        if 'spec_rad' in kwargs:
            self.reservoir_layer.set_spec_rad(kwargs['spec_rad'])
        if 'leakage_rate' in kwargs:
            self.reservoir_layer.set_leakage_rate(kwargs['leakage_rate'])
        if 'activation' in kwargs:
            self.reservoir_layer.set_activation(kwargs['activation'])
        if "input_scaling" in kwargs:
            self.set_input_scaling(kwargs['input_scaling'])
        if "input_bias" in kwargs:
            self.set_input_bias(kwargs["input_bias"])
        if "alpha" in kwargs:
            self.optimizer.set_alpha(kwargs['alpha'])
        # Add more as needed

    def get_hp(self, *args):
        """
        Get hyperparameters by name, e.g. ``get_hp("spec_rad", "activation")``.

        Parameters
        ----------
        *args : str
            Names as in :meth:`set_hp`. Without names, all hyperparameters are returned.

        Returns
        -------
        dict
            Name to current value.
        """
        all_hps = {
            'spec_rad': self.reservoir_layer.get_spec_rad(),
            'leakage_rate': self.reservoir_layer.get_leakage_rate(),
            'activation': self.reservoir_layer.get_activation(),
            "input_scaling": self.input_scaling,
            "input_bias": self.input_bias,
            "alpha": self.optimizer.get_alpha()
            # Add more as needed
        }
        if args:
            return {hp: all_hps[hp] for hp in args if hp in all_hps}
        else:
            return all_hps


class RC(CustomModel):  # the non-auto version
    """
    Non-autonomous reservoir computer: the output is a function of the input sequence only.

    Use it for forecasting from windows of data and for input/output tasks; see :class:`CustomModel`
    for the equations and the methods.

    Examples
    --------
    >>> import numpy as np
    >>> from pyreco.custom_models import RC
    >>> from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
    >>> x = np.random.default_rng(0).normal(size=(20, 30, 1))
    >>> y = np.tanh(x)
    >>> model = RC(seed=0)
    >>> model.add(InputLayer(input_shape=(30, 1)))
    >>> model.add(RandomReservoirLayer(nodes=50, seed=0))
    >>> model.add(ReadoutLayer(output_shape=(30, 1)))
    >>> model.compile(metrics=["nrmse"])
    >>> _ = model.fit(x, y)
    >>> model.predict(x).shape
    (20, 30, 1)
    """

    def __init__(self, seed: int = None):
        super().__init__(seed=seed)


class AutoRC(CustomModel):
    """
    Autonomous version of the reservoir computer.
    """

    def __init__(self, FeedbackLayer=None, seed: int = None):
        super().__init__(seed=seed)
        self.feedback_layer = FeedbackLayer
        # if self.feedback_layer is None:
        #     raise ValueError("FeedbackLayer must be provided for AutoRC.")
                

    def predict_ar(self, X: np.ndarray, n_steps: int = 10):
        """
        Perform auto-regressive prediction (time series forecasting).

        Args:
        X (np.ndarray): Initial input data.
        n_steps (int): Number of steps to predict into the future.

        Returns:
        np.ndarray: Predicted future states.
        """
        pass


class HybridRC(CustomModel):
    """
    Hybrid version of the reservoir computer.
    """

    def __init__(self):
        pass


if __name__ == "__main__":
    print("hello")
