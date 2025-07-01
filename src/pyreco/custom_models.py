import numpy as np
from abc import ABC
from typing import Union
import copy
import multiprocessing
from functools import partial

from pyreco.layers import (
    Layer,
    InputLayer,
    ReservoirLayer,
    ReadoutLayer,
)
from pyreco.optimizers import Optimizer, assign_optimizer
from pyreco.metrics import assign_metric
from pyreco.node_selector import NodeSelector
from pyreco.initializer import NetworkInitializer
from pyreco.utils_networks import rename_nodes_after_removal


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
    """
    Abstract base class for custom reservoir computing model.

    Has a syntax similar to the one of TensorFlow model API,
    e.g. using the model.add() statement to add layers to the model.

    A model hast an input layer, a reservoir layer and a readout layer.
    """

    def __init__(self):
        """
        Initialize the CustomModel with empty layers and default values.
        """
        # Initialize layers
        self.input_layer: InputLayer
        self.reservoir_layer: ReservoirLayer
        self.readout_layer: ReadoutLayer

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

        # Normalization flags
        self.normalize_inputs: bool = True
        self.normalize_outputs: bool = True

        # Scaling parameters (apply after normalization)
        self.input_scaling: float or np.ndarray = 1.0  # scalar or (d,) vector
        self.output_scaling: float = 1.0  # usually scalar

        # Stored normalization parameters (learned during training)
        self.input_mean: np.ndarray = None  # shape: (d,)
        self.input_std: np.ndarray = None  # shape: (d,)
        self.output_mean: np.ndarray = None  # shape: (k,)
        self.output_std: np.ndarray = None  # shape: (k,)

    def add(self, layer: Layer):
        """
        Add a layer to the model.

        Is type-sensitive and will assign the layer to the correct attribute.

        Args:
        layer (Layer): Layer to be added to the model.
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
        else:
            raise ValueError("Unsupported layer type.")

    # TODO: the following method should be implemented in the CustomModel class
    #   def _set_readin_nodes(self, nodes: Union[list, np.ndarray] = None):

    def compile(
        self,
        optimizer: str = "ridge",
        metrics: Union[list, str] = None,
        discard_transients: int = 0,
    ):
        """
        Configure the model for training.

        Args:
        optimizer (str): Name of the optimizer.
        metrics (list): List of metric names.
        discard_transients (int): Number of initial transient timesteps to discard.
        """

        # sanity checks for inputs
        if not isinstance(discard_transients, int):
            raise TypeError("discard_transients must be a positive integer!")
        elif discard_transients < 0:
            raise ValueError("discard_transients must be >=0")

        if metrics is None:
            metrics = ["mse"]

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

        # copy some layer properties to the model level for easier access
        self.num_states_in = self.input_layer.n_states
        self.num_states_out = self.readout_layer.n_states
        self.num_nodes = self.reservoir_layer.nodes

        # Sample the input connections: create W_in read-in weight matrix
        self._connect_input_to_reservoir()  # check for dependency injection here!

        # Set initial states of the reservoir
        # TODO: let the user specify the reservoir initialization method
        self._initialize_network(method="random_normal")

        # Select readout nodes according to the fraction specified by the user in the readout layer. By default, randomly sample nodes. User can also provide a list of nodes to use for readout.
        self._set_readout_nodes()

        # flag all layers as compiled (required for later manipulation from outside)
        self.input_layer._is_compiled = True
        self.reservoir_layer._is_compiled = True
        self.readout_layer._is_compiled = True

    def fit(self, x: np.ndarray,
            y: np.ndarray,
            n_init: int = 1,
            store_states: bool = False
            ) -> dict:
        """
        RC training with batch processing.
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
        if not isinstance(n_init, int):
            raise TypeError("Number of initializations must be an integer (positive).")
        if n_init < 1:
            raise ValueError("Number of initializations must be at least 1.")
        if not isinstance(store_states, bool):
            raise TypeError("store_states must be a boolean.")

        # Extract shapes of inputs and outputs, expected to be 3D arrays
        # of shape [n_batch, n_timesteps, n_features]
        n_batch, n_time_in = x.shape[0], x.shape[1]
        n_time_out, n_states_out = y.shape[-2], y.shape[-1]
        n_nodes = self.reservoir_layer.nodes

        # Input normalization (z-score per channel)

        if self.normalize_inputs:
            self.input_mean = x.mean(axis=(0, 1))  # mean over batches and time
            self.input_std = x.std(axis=(0, 1))
            self.input_std[self.input_std == 0] = 1.0  # avoid divide-by-zero
            x = (x - self.input_mean) / self.input_std


        # Output normalization (z-score per channel)

        if self.normalize_outputs:
            self.output_mean = y.mean(axis=(0, 1))  # mean over batches and time
            self.output_std = y.std(axis=(0, 1))
            self.output_std[self.output_std == 0] = 1.0
            y = (y - self.output_mean) / self.output_std

        # Apply input scaling
        x = x * self.input_scaling  # scalar or (features,)

        # Apply output scaling (optional, rare)
        y = y * self.output_scaling

        # discard transients (warmup phase). This is done by removing the first n_transients timesteps from the reservoir states.
        # Hence, the targets can have a maximum of (t_in - t_discard) steps, before we have to cut also from the targets
        # If the number of t_out steps is even smaller, we will discard more steps from the reservoir states
        self.num_transients_to_remove = 0
        if self.discard_transients >= n_time_in:
            raise ValueError(
                f"Number of transients to discard ({self.discard_transients}) must be smaller than the number of time steps in the input data ({n_time_in})."
            )
        if (n_time_in - self.discard_transients) < n_time_out:
            print(
                f"Discarding {self.discard_transients} time steps will reduce the number of output time steps to {n_time_in-self.discard_transients}. The given targets had {n_time_out} time steps."
            )
            # cut first steps from the targets to match the desired warmup phase
            y = y[:, self.discard_transients :, :]
            n_time_out = n_time_in - self.discard_transients
            self.num_transients_to_remove = self.discard_transients

        if (n_time_in - self.discard_transients) > n_time_out:
            # enlarge the number of transients to discard to match the output shape
            self.num_transients_to_remove = n_time_in - n_time_out
            print(
                f"discarding {self.num_transients_to_remove} reservoir states to match the number of time steps on the output."
            )

        # update some class attributes that depend on the training data
        self.num_time_in = n_time_in
        self.num_states_in = x.shape[-1]
        self.num_time_out = n_time_out
        self.num_states_out = n_states_out

        # Pre-allocate arrays for storing results
        n_R0 = np.zeros((n_init, n_nodes))
        n_weights = np.zeros(
            (n_init, len(self.readout_layer.readout_nodes), n_states_out)
        )
        n_scores = np.zeros(n_init)
        n_res_states = [] if store_states else None

        # Get metric functions that scores the model performance
        metric_fun = (
            self.metrics_fun[0]
            if self.metrics_fun
            else assign_metric("mean_squared_error")
        )

        # Batch process multiple reservoir initializations
        for i in range(n_init):
            if n_init > 1:
                print(f"initialization {i}/{n_init}: computing reservoir states")

            # train the model on the given data
            reservoir_states, _ = self._train_model(x=x, y=y)

            # Tracking the variation of initial conditions
            n_R0[i] = self.reservoir_layer.initial_res_states
            n_weights[i] = self.readout_layer.weights
            n_scores[i] = metric_fun(y, self.predict(x=x))
            if store_states:
                n_res_states.append(reservoir_states)

        # Select best initialization
        idx_optimal = np.argmin(n_scores)
        self.reservoir_layer.set_initial_state(n_R0[idx_optimal])
        self.readout_layer.weights = n_weights[idx_optimal]

        # Update trainable weights count
        self.num_trainable_weights = self.reservoir_layer.weights.size

        # Build history dictionary
        history = {
            "init_res_states": n_R0,
            "readout_weights": n_weights,
            "train_scores": n_scores,
        }

        if store_states:
            history["res_states"] = n_res_states

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions for given input.

        Args:
        x (np.ndarray): Input data of shape [n_batch, n_timestep, n_states]
        one_shot (bool): If True, don't re-initialize reservoir between samples.

        Returns:
        np.ndarray: Predictions of shape [n_batch, n_timestep, n_states]
        """
        # makes prediction for given input (single-step prediction)
        # expects inputs of shape [n_batch, n_timestep, n_states]
        # returns predictions in shape of [n_batch, n_timestep, n_states]

        # one_shot = True will *not* re-initialize the reservoir from sample to sample. Introduces a dependency on the
        # sequence by which the samples are given

        # TODO: external function that is going to check the input dimensionality
        # and raise an error if shape is not correct

        # Compute reservoir states. Returns reservoir states of shape
        # [n_batch, n_timesteps+1, n_nodes]
        # (n_timesteps+1 because we also store the initial state)
        # Normalize input (if enabled)
        if self.normalize_inputs:
            if self.input_mean is None or self.input_std is None:
                raise RuntimeError("Input normalization parameters not set. Fit the model first.")
            x = (x - self.input_mean) / self.input_std

        reservoir_states = self.compute_reservoir_state(x)

        # discard the given transients from the reservoir states, incl. initial reservoir state. Should give the size of (n_batch, n_time_out, n_nodes)
        del_mask = np.arange(0, self.num_transients_to_remove + 1)
        reservoir_states = np.delete(reservoir_states, del_mask, axis=1)

        # Masking non-readout nodes: if the user specified to not use all nodes for output, we can get rid of the non-readout node states
        reservoir_states = reservoir_states[:, :, self.readout_layer.readout_nodes]

        # make predictions y = R * W_out, W_out has a shape of [n_out, N]
        y_pred = np.einsum(
            "bik,jk->bij", reservoir_states, self.readout_layer.weights.T
        )
        # Undo output scaling (if any)
        y_pred = y_pred / self.output_scaling

        # -------------------------------
        # Denormalize output
        # -------------------------------
        if self.normalize_outputs:
            if self.output_mean is None or self.output_std is None:
                raise RuntimeError("Outpaut normalization parameters not set. Fit the model first.")
            y_pred = y_pred * self.output_std + self.output_mean

        return y_pred

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, metrics: Union[str, list, None] = None
    ) -> tuple:
        """
        Evaluate metrics on predictions made for input data.

        Args:
        x (np.ndarray): Input data of shape [n_batch, n_timesteps, n_states]
        y (np.ndarray): Target data of shape [n_batch, n_timesteps_out, n_states_out]
        metrics (Union[str, list, None], optional): List of metric names or a single metric name. If None, use metrics from .compile()

        Returns:
        tuple: Metric values
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

        # remove the time steps that were discarded during training (transient removal, generic seq-to-seq modeling)
        n_time_out = y_pred.shape[-2]
        y = y[:, -n_time_out:, :]
        # if self.discard_transients > 0:
        # y = y[:, self.discard_transients :, :]

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

        # discard the requested transients from the reservoir states, incl. initial reservoir state. Should give the size of (n_batch, n_time_out, n_nodes)
        # self.num_transients_to_remove
        del_mask = np.arange(0, self.num_transients_to_remove + 1)
        # del_mask = np.arange(0, self.discard_transients + 1)
        reservoir_states = np.delete(reservoir_states, del_mask, axis=1)

        # # now select only the reservoir states that are needed for the output
        # reservoir_states = reservoir_states[:, -self.num_time_out :, :]

        # Masking non-readout nodes: if the user specified to not use all nodes for output, we can get rid of the non-readout node states
        # TODO: check if the readout nodes are in the correct order!!!
        reservoir_states = reservoir_states[:, :, self.readout_layer.readout_nodes]

        # Training: Solve regression problem y = R^T * W_out
        # First reshape reservoir states and targets such that we regress across
        # all batches and time steps
        b = y.reshape(n_batch * n_time_out, n_states_out)
        num_readout = len(self.readout_layer.readout_nodes)
        A = reservoir_states.reshape(n_batch * n_time_out, num_readout)

        # Solve the regression problem b = Ax to find readout weights
        self.readout_layer.weights = self.optimizer.solve(A=A, b=b)

        # is there is only a single system state to predict, we need to add that dim
        # TODO: move this to the sanity checks and add an artificial dimension prior to fitting!
        if self.readout_layer.weights.ndim == 1:
            self.readout_layer.weights = np.expand_dims(
                self.readout_layer.weights, axis=-1
            )

        return reservoir_states, self.readout_layer.weights

    def remove_reservoir_nodes(self, nodes: list):
        # removes specific nodes from the reservoir matrix, and
        # deletes accordingly the relevant readin weights and readout weights

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
        # Find non-zero rows
        non_zero_rows = np.all(self.input_layer.weights != 0, axis=1)

        # Get the indices of zero rows
        non_zero_row_indices = np.where(non_zero_rows)[0]
        self.reservoir_layer.input_receiving_nodes = non_zero_row_indices

        # 3. remove nodes from the list of readout-nodes in the readout layer
        # update the indices in the readout.readout_nodes list
        self.readout_layer.readout_nodes = rename_nodes_after_removal(
            original_nodes=self.readout_layer.readout_nodes, removed_nodes=nodes
        )
        self.readout_layer.update_layer_properties()

        # TODO: any more attributes to change here?

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
            )
            nodes = selector.select_nodes(fraction=self.readout_layer.fraction_out)

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

    def _initialize_network(self, method: str = "random_normal"):
        """
        Initialize the reservoir states.

        Args:
        method (str, optional): Method for sampling initial states.
        """
        num_nodes = self.reservoir_layer.nodes
        initializer = NetworkInitializer(method=method)
        init_states = initializer.gen_initial_states(num_nodes)
        self._set_init_states(init_states=init_states)

    def _connect_input_to_reservoir(self):
        """
        Wire input layer with reservoir layer. Creates a random matrix of shape
        [nodes x n_states_in], i.e. number of reservoir nodes x state dimension of input.
        If no full connection is desired, a fraction of nodes will be selected according to the fraction_input parameter of the reservoir layer.

        """

        # generate random input connection matrix [nodes, n_states_in]
        net_generator = NetworkInitializer(method="random_normal")
        full_input_weights = net_generator.gen_initial_states(
            shape=(self.num_nodes, self.num_states_in)
        )

        # select read-in node indices according to the fraction specified by the user
        node_selector = NodeSelector(
            total_nodes=self.num_nodes,
            strategy="random_uniform_wo_repl",
        )

        # select the fraction of nodes that are input nodes [nodes, n_states_in]
        input_receiving_nodes = node_selector.select_nodes(
            fraction=self.reservoir_layer.fraction_input
        )

        self.reservoir_layer.input_receiving_nodes = input_receiving_nodes
        node_mask = np.ones_like(full_input_weights)
        node_mask[input_receiving_nodes] = 0

        # set the input layer weight matrix
        self._set_readin_weights(weights=(full_input_weights * node_mask))

    def compute_reservoir_state(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of reservoir states with batch processing.

        Args:
            x (np.ndarray): Input data of shape [n_batch, n_timesteps, n_states]

        Returns:
            np.ndarray: Reservoir states of shape [(n_batch * n_timesteps), N]
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

        # 2. now step through time to compute reservoir states
        for t in range(n_time):

            # compute dot(A, r(t)) for all batches
            reservoir_contrib = np.einsum("ij,bj->bi", A, states[:, t])

            # now compute r(t+1) = (1-alpha) * r(t) + alpha * g(r(t) * A + x * W_in)
            states[:, t + 1] = (1 - alpha) * states[:, t] + alpha * activation(
                reservoir_contrib + input_contrib[:, t]
            )

        # flatten reservoir states along batch dimension:
        # [(n_batch * n_timesteps), num_nodes]
        # states[:, 1:].reshape(-1, num_nodes)
        return states

    def fit_evolve(self, X: np.ndarray, y: np.ndarray):
        # build an evolving reservoir computer: performance-dependent node addition and removal

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

    def set_normalization(
            self,
            normalize_inputs: bool = None,
            normalize_outputs: bool = None,
    ):
        """
        Set normalization flags independently.

        Args:
            normalize_inputs (bool or None): Enable/disable input normalization. If None, leave unchanged.
            normalize_outputs (bool or None): Enable/disable output normalization. If None, leave unchanged.
        """
        if normalize_inputs is not None:
            self.normalize_inputs = normalize_inputs
        if normalize_outputs is not None:
            self.normalize_outputs = normalize_outputs

    def set_input_scaling(self, input_scaling: float or np.ndarray):
        """
        Set input scaling factor.

        Args:
            input_scaling (float or np.ndarray): Scalar or per-channel vector.
        """
        self.input_scaling = input_scaling

    def set_output_scaling(self, output_scaling: float):
        """
        Set output scaling factor.

        Args:
            output_scaling (float): Scalar value for output scaling.
        """
        self.output_scaling = output_scaling

    def set_hp(self, **kwargs):
        """
        Set one or more reservoir hyperparameters.
        Supported kwargs: spec_rad, leakage_rate, activation
        """
        supported_hps = {"spec_rad", "leakage_rate", "activation"}
        unsupported = set(kwargs) - supported_hps
        if unsupported:
            raise ValueError(f"Unsupported hyperparameter(s): {', '.join(unsupported)}")

        if 'spec_rad' in kwargs:
            self.reservoir_layer.set_spec_rad(kwargs['spec_rad'])
        if 'leakage_rate' in kwargs:
            self.reservoir_layer.set_leakage_rate(kwargs['leakage_rate'])
        if 'activation' in kwargs:
            self.reservoir_layer.set_activation(kwargs['activation'])
        # Add more as needed

    def get_hp(self, *args):
        """
        Get one or more reservoir hyperparameters. If no args, returns all.
        Usage: get_hp('spec_rad', 'activation') or get_hp()
        """
        all_hps = {
            'spec_rad': self.reservoir_layer.get_spec_rad(),
            'leakage_rate': self.reservoir_layer.get_leakage_rate(),
            'activation': self.reservoir_layer.get_activation(),
            # Add more as needed
        }
        if args:
            return {hp: all_hps[hp] for hp in args if hp in all_hps}
        else:
            return all_hps


class RC(CustomModel):  # the non-auto version
    """
    Non-autonomous version of the reservoir computer.
    """

    def __init__(self):
        # at the moment we do not have any arguments to pass
        super().__init__()


class AutoRC(CustomModel):
    """
    Autonomous version of the reservoir computer.
    """

    def __init__(self):
        pass

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
