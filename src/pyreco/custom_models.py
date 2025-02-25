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

        self.n_time_in: int
        self.n_time_out: int
        self.n_states_in: int
        self.n_states_out: int

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
        metrics: list = ["mse"],
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

        # Select readout nodes according to the fraction specified by the user in the readout layer. By default, randomly sample nodes. User can also provide a list of nodes to use for readout.
        self._set_readout_nodes()

    def fit(
        self, x: np.ndarray, y: np.ndarray, n_init: int = 1, store_states: bool = False
    ):
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
        n_nodes = self.num_nodes

        # discard transients (warmup phase). This is done by removing the first n_transients timesteps from the reservoir states.
        # Hence, the targets can have a maximum of (t_in - t_discard) steps, before we have to cut also from the targets
        # If the number of t_out steps is even smaller, we will discard more steps from the reservoir states
        if self.discard_transients >= n_time_in:
            raise ValueError(
                f"Number of transients to discard ({self.discard_transients}) must be smaller than the number of time steps in the input data ({n_time_in})."
            )
        if (n_time_in - self.discard_transients) < n_time_out:
            print(
                f"Discarding {self.discard_transients} time steps will reduce the number of output time steps to {n_time_in-self.discard_transients}. The given targets had {n_time_out} time steps."
            )
            # cut first steps from the targets to match the desired warmup phase
            y = y[:, (n_time_in - self.discard_transients) :, :]
            n_time_out = n_time_in - self.discard_transients

        if (n_time_in - self.discard_transients) > n_time_out:
            # enlarge the number of transients to discard to match the output shape
            self.discard_transients = n_time_in - n_time_out
            print(
                f"discarding {self.discard_transients} reservoir states to match the number of time steps on the output."
            )

        # update some class attributes that depend on the training data
        self.n_time_in = n_time_in
        self.n_states_in = x.shape[-1]
        self.n_time_out = n_time_out
        self.n_states_out = n_states_out

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

        # Set initial states of the reservoir
        # TODO: let the user specify the reservoir initialization method
        self._initialize_network(method="random_normal")

        # Compute reservoir states. This is the most time-consuming part of the training process.
        # returns reservoir states of shape [n_batch, n_timesteps+1, n_nodes]
        # (n_timesteps+1 because we also store the initial state)
        reservoir_states = self.compute_reservoir_state(x)

        # discard the given transients from the reservoir states, incl. initial reservoir state. Should give the size of (n_batch, n_time_out, n_nodes)
        del_mask = np.arange(0, self.discard_transients + 1)
        reservoir_states = np.delete(reservoir_states, del_mask, axis=1)

        # Masking non-readout nodes: if the user specified to not use all nodes for output, we can get rid of the non-readout node states
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
                f"Read-in weights matrix has to have the shape [n_nodes, n_states_in], i.e. {self.reservoir_layer.nodes}, {self.n_states_in}]"
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
        self.readout_layer.readout_nodes = nodes

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
            graph=full_input_weights,
            strategy="random_uniform_wo_repl",
        )

        # select the fraction of nodes that are input nodes [nodes, n_states_in]
        node_mask = node_selector.select_nodes(
            fraction=self.reservoir_layer.fraction_input
        )

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

    # @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions for given input (single-step prediction).

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
        reservoir_states = self.compute_reservoir_state(x)

        # discard the given transients from the reservoir states, incl. initial reservoir state. Should give the size of (n_batch, n_time_out, n_nodes)
        del_mask = np.arange(0, self.discard_transients + 1)
        reservoir_states = np.delete(reservoir_states, del_mask, axis=1)

        # Masking non-readout nodes: if the user specified to not use all nodes for output, we can get rid of the non-readout node states
        reservoir_states = reservoir_states[:, :, self.readout_layer.readout_nodes]

        # make predictions y = R * W_out, W_out has a shape of [n_out, N]
        y_pred = np.einsum(
            "bik,jk->bij", reservoir_states, self.readout_layer.weights.T
        )

        return y_pred

    # @abstractmethod
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

        # remove some initial transients from the ground truth if discard transients is active
        # TODO: this should be done in the predict method

        # get metric values
        metric_values = []
        for _metric_fun in metric_funs:
            metric_values.append(float(_metric_fun(y, y_pred)))

        return metric_values

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
