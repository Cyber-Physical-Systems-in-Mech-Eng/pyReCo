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
from .optimizers import Optimizer, assign_optimizer
from .metrics import assign_metric
from .utils_networks import (
    gen_init_states,
    # set_spec_rad,
    # is_zero_col_and_row,
    # remove_node,
    get_num_nodes,
    # compute_spec_rad,
)
# from .metrics import assign_metric
from .network_prop_extractor import NetworkQuantifier  # NodePropExtractor


import numpy as np

def sample_random_nodes(total_nodes: int, fraction: float) -> np.ndarray:
    """
    Select a subset of randomly chosen nodes.

    Parameters:
    ----------
    total_nodes : int
        Total number of available nodes.
    fraction : float
        Fraction of nodes to select.

    Returns:
    -------
    np.ndarray
        Array of randomly selected node indices.
    """
    return np.random.choice(
        total_nodes, size=int(total_nodes * fraction), replace=False
    )


def discard_transients_indices(n_batches, n_timesteps, transients):
    """
    Discards the indices of transients in a sequence of batches.

    Args:
        n_batches (int): The number of batches.
        n_timesteps (int): The number of timesteps in each batch.
        transients (int): The number of transients to discard.

    Returns:
        list: A list of indices to be removed.

    """
    indices_to_remove = []
    for i in range(n_batches * n_timesteps):
        t = i % n_timesteps  # Current timestep within the batch
        if t < transients:
            indices_to_remove.append(i)
    return indices_to_remove


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

        Parameters
        ----------
        None

        Returns
        -------
        None
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

    def add(self, layer: Layer):
            """
            Add a layer to the model.

            Parameters
            ----------
            layer : Layer
                Layer to be added to the model.

            Notes
            -----
            This method is type-sensitive and will assign the layer to the correct attribute.
            If the layer is an instance of `InputLayer`, it will be assigned to the `input_layer` attribute.
            If the layer is a subclass of `ReservoirLayer`, it will be assigned to the `reservoir_layer` attribute.
            If the layer is an instance of `ReadoutLayer`, it will be assigned to the `readout_layer` attribute.
            """
            if isinstance(layer, InputLayer):
                self.input_layer = layer
            elif issubclass(type(layer), ReservoirLayer):
                self.reservoir_layer = layer
            elif isinstance(layer, ReadoutLayer):
                self.readout_layer = layer

    # TODO: the following method should be implemented in the CustomModel class
    #   def _set_readin_nodes(self, nodes: Union[list, np.ndarray] = None):

    """
    The setter methods are used to set the parameters of the model.
    """

    def _set_readout_nodes(self, nodes: Union[list, np.ndarray] = None):
        """
        Set the nodes that will be linked to the output.

        Parameters
        ----------
        nodes : Union[list, np.ndarray], optional
            Specific nodes to use for readout provided as indices. If None, randomly sample nodes.

        Notes
        -----
        If `nodes` is not provided, the function will randomly sample nodes from the reservoir layer.

        """
        if nodes is None:
            nodes = sample_random_nodes(
                total_nodes=self.reservoir_layer.nodes,
                fraction=self.readout_layer.fraction_out,
            )
        self.readout_layer.readout_nodes = nodes

    def _set_optimizer(self, optimizer: Union[str, Optimizer]):
        """
        Set the optimizer that will find the readout weights.

        Parameters
        ----------
        optimizer : Union[str, Optimizer]
            Name of the optimizer or an Optimizer instance.

        Returns
        -------
        None
        """
        self.optimizer = assign_optimizer(optimizer)

    def _set_metrics(self, metrics: Union[list, str]):
            """
            Set the metric(s) for model evaluation.

            Parameters:
            ----------
            metrics : Union[list, str]
                List of metric names or a single metric name.

            Returns:
            -------
            None

            Notes:
            ------
            This method sets the metric(s) to be used for model evaluation. The metrics can be provided as a list of metric names or a single metric name. The metric names should be valid and supported by the system.

            Examples:
            ---------
            >>> model._set_metrics(['accuracy', 'precision'])
            >>> model._set_metrics('f1_score')
            """
            if isinstance(metrics, str):  # only single metric given
                self.metrics = [metrics]
            else:
                self.metrics = metrics  # if metrics is a list of strings.

            # assign the metric functions (callable) according to the metric names
            self.metrics_fun = []  # has been initialized, we reset it here
            for metric in self.metrics:
                self.metrics_fun.append(assign_metric(metric))

    def _set_init_states(self, init_states=None, method=None):
        """
        Set the initial states of the reservoir nodes.

        Parameters:
        ----------
        init_states : np.ndarray, optional
            Array of initial states. If None, sample initial states using the specified method.
        method : str, optional
            Method for sampling initial states.

        Raises:
        ------
        ValueError
            If the provided initial states do not match the number of reservoir nodes.

        Notes:
        ------
        If `init_states` is not provided and `method` is specified, initial states are sampled using the specified method.

        If neither `init_states` nor `method` is provided, a ValueError is raised.

        """
        if init_states is not None:
            if init_states.shape[0] != self.reservoir_layer.nodes:
                raise (
                    ValueError(
                        "initial states not matching the number of reservoir nodes!"
                    )
                )
            self.reservoir_layer.set_initial_state(r_init=init_states)
        elif (init_states is None) and (method is not None):
            init_states = gen_init_states(self.reservoir_layer.nodes, method)
            self.reservoir_layer.set_initial_state(r_init=init_states)
        else:
            raise (
                ValueError(
                    "provide either an array of initial states or a method for sampling"
                )
            )

    def _connect_input_to_reservoir(self, nodes: Union[list, np.ndarray] = None):
        """
        Wire input layer with reservoir layer.

        Creates a random matrix of shape [nodes x n_states], i.e. number of reservoir nodes x state dimension of input.
        If no full connection is desired, a fraction of nodes will be selected according to the fraction_input parameter of the reservoir layer.

        Parameters:
        ----------
        nodes : Union[list, np.ndarray], optional
            Specific nodes to connect to the input. If None, randomly sample nodes.

        Returns:
        -------
        None
        """
        num_input_states = self.input_layer.n_states
        num_reservoir_nodes = self.reservoir_layer.nodes

        # generate random input connection matrix [nodes, n_states]
        full_input_weights = np.random.randn(num_input_states, num_reservoir_nodes)

        # select read-in node indices according to the fraction specified by the user
        if nodes is None:
            input_nodes = sample_random_nodes(
                total_nodes=num_reservoir_nodes,
                fraction=self.reservoir_layer.fraction_input,
            )
        else:
            input_nodes = nodes

        # mask the input weights matrix to only have the selected nodes
        mask = np.zeros_like(full_input_weights) 
        mask[:, input_nodes] = 1
        self.input_layer.weights = full_input_weights * mask

    def compile(
        self,
        optimizer: str = "ridge",
        metrics: list = ["mse"],
        discard_transients: int = 0,
    ):
        """
        Configure the model for training.

        Parameters:
        ----------
        optimizer : str, optional
            Name of the optimizer. Default is "ridge".
        metrics : list, optional
            List of metric names. Default is ["mse"].
        discard_transients : int, optional
            Number of initial transient timesteps to discard. Default is 0.
        """
        # set the metrics (like in TensorFlow)
        self._set_metrics(metrics)

        # set the optimizer that will find the readout weights
        self._set_optimizer(optimizer)

        # 1. check consistency of layers, data shapes etc.
        # TODO: do we have input, reservoir and readout layer?
        # TODO: are all shapes correct on input and output side?

        # 2. Sample the input connections: create W_in read-in weight matrix
        self._connect_input_to_reservoir()  # check for dependency injection here!

        # 3. Select readout nodes according to the fraction specified by the user in the readout layer
        self._set_readout_nodes()

        # 4. set reservoir initialization

        # 5. discarding transients from reservoir states
        if discard_transients < 0:
            raise (ValueError("discard_transients must be >= 0!"))
        self.discard_transients = int(
            discard_transients
        )  # will not remove transients if 0

    def compute_reservoir_state(self, X: np.ndarray, seed=None) -> np.ndarray:
        """
        Compute reservoir states with batch processing.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_batch, n_timesteps, n_states)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Reservoir states of shape (n_batch * n_timesteps, N)
        """
        # Extract shapes and parameters
        n_batch, n_time, n_states = X.shape
        N = self.reservoir_layer.nodes
        activation_name = self.reservoir_layer.activation
        alpha = self.reservoir_layer.leakage_rate
        A = self.reservoir_layer.weights
        W_in = self.input_layer.weights

        # Pre-allocate arrays
        states = np.zeros((n_batch, n_time + 1, N))
        states[:, 0] = self.reservoir_layer.initial_res_states

        # vectorized computation of reservoir states
        input_contrib = np.einsum("ij,btj->bti", W_in.T, X)
        for t in range(n_time):
            reservoir_contrib = np.einsum("ij,bj->bi", A, states[:, t])
            states[:, t + 1] = (1 - alpha) * states[:, t] + alpha * self.reservoir_layer.activation_fun(
                reservoir_contrib + input_contrib[:, t]
            )

        return states[:, 1:].reshape(-1, N)

    def fit(
        self, X: np.ndarray, y: np.ndarray, n_init: int = 1, store_states: bool = False
    ):
        """
        Optimized training with batch processing.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_batch, n_time, n_features), where n_batch is the number of samples, 
            n_time is the time dimension, and n_features is the number of features in the input data.
        y : np.ndarray
            Target data of shape (n_batch, n_time, n_states_out), where n_batch is the number of samples, 
            n_time is the number of time steps, and n_states_out is the number of output states.
        n_init : int, optional
            The number of initializations to perform. Default is 1.
        store_states : bool, optional
            Whether to store the reservoir states during training. Default is False.

        Returns
        -------
        dict
            A dictionary containing the training history with the following keys:
                - 'init_res_states' : np.ndarray
                    The initial reservoir states for each initialization of shape (n_init, n_nodes).
                - 'readout_weights' : np.ndarray
                    The readout weights for each initialization of shape (n_init, n_nodes, n_states_out).
                - 'train_scores' : np.ndarray
                    The scores of the model for each initialization of shape (n_init,).
                - 'res_states' : list of np.ndarray, optional
                    The reservoir states for each batch, returned only if `store_states` is True.

        Notes
        -----
        The function uses batch processing for training, performs multiple initializations, 
        computes reservoir states, and solves a regression problem to update the readout layer weights.
        """
        n_batch, n_time, n_states_out = X.shape[0], X.shape[1], y.shape[-1]
        n_nodes = self.reservoir_layer.nodes

        # Pre-allocate arrays for storing results
        n_R0 = np.zeros((n_init, n_nodes))
        n_weights = np.zeros((n_init, n_nodes, n_states_out))
        n_scores = np.zeros(n_init)
        n_res_states = [] if store_states else None

        # Get metric functions that scores the model performance
        metric_fun = (
            self.metrics_fun[0]
            if self.metrics_fun
            else assign_metric("mean_squared_error")
        )

        # Batch process multiple initializations
        for i in range(n_init):
            if n_init > 1:
                print(f"initialization {i}/{n_init}: computing reservoir states")

            # Set initial states
            self._set_init_states(method=self.reservoir_layer.init_res_sampling)
            n_R0[i] = self.reservoir_layer.initial_res_states

            # Compute reservoir states
            reservoir_states = self.compute_reservoir_state(X)

            # Handle transients efficiently
            if self.discard_transients > 0:
                indices_to_remove = discard_transients_indices(
                    n_batch, n_time, self.discard_transients
                )  # by juan
                reservoir_states = np.delete(
                    reservoir_states, indices_to_remove, axis=0
                )
                # now the array should have the size of (n_batch*(n_time-discard), n_nodes)

                # remove the transients from the targets
                y = y[:, self.discard_transients :, :]

                # update the value of n_time
                n_time -= self.discard_transients

            # Efficient masking for readout nodes
            A = np.zeros_like(reservoir_states)
            A[:, self.readout_layer.readout_nodes] = reservoir_states[
                :, self.readout_layer.readout_nodes
            ]

            # Reshape targets efficiently
            b = y.reshape(n_batch * n_time, n_states_out)

            # Solve regression problem y = W_out * R
            if n_batch == 1:
                self.readout_layer.weights = np.expand_dims(
                    self.optimizer.solve(A=A, b=b), axis=-1
                )
            else:
                self.readout_layer.weights = self.optimizer.solve(A=A, b=b)

            # is there is only a single system state to predict, we need to add that dim
            # TODO: move this to the sanity checks and add an artificial dimension prior to fitting!
            if self.readout_layer.weights.ndim == 1:
                self.readout_layer.weights = np.expand_dims(self.readout_layer.weights, axis=-1)

            # store weights for this initialization
            n_weights[i] = self.readout_layer.weights

            # Compute score
            n_scores[i] = metric_fun(y, self.predict(X=X))

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

    def fit_evolve(self, X: np.ndarray, y: np.ndarray):
        """
        Build an evolving reservoir computer: performance-dependent node addition and removal.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        y : np.ndarray
            Target data of shape (n_samples,).

        Returns
        -------
        unknown
            The training history. The exact type is not known.
        """
        history = None
        return history

    def evaluate_node_removal(
        self, X, y, loss_fun, init_score, del_idx, current_num_nodes
    ):
        """
        Evaluates the performance of the model after removing a node.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features), where n_samples is the number of samples 
            and n_features is the number of input features.
        y : np.ndarray
            Target data of shape (n_samples, n_time, n_states_out), where n_samples is the number of 
            samples, n_time is the number of time steps, and n_states_out is the number of output states.
        loss_fun : callable
            A loss function used to evaluate the model performance. It should take the true target values 
            and predicted values as inputs and return a scalar loss value.
        init_score : float
            The initial score (loss) of the model before node removal, used for comparison after pruning.
        del_idx : int
            The index of the node to be removed from the model.
        current_num_nodes : int
            The current number of nodes in the model before the node removal.

        Returns
        -------
        float
            The score (loss) of the model after removing the node, as computed by the given loss function.

        Notes
        -----
        This method creates a temporary deep copy of the model, removes a node, retrains the model, 
        and evaluates the performance. The score after removal is compared with the original score to 
        assess the impact of the node removal.
        """
        # Create a deep copy of the current model
        temp_model = copy.deepcopy(self)

        # Remove node from the temporary model
        temp_model.remove_node(del_idx)

        # Train the temporary model
        temp_model.fit(X, y)
        y_discarded = y[:, self.discard_transients :, :]
        # Evaluate the temporary model
        temp_score = loss_fun(y_discarded, temp_model.predict(X=X))

        print(
            f"Pruning node {del_idx} / {current_num_nodes}: loss = {temp_score:.5f}, original loss = {init_score:.5f}"
        )

        return temp_score

    def fit_prune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_metric="mse",
        max_perf_drop=0.1,
        frac_rem_nodes=0.20,
        patience=None,
        prop_extractor=None,
    ):
        """
        Build a reservoir computer by performance-informed pruning of the initial reservoir network.

        This method prunes the network down to better performance OR a tolerated performance reduction.

        Args:
            X (np.ndarray): Input data of shape [n_batch, n_time_in, n_states_in]
            y (np.ndarray): Target data of shape [n_batch, n_time_out, n_states_out]
            loss_metric (str): Metric for performance-informed node removal. Must be a member of existing metrics in pyReCo.
            max_perf_drop (float): Maximum allowed performance drop before stopping pruning. Default: 0.1 (10%)
            frac_rem_nodes (float): Fraction of nodes to attempt to remove in each iteration. Default: 0.01 (1%)
            patience (int): Number of consecutive performance decreases allowed before early stopping
            prop_extractor (object): Object to extract network properties during pruning

        Returns:
            dict: History of the pruning process
        """

        # Ensure frac_rem_nodes is within the valid range [0, 1]
        frac_rem_nodes = max(0.0, min(1.0, frac_rem_nodes))

        # Get a callable loss function for performance-informed node removal
        loss_fun = assign_metric(loss_metric)

        # Initialize reservoir states
        self._set_init_states(method=self.reservoir_layer.init_res_sampling)

        # Get size of original reservoir
        num_nodes = self.reservoir_layer.weights.shape[0]

        # Set default patience if not specified
        if patience is None:
            patience = num_nodes

        # Compute initial score of full network on training set
        self.fit(X, y)
        y_discarded = y[:, self.discard_transients :, :]
        init_score = loss_fun(y_discarded, self.predict(X=X))

        def keep_pruning(init_score, current_score, max_perf_drop):
            """
            Determine if pruning should continue based on the current model's performance.

            Parameters
            ----------
            init_score : float
                The initial score (loss) of the model before any nodes were removed.
            current_score : float
                The current score (loss) of the model after removing some nodes.
            max_perf_drop : float
                The maximum acceptable performance drop (as a fraction of the initial score). 
                If the current score exceeds the initial score by more than this fraction, pruning stops.

            Returns
            -------
            bool
                True if pruning should continue, False otherwise.

            Notes
            -----
            This function compares the current score against the initial score and the maximum allowed 
            performance drop. If the performance has dropped too much, pruning will stop.
            """
            if current_score < (init_score * (1.0 + max_perf_drop)):
                return True
            else:
                print("Pruning stopping criterion reached.")
                return False

        # Initialize property extractor if not provided. TODO needs to be improved
        if prop_extractor is None:
            prop_extractor = NetworkQuantifier()

        # Initialize pruning variables
        i = 0
        current_score = init_score
        current_num_nodes = get_num_nodes(self.reservoir_layer.weights)
        score_per_node = []
        history = {
            "pruned_nodes": [-1],
            "pruned_nodes_scores": [init_score],
            "num_nodes": [current_num_nodes],
            "network_properties": [],
        }

        # Extract initial network properties
        initial_props = prop_extractor.extract_properties(self.reservoir_layer.weights)
        history["network_properties"].append(initial_props)

        consecutive_increases = 0
        best_score = init_score

        best_model = copy.deepcopy(self)

        # Main pruning loop
        while i < num_nodes:
            print(f"Pruning iteration {i}")

            # Calculate number of nodes to try removing this iteration
            num_nodes_to_try = max(1, int(current_num_nodes * frac_rem_nodes))

            score_per_node.append([])
            max_loss = init_score

            # Prepare the partial function for multiprocessing
            evaluate_func = partial(
                self.evaluate_node_removal,
                X,
                y,
                loss_fun,
                init_score,
                current_num_nodes=current_num_nodes,
            )

            # Use multiprocessing to evaluate node removals in parallel
            with multiprocessing.Pool() as pool:
                results = pool.map(evaluate_func, range(current_num_nodes))

            # Process the results
            score_per_node[i] = results
            max_loss = max(max_loss, max(results))

            # Find nodes which affect the loss the least
            max_loss = max_loss + 1
            score_per_node[i] = [
                max_loss if x is None else x for x in score_per_node[i]
            ]
            sorted_indices = np.argsort(score_per_node[i])
            nodes_to_remove = sorted_indices[:num_nodes_to_try]

            if keep_pruning(init_score, current_score, max_perf_drop):
                # Remove node from all layers
                self.remove_node(nodes_to_remove)

                # Retrain and evaluate
                self.fit(X, y)
                y_discarded = y[:, self.discard_transients :, :]
                current_score = loss_fun(y_discarded, self.predict(X=X))
                rel_score = (current_score - init_score) / init_score * 100

                current_num_nodes = self.reservoir_layer.nodes

                print(
                    f"Removing node {nodes_to_remove}: new loss = {current_score:.5f}, original loss = {init_score:.5f} ({rel_score:+.2f} %); {current_num_nodes} nodes remain"
                )

                # Check for early stopping and update best model
                if current_score < best_score:
                    best_score = current_score
                    best_model = copy.deepcopy(self)
                    consecutive_increases = 0
                else:
                    consecutive_increases += 1
                    if consecutive_increases >= patience:
                        print(
                            f"Stopping pruning: Loss increased for {patience} consecutive iterations."
                        )
                        break

                # Extract and store network properties
                network_props = prop_extractor.extract_properties(
                    self.reservoir_layer.weights
                )
                history["network_properties"].append(network_props)

                # Update pruning history
                history["pruned_nodes"].append(nodes_to_remove.tolist())
                history["pruned_nodes_scores"].append(
                    [score_per_node[i][node] for node in nodes_to_remove]
                )
                history["num_nodes"].append(current_num_nodes)
            else:
                break

            i += 1

        # Add best score to history
        history["best_score"] = best_score

        return history, best_model

    # @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for given input (single-step prediction).

        Parameters
        ----------
        X : np.ndarray
            Input data of shape [n_batch, n_timestep, n_states].

        Returns
        -------
        np.ndarray
            Predictions of shape [n_batch, n_timestep, n_states]
        """

        # makes prediction for given input (single-step prediction)
        # expects inputs of shape [n_batch, n_timestep, n_states]
        # returns predictions in shape of [n_batch, n_timestep, n_states]

        # one_shot = True will *not* re-initialize the reservoir from sample to sample. Introduces a dependency on the
        # sequence by which the samples are given

        # TODO Merten: return some random number that have the correct shape

        # TODO: external function that is going to check the dimensionality
        # and raise an error if shape is not correct
        n_batch, n_time, n_states = X.shape[0], X.shape[1], X.shape[2]
        n_nodes = self.reservoir_layer.nodes

        # iterate over batch to obtain predictions
        reservoir_states = self.compute_reservoir_state(X)

        # Removing transients AKA Warm-up and update time
        # TODO: this is a lot of boilerplate code. @Juan reuse the function from .fit
        if self.discard_transients >= n_time:
            raise ValueError(
                f"Cannot discard {self.discard_transients} as the number of time steps is {n_time}"
            )
        if self.discard_transients > 0:

            # removes the first <discard_transients> from the reservoir states and from the targets
            # reservoir_states.shape is 2d, as we concatenated along the batch dimension: [n_time * n_batch, n_nodes]
            # hence we have to remove slices from the state matrix, or re-shape it into 3D, cut off some time steps
            # for each batch, and then reshape to 2D again.
            # TODO: please check if the reshaping really is correct, i.e. such that the first n_time entries of reservoir_states are the continuous reservoir states!
            indices_to_remove = discard_transients_indices(
                n_batch, n_time, self.discard_transients
            )
            reservoir_states = np.delete(reservoir_states, indices_to_remove, axis=0)
            # now the array should have the size of (n_batch*(n_time-discard), n_nodes)

            # update the value of n_time
            n_time = -self.discard_transients

            # reservoir_states, X, y = TransientRemover('RXY', reservoir_states, X, y, self.discard_transients)

        # make predictions y = R * W_out, W_out has a shape of [n_out, N]
        y_pred = np.dot(reservoir_states, self.readout_layer.weights)

        # reshape predictions into 3D [n_batch, n_time_out, n_state_out]
        n_time_out = int(y_pred.shape[0] / n_batch)
        n_states_out = y_pred.shape[-1]
        y_pred = y_pred.reshape(n_batch, n_time_out, n_states_out)

        return y_pred

    # @abstractmethod
    def evaluate(
        self, X: np.ndarray, y: np.ndarray, metrics: Union[str, list, None] = None
    ) -> tuple:
        """
        Evaluate metrics on predictions made for input data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape [n_batch, n_timesteps, n_states].

        y : np.ndarray
            Target data of shape [n_batch, n_timesteps_out, n_states_out].

        metrics : Union[str, list of str, None], default=None
            List of metric names or a single metric name. If None, uses metrics from `.compile()`.

        Returns
        -------
        tuple
            A tuple containing the metric values.
        """
        # evaluate metrics on predictions made for input data
        # expects: X of shape [n_batch, n_timesteps, n_states]
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
        y_pred = self.predict(X)

        # remove some initial transients from the ground truth if discard transients is active
        n_time = y.shape[1]
        if self.discard_transients >= n_time:
            raise ValueError(
                f"Cannot discard {self.discard_transients} as the number of time steps is {n_time}"
            )

        if self.discard_transients > 0:
            y = y[:, self.discard_transients :, :]

        # get metric values
        metric_values = []
        for _metric_fun in metric_funs:
            metric_values.append(float(_metric_fun(y, y_pred)))

        return metric_values

    # @abstractmethod
    def get_params(self, deep=True):
        """
        Get parameters for scikit-learn compatibility.

        Parameters
        ----------
        deep : bool, default=True
            If True, return a deep copy of parameters.

        Returns
        -------
        dict
            Dictionary of model parameters, with layer names as keys and their corresponding values.
        """
        # needed for scikit-learn compatibility
        return {
            "input_layer": self.input_layer,
            "reservoir_layer": self.reservoir_layer,
            "readout_layer": self.readout_layer,
        }

    # @abstractmethod
    def save(self, path: str):
        """
        Store the model to disk.

        Parameters
        ----------
        path : str
            Path to save the model.

        Returns
        -------
        None
        """
        # store the model to disk
        pass

    def plot(self, path: str):
        """
        Print the model to some figure file.

        Parameters
        ----------
        path : str
            Path to save the figure.

        Returns
        -------
        None
        """
        # print the model to some figure file
        pass

    def remove_node(self, node_indices):
        """
        Remove one or multiple nodes from all relevant layers of the reservoir computer.

        Parameters
        ----------
        node_indices : int or list or np.ndarray
            Index or indices of the nodes to be removed.
        Returns
        -------
        None
        """
        # Convert single integer to list
        if isinstance(node_indices, int):
            node_indices = [node_indices]

        # Remove nodes from reservoir layer weights
        self.reservoir_layer.weights = np.delete(
            self.reservoir_layer.weights, node_indices, axis=0
        )
        self.reservoir_layer.weights = np.delete(
            self.reservoir_layer.weights, node_indices, axis=1
        )

        # Remove nodes from initial reservoir states
        self.reservoir_layer.initial_res_states = np.delete(
            self.reservoir_layer.initial_res_states, node_indices, axis=0
        )

        # Remove nodes from input layer weights
        self.input_layer.weights = np.delete(
            self.input_layer.weights, node_indices, axis=1
        )

        # Remove nodes from readout layer weights
        self.readout_layer.weights = np.delete(
            self.readout_layer.weights, node_indices, axis=0
        )

        # Update readout nodes
        mask = np.ones(len(self.readout_layer.readout_nodes), dtype=bool)
        for idx in node_indices:
            mask[self.readout_layer.readout_nodes == idx] = False
        self.readout_layer.readout_nodes = self.readout_layer.readout_nodes[mask]

        # Adjust the indices of the remaining readout nodes
        for idx in sorted(node_indices, reverse=True):
            self.readout_layer.readout_nodes[
                self.readout_layer.readout_nodes > idx
            ] -= 1

        # Update node count
        self.reservoir_layer.nodes -= len(node_indices)


class RC(CustomModel):  # the non-auto version
    """
    Non-autonomous version of the reservoir computer.
    """

    def __init__(self):
        """
        Initialize the object.

        At the moment, no arguments are passed to the constructor. This method calls 
        the constructor of the parent class using `super()`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # at the moment we do not have any arguments to pass
        super().__init__()


class AutoRC(CustomModel):
    """
    Autonomous version of the reservoir computer.
    """

    def __init__(self):
        """
        Initialize the object.

        This method is the constructor for the class. It currently does not initialize
        any attributes or perform any operations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def predict_ar(self, X: np.ndarray, n_steps: int = 10):
        """
        Perform auto-regressive prediction for time series forecasting.

        Parameters
        ----------
        X : np.ndarray
            Initial input data.
        n_steps : int
            Number of steps to predict into the future.

        Returns
        -------
        np.ndarray
            Predicted future states.
        """
        pass


class HybridRC(CustomModel):
    """
    Hybrid version of the reservoir computer.
    """

    def __init__(self):
        """
        Initialize the `HybridRC` model. Is passed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass


if __name__ == "__main__":
    print("hello")
