import numpy as np
import random
from abc import ABC, abstractmethod
import os
import sys
from typing import Union
import copy
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import multiprocessing
from functools import partial

from .remove_transients import TransientRemover
from .layers import Layer, InputLayer, ReservoirLayer, ReadoutLayer, RandomReservoirLayer
from .optimizers import Optimizer, assign_optimizer
from .metrics import assign_metric
from .utils_networks import gen_init_states, set_spec_rad, is_zero_col_and_row, remove_node, get_num_nodes, \
    compute_spec_rad
from .metrics import assign_metric, available_metrics
from .network_prop_extractor import NetworkQuantifier, NodePropExtractor


def sample_random_nodes(total_nodes: int, fraction: float):
    """
    Select a subset of randomly chosen nodes.

    Args:
    total_nodes (int): Total number of nodes.
    fraction (float): Fraction of nodes to select.

    Returns:
    np.ndarray: Array of randomly selected node indices.
    """
    return np.random.choice(total_nodes, size=int(total_nodes * fraction), replace=False)


def discard_transients_indices(n_batches, n_timesteps, transients):
    """
    Generate indices of transient timesteps to be discarded across multiple batches.

    Args:
    n_batches (int): Number of batches.
    n_timesteps (int): Number of timesteps in each batch.
    transients (int): Number of initial timesteps to be considered as transients.

    Returns:
    list: A list of indices to be removed, corresponding to transient timesteps.
    """
    return [i for i in range(n_batches * n_timesteps) if i % n_timesteps < transients]


from matplotlib import pyplot as plt


# the Model class will be the super-class (abstract base class, ABC) for non-autonomous, hybrid, and autonomous RC
# models

# #### <model> class methods
# - model.compile(optimizer, metrics, ...)
# - model.fit(X, y)
# - model.predict(X)
# - model.predict_ar()
# - model.evaluate(X, y)
# - model.save(path)


from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from abc import ABC
import numpy as np
import copy
import multiprocessing
from functools import partial
from typing import Union

class CustomModel(ABC):
    """
    Abstract base class for custom reservoir computing models.
    """

    def __init__(self):
        """
        Initialize the CustomModel with empty layers and default values.
        """
        # Initialize layers
        self.input_layer: InputLayer
        self.reservoir_layer: ReservoirLayer
        self.readout_layer: ReadoutLayer

        self.metrics = []
        self.optimizer: Optimizer
        self.discard_transients = 0
        self.trainable_weights: int

    def add(self, layer: Layer):
        """
        Add a layer to the model.

        Args:
        layer (Layer): Layer to be added to the model.
        """
        if type(layer) == InputLayer:
            self.input_layer = layer
        elif issubclass(type(layer), ReservoirLayer):
            self.reservoir_layer = layer
        elif type(layer) == ReadoutLayer:
            self.readout_layer = layer

    def _set_readout_nodes(self, nodes: Union[list, np.ndarray] = None):
        """
        Set the nodes that will be linked to the output.

        Args:
        nodes (Union[list, np.ndarray], optional): Specific nodes to use for readout. If None, randomly sample nodes.
        """
        if nodes is None:
            nodes = sample_random_nodes(total_nodes=self.reservoir_layer.nodes,
                                        fraction=self.readout_layer.fraction_out)
        self.readout_layer.readout_nodes = nodes

    def _connect_input_to_reservoir(self, nodes: Union[list, np.ndarray] = None):
        """
        Wire input layer with reservoir layer. Creates a random matrix of shape nodes x n_states, i.e. number of
        reservoir nodes x state dimension of input.

        Args:
        nodes (Union[list, np.ndarray], optional): Specific nodes to connect to the input. If None, randomly sample nodes.
        """
        input_size = self.input_layer.n_states
        n = self.reservoir_layer.nodes

        # generate random input connection matrix [nodes, n_states]
        self.input_layer.weights = np.random.randn(input_size, n)

        # select the read-in connections
        if (nodes is None) and (self.reservoir_layer.fraction_input < 1.0):
            input_nodes = sample_random_nodes(total_nodes=n, fraction=self.reservoir_layer.fraction_input)
        else:
            input_nodes = nodes

        mask = np.zeros_like(self.input_layer.weights)
        mask[:, input_nodes] = 1
        self.input_layer.weights = self.input_layer.weights * mask

    def _set_optimizer(self, optimizer: str):
        """
        Set the optimizer that will find the readout weights.

        Args:
        optimizer (str): Name of the optimizer.
        """
        self.optimizer = assign_optimizer(optimizer)

    def _set_metrics(self, metrics: Union[list, str]):
        """
        Set the metrics for evaluation.

        Args:
        metrics (Union[list, str]): List of metric names or a single metric name.
        """
        if type(metrics) == str:  # only single metric given
            self.metrics = [metrics]
        else:
            self.metrics = metrics  #by juan. metrics is a list of strings.
        self.metrics_fun = []  # the actual callable metric functions
        for metric in self.metrics:
            self.metrics_fun.append(assign_metric(metric))

    def _set_init_states(self, init_states=None, method=None):
        """
        Set the initial states of the reservoir layer.

        Args:
        init_states (np.ndarray, optional): Array of initial states. If None, sample initial states using the specified method.
        method (str, optional): Method for sampling initial states.
        """
        if init_states is not None:
            if init_states.shape[0] != self.reservoir_layer.nodes:
                raise (ValueError('initial states not matching the number of reservoir nodes!'))
            self.reservoir_layer.set_initial_state(r_init=init_states)
        elif (init_states is None) and (method is not None):
            init_states = gen_init_states(self.reservoir_layer.nodes, method)
            self.reservoir_layer.set_initial_state(r_init=init_states)
        else:
            raise (ValueError('provide either an array of initial states or a method for sampling those'))

    def compile(self, optimizer: str = 'ridge', metrics: list = ['mse'], discard_transients: int = 0):
        """
        Configure the model for training.

        Args:
        optimizer (str): Name of the optimizer.
        metrics (list): List of metric names.
        discard_transients (int): Number of initial transient timesteps to discard.
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
            raise (ValueError('discard_transients must be >= 0!'))
        self.discard_transients = int(discard_transients)  # will not remove transients if 0

    def compute_reservoir_state(self, X: np.ndarray, seed=None) -> np.ndarray:
        """
        Compute reservoir states for the given input data.

        Args:
        X (np.ndarray): Input data of shape [n_batch, n_timesteps, n_states]
        seed (int, optional): Random seed for reproducibility.

        Returns:
        np.ndarray: Reservoir states of shape [(n_batch * n_timesteps), N]
        """
        # expects an input of shape [n_batch, n_timesteps, n_states]
        # returns the reservoir states of shape [(n_batch * n_timesteps), N]

        # (except for .predict)! Don't let the prediction on a sample depend on which
        # sample you computed before.

        # get data shapes
        n_samples = X.shape[0]
        n_time = X.shape[1]
        n_states = X.shape[2]

        # extract layer properties for easier syntax below:
        N = self.reservoir_layer.nodes
        g = self.reservoir_layer.activation_fun
        alpha = self.reservoir_layer.leakage_rate
        A = self.reservoir_layer.weights
        W_in = self.input_layer.weights
        R0 = self.reservoir_layer.initial_res_states

        # now loop over all samples, compute reservoir states for each sample, and re-initialize reservoir (in case
        # one_shot parameter = False

        R_all = []  # collects reservoir states [n_sample, [n_nodes, n_time]]
        for sample in range(n_samples):  # loop over training samples, the batch
            # print(f'{sample}/{n_samples} ...')

            R = np.zeros([N, n_time + 1])
            # if one_shot and sample>0:   # re-use last reservoir state from previous sample
            #     R0 = R_all[-1][:,-1]

            R[:, 0] = R0

            for t, x in enumerate(X[sample, :, :]):  # go through time steps (X[1]) for current training sample
                R[:, t + 1] = (1 - alpha) * R[:, t] + alpha * g(np.dot(A, R[:, t].T) + np.dot(W_in.T, x))

            R_all.append(R[:, 1:])  # get rid of initial reservoir state

        # concatenate all reservoir states
        R_all = np.hstack(R_all).T

        return R_all  # all reservoir states: [n_nodes, (n_batch * n_time)]

    def fit(self, X: np.ndarray, y: np.ndarray, one_shot: bool = False, n_init: int = 1, store_states: bool = False):
        """
        Train the reservoir computer on the given data.

        Args:
        X (np.ndarray): Input data of shape [n_batch, n_time_in, n_states_in]
        y (np.ndarray): Target data of shape [n_batch, n_time_out, n_states_out]
        one_shot (bool): If True, don't re-initialize reservoir between samples.
        n_init (int): Number of times to sample initial reservoir states.
        store_states (bool): If True, store full time trace of reservoir states.

        Returns:
        dict: History of the training process.
        """
        # expects data in particular format that is reasonable for univariate/multivariate time series data
        # - X input data of shape [n_batch, n_time_in, n_states_in]
        # - y target data of shape [n_batch, n_time_out, n_states_out]
        # - n_init: number of times that initial reservoir states are sampled.
        # - store_states returns the full time trace of reservoir states (memory-heavy!)

        n_batch = X.shape[0]
        n_time = X.shape[1]
        n_states_out = y.shape[-1]
        n_nodes = self.reservoir_layer.nodes

        # one_shot = True will *not* re-initialize the reservoir from sample to sample. Introduces a dependency on the
        # sequence by which the samples are given

        # train the RC to the given data. Will also need to check for consistency of everything
        # returns some values that describe how well the training went

        # loop across many different initial conditions for the reservoir states R(t=0) --> n_init
        # TODO: parallelize the loop over multiple initial reservoir states.
        n_R0, n_weights, n_scores, n_res_states = [], [], [], []
        for i in range(n_init):
            print(f'initialization {i}/{n_init}: computing reservoir states')

            # set the initial reservoir state (should involve some randomness if n_init > 1)
            # TODO: call a pre-defined initializer according to some input (0, random normal, uniform)
            self._set_init_states(method=self.reservoir_layer.init_res_sampling)

            # feed training data through RC and obtain reservoir states
            reservoir_states = self.compute_reservoir_state(X)  # complete n states

            # Removing transients AKA Warm-up and update time
            if self.discard_transients >= n_time:
                raise ValueError(f'Cannot discard {self.discard_transients} as the number of time steps is {n_time}')
            if self.discard_transients > 0:
                print(f'discarding first {self.discard_transients} transients during training')

                # removes the first <discard_transients> from the reservoir states and from the targets

                # WORKAROUND
                # reservoir_states.shape is 2d, as we concatenated along the batch dimension: [n_time * n_batch, n_nodes]
                # hence we have to remove slices from the state matrix, or re-shape it into 3D, cut off some time steps
                # for each batch, and then reshape to 2D again.
                indices_to_remove = discard_transients_indices(n_batch, n_time, self.discard_transients)  #by juan
                reservoir_states = np.delete(reservoir_states, indices_to_remove, axis=0)
                print("reservoir_states.shape: ", reservoir_states.shape)
                # now the array should have the size of (n_batch*(n_time-discard), n_nodes)

                # remove the transients from the targets
                y = y[:, self.discard_transients:, :]

                # update the value of n_time
                n_time -= self.discard_transients

                # reservoir_states, X, y = TransientRemover('RXY', reservoir_states, X, y, self.discard_transients)

            # set up the linear regression problem Ax=b, A=R, b=y, x=W_out
            # mask reservoir states that are not selected as output nodes
            # TODO: make this more efficient, do not create full mask array

            #Initialize A as a zero array with the same shape as reservoir_states
            # Assign values only to columns corresponding to readout nodes
            # by juan
            A = np.zeros_like(reservoir_states)
            A[:, self.readout_layer.readout_nodes] = reservoir_states[:, self.readout_layer.readout_nodes]

            # reshape targets y [n_batch, n_time, n_out] to [(n_batch * n_time), n_out],
            # i.e. stack all targets into long vector along time dimension
            b = y.reshape(n_batch * n_time, n_states_out)

            # 2. solve regression problem and update readout matrix
            # R * W_out = y --> [n_nodes, (n_batch * n_time)].T * W_out = [(n_batch * n_time), n_out]
            self.readout_layer.weights = self.optimizer.solve(A=A, b=b)

            # 3. compute score on training set (required to find the best initial reservoir state)
            # Calculates the loss with the first metric chosen by the user
            if self.metrics_fun:
                loss_fun = self.metrics_fun[0]
            else:
                # Default to mean squared error if no metrics specified
                loss_fun = assign_metric('mean_squared_error')

            score = loss_fun(y, self.predict(X=X))

            # store intermediate results for the n_init loop
            n_R0.append(self.reservoir_layer.initial_res_states)
            n_weights.append(self.readout_layer.weights)
            n_scores.append(score)

            if store_states:
                n_res_states.append(reservoir_states)

        # select the best prediction, i.e. the best initial condition
        if n_init > 1:
            idx_optimal = np.argmin(n_scores)
        else:
            idx_optimal = 0
        self.reservoir_layer.set_initial_state(n_R0[idx_optimal])
        self.readout_layer.weights = n_weights[idx_optimal]

        # specify the number of trainable weights by the size of the readout matrix
        self.trainable_weights = self.reservoir_layer.weights.size

        # built a history object to store detailed information about the training process
        history = dict()
        history['init_res_states'] = n_R0
        history['readout_weights'] = n_weights
        history['train_scores'] = n_scores

        if store_states:
            history['res_states'] = n_res_states
        # uncertain if we should store the reservoir states by default, will be a large memory consumption

        return history

    def fit_evolve(self, X: np.ndarray, y: np.ndarray):
        # build an evolving reservoir computer: performance-dependent node addition and removal

        history = None
        return history

    def evaluate_node_removal(self, X, y, loss_fun, init_score, del_idx, current_num_nodes):
        # Create a deep copy of the current model
        temp_model = copy.deepcopy(self)
        
        # Remove node from the temporary model
        temp_model.remove_node(del_idx)

        # Train the temporary model
        temp_model.fit(X, y)
        
        # Evaluate the temporary model
        temp_score = loss_fun(y, temp_model.predict(X=X))

        print(f'Pruning node {del_idx} / {current_num_nodes}: loss = {temp_score:.5f}, original loss = {init_score:.5f}')

        return temp_score

    def fit_prune(self, X: np.ndarray, y: np.ndarray, loss_metric='mse', max_perf_drop=0.1, frac_rem_nodes=0.01,
                  patience=None, prop_extractor=None):
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
        init_score = loss_fun(y, self.predict(X=X))

        def keep_pruning(init_score, current_score, max_perf_drop):
            """
                Determine if pruning should continue based on current performance.
                """
            if current_score < (init_score * (1.0 + max_perf_drop)):
                return True
            else:
                print('Pruning stopping criterion reached.')
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
            'pruned_nodes': [-1],
            'pruned_nodes_scores': [init_score],
            'num_nodes': [current_num_nodes],
            'network_properties': []
        }

        # Extract initial network properties
        initial_props = prop_extractor.extract_properties(self.reservoir_layer.weights)
        history['network_properties'].append(initial_props)

        consecutive_increases = 0
        best_score = init_score

        # Main pruning loop
        while keep_pruning(init_score, current_score, max_perf_drop) and (i < num_nodes):
            print(f'Pruning iteration {i}')

            # Calculate number of nodes to try removing this iteration
            num_nodes_to_try = max(1, int(current_num_nodes * frac_rem_nodes))

            score_per_node.append([])
            max_loss = init_score

            # Prepare the partial function for multiprocessing
            evaluate_func = partial(self.evaluate_node_removal, X, y, loss_fun, init_score, current_num_nodes=current_num_nodes)

            # Use multiprocessing to evaluate node removals in parallel
            with multiprocessing.Pool() as pool:
                results = pool.map(evaluate_func, range(current_num_nodes))

            # Process the results
            score_per_node[i] = results
            max_loss = max(max_loss, max(results))

            # Find nodes which affect the loss the least
            max_loss = max_loss + 1
            score_per_node[i] = [max_loss if x is None else x for x in score_per_node[i]]
            sorted_indices = np.argsort(score_per_node[i])
            nodes_to_remove = sorted_indices[:num_nodes_to_try]

            # Permanently remove selected nodes
            for idx_del_node in nodes_to_remove:
                if keep_pruning(init_score, current_score, max_perf_drop):
                    # Remove node from all layers
                    self.remove_node(idx_del_node)

                    # Retrain and evaluate
                    self.fit(X, y)
                    current_score = loss_fun(y, self.predict(X=X))
                    rel_score = (current_score - init_score) / init_score * 100

                    current_num_nodes = self.reservoir_layer.nodes

                    print(
                        f'Removing node {idx_del_node}: new loss = {current_score:.5f}, original loss = {init_score:.5f} ({rel_score:+.2f} %); {current_num_nodes} nodes remain')

                    # Check for early stopping
                    if current_score > best_score:
                        consecutive_increases += 1
                        if consecutive_increases >= patience:
                            print(f'Stopping pruning: Loss increased for {patience} consecutive iterations.')
                            return history
                    else:
                        consecutive_increases = 0
                        best_score = current_score

                    # Extract and store network properties
                    network_props = prop_extractor.extract_properties(self.reservoir_layer.weights)
                    history['network_properties'].append(network_props)

                    # Update pruning history
                    history['pruned_nodes'].append(idx_del_node)
                    history['pruned_nodes_scores'].append(score_per_node[i][idx_del_node])
                    history['num_nodes'].append(current_num_nodes)
                else:
                    break

            i += 1

        return history

    # @abstractmethod
    def predict(self, X: np.ndarray, one_shot: bool = False) -> np.ndarray:
        """
        Make predictions for given input (single-step prediction).

        Args:
        X (np.ndarray): Input data of shape [n_batch, n_timestep, n_states]
        one_shot (bool): If True, don't re-initialize reservoir between samples.

        Returns:
        np.ndarray: Predictions of shape [n_batch, n_timestep, n_states]
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
            raise ValueError(f'Cannot discard {self.discard_transients} as the number of time steps is {n_time}')
        if self.discard_transients > 0:
            print(f'discarding transients from the reservoir states of shape {reservoir_states.shape}')

            print(f'shape of inputs X: {X.shape}')

            # removes the first <discard_transients> from the reservoir states and from the targets
            # reservoir_states.shape is 2d, as we concatenated along the batch dimension: [n_time * n_batch, n_nodes]
            # hence we have to remove slices from the state matrix, or re-shape it into 3D, cut off some time steps
            # for each batch, and then reshape to 2D again.
            # TODO: please check if the reshaping really is correct, i.e. such that the first n_time entries of reservoir_states are the continuous reservoir states!
            indices_to_remove = discard_transients_indices(n_batch, n_time, self.discard_transients)
            reservoir_states = np.delete(reservoir_states, indices_to_remove, axis=0)
            print("reservoir_states.shape: ", reservoir_states.shape)
            # now the array should have the size of (n_batch*(n_time-discard), n_nodes)

            # update the value of n_time
            n_time = - self.discard_transients

            # reservoir_states, X, y = TransientRemover('RXY', reservoir_states, X, y, self.discard_transients)

        # make predictions y = R * W_out, W_out has a shape of [n_out, N]
        y_pred = np.dot(reservoir_states, self.readout_layer.weights)

        # reshape predictions into 3D [n_batch, n_time_out, n_state_out]
        n_time_out = int(y_pred.shape[0] / n_batch)
        n_states_out = y_pred.shape[-1]
        y_pred = y_pred.reshape(n_batch, n_time_out, n_states_out)

        return y_pred

    # @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 metrics: Union[str, list, None] = None) -> tuple:
        """
        Evaluate metrics on predictions made for input data.

        Args:
        X (np.ndarray): Input data of shape [n_batch, n_timesteps, n_states]
        y (np.ndarray): Target data of shape [n_batch, n_timesteps_out, n_states_out]
        metrics (Union[str, list, None], optional): List of metric names or a single metric name. If None, use metrics from .compile()

        Returns:
        tuple: Metric values
        """
        # evaluate metrics on predictions made for input data
        # expects: X of shape [n_batch, n_timesteps, n_states]
        # expects: y of shape [n_batch, n_timesteps_out, n_states_out]
        # depends on self.metrics = metrics from .compile()
        # returns float, if multiple metrics, then in given order (TODO: implement this)

        if metrics is None:  # user did not specify metric, take the one(s) given to .compile()
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
            raise ValueError(f'Cannot discard {self.discard_transients} as the number of time steps is {n_time}')

        if self.discard_transients > 0:
            y = y[:, self.discard_transients:, :]

        # get metric values
        metric_values = []
        for _metric_fun in metric_funs:
            metric_values.append(float(_metric_fun(y, y_pred)))

        return metric_values

    # @abstractmethod
    def get_params(self, deep=True):
        """
        Get parameters for scikit-learn compatibility.

        Args:
        deep (bool): If True, return a deep copy of parameters.

        Returns:
        dict: Dictionary of model parameters.
        """
        # needed for scikit-learn compatibility
        return {
            'input_layer': self.input_layer,
            'reservoir_layer': self.reservoir_layer,
            'readout_layer': self.readout_layer
        }

    # @abstractmethod
    def save(self, path: str):
        """
        Store the model to disk.

        Args:
        path (str): Path to save the model.
        """
        # store the model to disk
        pass

    def plot(self, path: str):
        """
        Print the model to some figure file.

        Args:
        path (str): Path to save the figure.
        """
        # print the model to some figure file
        pass

    def one_shot_pruning(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray, 
                         loss_metric='mse', prune_percent=0.1, n_calls=50, 
                         node_prop_extractor=None):
        """
        Perform one-shot pruning of the reservoir using Bayesian optimization.

        Args:
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation target data.
        loss_metric (str): Metric to use for evaluating model performance.
        prune_percent (float): Percentage of nodes to prune.
        n_calls (int): Number of iterations for Bayesian optimization.
        node_prop_extractor (NodePropExtractor, optional): Object to extract node properties.

        Returns:
        dict: History of the pruning process.
        """
        if node_prop_extractor is None:
            node_prop_extractor = NodePropExtractor()

        loss_fun = assign_metric(loss_metric)

        # Initial fit and score
        self.fit(X_train, y_train)
        init_score = loss_fun(y_val, self.predict(X_val))

        # Extract        
        properties = node_prop_extractor.extract_properties(self.reservoir_layer.weights)
        property_names = list(properties.keys())
        space = [Real(0, 1, name=f'{prop}_weight') for prop in property_names]

        best_score = float('inf')
        best_model = None

        @use_named_args(space)
        def objective(**params):
            nonlocal best_score, best_model
            pruned_model =            property_weights = [params[f'{prop}_weight'] for prop in property_names]
            pruned_model._prune(prune_percent, property_weights, node_prop_extractor)
            pruned_model.fit(X_train, y_train)
            y_pred = pruned_model.predict(X_val)
            score = loss_fun(y_val, y_pred) 
            
            if score < best_score:
                best_score = score
                best_model = copy.deepcopy(pruned_model)
            
            return score

        # Perform Bayesian optimization
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        # Update the current model with the best pruned model
        self.__dict__.update(best_model.__dict__)

        history = {
            'init_score': init_score,
            'final_score': best_score,
            'num_nodes': self.reservoir_layer.nodes,
            'best_params': dict(zip(property_names, result.x)),
        }

        return history

    def _prune(self, prune_percent, property_weights, node_prop_extractor):
        """
        Internal method to prune the reservoir based on node properties and weights.

        Args:
        prune_percent (float): Percentage of nodes to prune.
        property_weights (list): Weights for each node property.
        node_prop_extractor (NodePropExtractor): Object to extract node properties.
        """
        properties = node_prop_extractor.extract_properties(self.reservoir_layer.weights)
        
        # Calculate importance scores
        importance = np.zeros(self.reservoir_layer.nodes)
        for prop, weight in zip(properties.values(), property_weights):
            importance += weight * np.array(list(prop.values()))
        
        # Determine which nodes to keep
        n_keep = int(self.reservoir_layer.nodes * (1 - prune_percent))
        keep_indices = np.argsort(importance)[-n_keep:]
        
        # Update reservoir weights
        self.reservoir_layer.weights = self.reservoir_layer.weights[keep_indices][:, keep_indices]
        self.reservoir_layer.initial_res_states = self.reservoir_layer.initial_res_states[keep_indices]
        self.input_layer.weights = self.input_layer.weights[:, keep_indices]
        if self.readout_layer.weights is not None:
            self.readout_layer.weights = self.readout_layer.weights[keep_indices, :]
        
        # Update readout nodes
        self.readout_layer.readout_nodes = [i for i, idx in enumerate(keep_indices) if idx in self.readout_layer.readout_nodes]
        
        # Update the number of nodes in the reservoir
        self.reservoir_layer.nodes = n_keep

    def remove_node(self, node_index):
        """
        Remove a node from all relevant layers of the reservoir computer.

        Args:
        node_index (int): Index of the node to be removed.
        """
        # Remove node from reservoir layer weights
        self.reservoir_layer.weights = np.delete(self.reservoir_layer.weights, node_index, axis=0)
        self.reservoir_layer.weights = np.delete(self.reservoir_layer.weights, node_index, axis=1)

        # Remove node from initial reservoir states
        self.reservoir_layer.initial_res_states = np.delete(self.reservoir_layer.initial_res_states, node_index, axis=0)

        # Remove node from input layer weights
        self.input_layer.weights = np.delete(self.input_layer.weights, node_index, axis=1)

        # Remove node from readout layer weights
        self.readout_layer.weights = np.delete(self.readout_layer.weights, node_index, axis=0)

        # Update node count
        self.reservoir_layer.nodes = self.reservoir_layer.weights.shape[0]

        # Update readout nodes
        mask = self.readout_layer.readout_nodes != node_index
        self.readout_layer.readout_nodes = self.readout_layer.readout_nodes[mask]
        self.readout_layer.readout_nodes[self.readout_layer.readout_nodes > node_index] -= 1

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_calls=50):
        """
        Tune hyperparameters using Bayesian optimization.
        """
        space = [
            Integer(99, 100, name='nodes'),
            Real(0.1, 1.0, name='spec_rad'),
            Real(0.1, 1.0, name='leakage_rate'),
            Real(0.1, 1.0, name='fraction_input'),
            Real(0.1, 1.0, name='fraction_out'),
        ]

        @use_named_args(space)
        def objective(**params):
            # Create a new model with the suggested hyperparameters
            model = RC()
            model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(RandomReservoirLayer(nodes=params['nodes'], 
                                    spec_rad=params['spec_rad'], 
                                    leakage_rate=params['leakage_rate'], 
                                    fraction_input=params['fraction_input']))
            model.add(ReadoutLayer(output_shape=(y_train.shape[1], y_train.shape[2]), fraction_out=params['fraction_out']))
            
            # Compile and fit the model
            model.compile( metrics=['mse'])
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = model.evaluate(X_val, y_val)[0]  # Assuming MSE is the first metric
            return val_score

        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        # Set the best hyperparameters to the current model
        best_params = dict(zip([dim.name for dim in space], result.x))
        
        # Recreate the model with the best hyperparameters
        self.__init__()  # Reset the model
        self.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))
        self.add(RandomReservoirLayer(nodes=best_params['nodes'], 
                                spec_rad=best_params['spec_rad'], 
                                leakage_rate=best_params['leakage_rate'], 
                                fraction_input=best_params['fraction_input']))
        self.add(ReadoutLayer(output_shape=(y_train.shape[1], y_train.shape[2]), fraction_out=best_params['fraction_out']))
        
        # Compile and fit the model with the best hyperparameters
        self.compile( metrics=['mse'])
        self.fit(X_train, y_train)

        return best_params, result.fun



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


if __name__ == '__main__':
    print('hello')