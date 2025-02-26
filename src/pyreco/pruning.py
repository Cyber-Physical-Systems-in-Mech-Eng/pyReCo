"""
Capabilities to prune an existing RC model, i.e. try to cut reservoir nodes and improve 
performance while reducing the reservoir size
"""

import numpy as np
from pyreco.custom_models import RC
from pyreco.node_selector import NodeSelector
import networkx as nx
import math
from typing import Union
import copy


class NetworkPruner:
    # implements a pruning object for pyreco objects.

    def __init__(
        self,
        target_score: float = None,
        stop_at_minimum: bool = True,
        min_num_nodes: int = 2,
        patience: int = 0,
        candidate_fraction: float = 0.1,
        remove_isolated_nodes: bool = False,
        criterion: str = "mse",
        metrics: Union[list, str] = ["mse"],
        maintain_spectral_radius: bool = False,
        node_props_extractor=None,
        graph_props_extractor=None,
        return_best_model: bool = True,
    ):
        """
        Initializer for the pruning class.

        Parameters:
        - target_score (float): The test set score that the user aims at. Pruning stops
        once this score is reached.

        - stop_at_minimum (bool): Whether to stop at the local minimum of the test set
        score. When set to False, pruning continues until the minimal number of nodes
        in <min_num_nodes>.

        - min_num_nodes (int): Stop pruning when arriving at this number of nodes.
        Conflicts if stop_at_minimum is set to True but also a min_num_nodes is given.

        - patience (int): We allow a patience, i.e. keep pruning after we reached a
        (local) minimum of the test set score. Depends on the size of the original
        reservoir network, defaults to 10% of initial reservoir nodes.

        - candidate_fraction (float): number of randomly chosen reservoir nodes during
        every pruning iteration that is a candidate for pruning. Refers to the fraction of nodes w.r.t. current number of nodes during pruning iteration.

        - remove_isolated_nodes (bool): Whether to remove isolated nodes during pruning.

        - criterion (str): The criterion to be used for steering the node pruning. Default is "mse".

        - metrics (list or str): The metrics to be used for evaluating the pruned model. Default is ["mse"].

        - maintain_spectral_radius (bool): Whether to maintain the spectral radius of the reservoir layer during pruning.
        """

        # Sanity checks for the input parameter types and values
        if target_score is not None and not isinstance(target_score, float):
            raise TypeError("target_score must be a float")

        if not isinstance(stop_at_minimum, bool):
            raise TypeError("stop_at_minimum must be a boolean")

        if min_num_nodes is not None:
            if not isinstance(min_num_nodes, int):
                raise TypeError("min_num_nodes must be an integer")
            if min_num_nodes <= 1:
                raise ValueError("min_num_nodes must be larger than 1")

        if patience is not None and not isinstance(patience, int):
            raise TypeError("patience must be an integer")

        if not isinstance(candidate_fraction, float):
            raise TypeError("candidate_fraction must be a float in (0, 1]")

        if candidate_fraction <= 0 or candidate_fraction > 1:
            raise ValueError("candidate_fraction must be a float in (0, 1]")

        if not isinstance(criterion, str):
            raise TypeError("criterion must be a string")

        # Assigning the parameters to instance variables
        if target_score is None:
            self.target_score = 0.0
        else:
            self.target_score = target_score
        self.criterion = criterion
        self.stop_at_minimum = stop_at_minimum
        self.min_num_nodes = min_num_nodes
        self.patience = patience
        self.candidate_fraction = candidate_fraction
        self.remove_isolated_nodes = remove_isolated_nodes
        self.metrics = metrics
        self.return_best_model = return_best_model

        self.maintain_spectral_radius = maintain_spectral_radius
        self.node_props_extractor = node_props_extractor
        self.graph_props_extractor = graph_props_extractor

        # store the history of the pruning process here
        self.history = {}

        # initialize attributes that will be used during pruning
        self._curr_loss = None
        self._curr_num_nodes = None
        self._curr_loss_history = []
        self._patience_counter = 0
        self._curr_metrics = None

    def prune(self, model: RC, data_train: tuple, data_val: tuple):
        """
        Prune a given model by removing nodes.

        Parameters:
        - model (RC): The reservoir computer model to prune.
        - data_train (tuple): Training data.
        - data_val (tuple): Validation data.
        """

        # Sanity checks for the input parameter types and values
        if not isinstance(model, RC):
            raise TypeError("model must be an instance of RC")

        if not isinstance(data_train, tuple) or not isinstance(data_val, tuple):
            raise TypeError("data_train and data_val must be tuples")

        if len(data_train) != 2 or len(data_val) != 2:
            raise ValueError("data_train and data_val must have 2 elements each")

        for idx, elem in enumerate(data_train):
            if not isinstance(elem, list):
                if not isinstance(elem, np.ndarray):
                    raise TypeError(f"data_train[{idx}] must be a list or numpy array")

        for idx, elem in enumerate(data_val):
            if not isinstance(elem, list):
                if not isinstance(elem, np.ndarray):
                    raise TypeError(f"data_val[{idx}] must be a list or numpy array")

        if len(data_train[0]) != len(data_train[1]):
            raise ValueError(
                "data_train[0] and data_train[1] must have the same length, "
                "i.e. same number of samples"
            )

        if len(data_val[0]) != len(data_val[1]):
            raise ValueError(
                "data_val[0] and data_val[1] must have the same length, "
                "i.e. same number of samples"
            )

        # obtain training and testing data
        x_test, y_test = data_val[0], data_val[1]
        x_train, y_train = data_train[0], data_train[1]

        # Assigning the parameters to instance variables that can not be set
        # in the initializer, as they depend on the model and data
        # TODO: any?

        # initialize the quantities that affect the stop condition
        self._curr_loss = model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]
        self._curr_num_nodes = model.reservoir_layer.nodes
        self._curr_loss_history = [self._curr_loss]
        self._patience_counter = 0

        # initialize quantities that we track for the pruning history
        # these do not affect the pruning process
        self._curr_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)

        # Store all relevant information during pruning inside self.history
        # self._update_pruning_history()

        _pruned_models = [copy.deepcopy(model)]

        iter_count = 0
        while True:  # self._curr_num_nodes>self.min_num_nodes:

            print(f"pruning iteration {iter_count}")

            print(
                f"current reservoir size: {self._curr_num_nodes}, current loss: {self._curr_loss:.8f}"
            )

            # propose a list of nodes to prune using a random uniform distribution. If the user specified a candidate_fraction of 1.0, we will try out all nodes
            _num_nodes_to_prune = math.ceil(
                self.candidate_fraction * self._curr_num_nodes
            )
            selector = NodeSelector(
                total_nodes=self._curr_num_nodes, strategy="random_uniform_wo_repl"
            )
            # obtain nodes that are proposed for pruning
            _curr_candidate_nodes = selector.select_nodes(num=_num_nodes_to_prune)
            print(
                f"propose {_num_nodes_to_prune}/{self._curr_num_nodes} nodes for pruning"
            )

            # track the performance of the RC with the candidate nodes removed
            _candidate_scores = []
            _candidate_models = []

            # _node_scores = []
            # _cand_node_props = []
            # _cand_graph_props = []

            # iteratate over the candidate nodes: delete one-by-one, measure performance,
            # and also track node/graph-level properties
            for node in _curr_candidate_nodes:

                # get a copy of the original model to try out the deletion
                _model = copy.deepcopy(model)

                # remove current candidate node
                _model.remove_reservoir_nodes(nodes=[node])

                # TODO: remove isolated nodes using utility function from utils_networks
                # if self.remove_isolated_nodes:
                # iso_nodes = ...
                # _model.remove_reservoir_nodes(nodes=[iso_nodes])

                # TODO: maintain the spectral radius of the reservoir layer
                # if self.maintain_spectral_radius:
                # spec_rad = model.reservoir_layer.spectral_radius
                # _model.set_spec_rad(spec_rad)

                # pruning requires re-fitting the model
                _model.fit(x=x_train, y=y_train)

                # evaluate the pruned model
                _score = _model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]

                print(
                    f"deletion of candidate node {node}. loss: \t{_score:.6f} ({(self._curr_loss-_score)/self._curr_loss:+.3%})"
                )

                # TODO: extract node and graph properties for candidate
                _graph = _model.reservoir_layer.weights
                # _node_props = self.node_props_extractor.characterize(_graph, node)
                # _graph_props = self.graph_props_extractor.characterize(_graph)

                # store the relevant candidate information
                _candidate_scores.append(_score)
                _candidate_models.append(_model)
                # _candidate_node_props.append(_node_props)
                # _candidate_graph_props.append(_graph_props)

                # delete temporary variables (just for safety)
                del _model, _score, _graph

            # after trying out all candidate nodes, we need to select the node to prune,
            # i.e. the one that has the smallest loss among all candidate nodes
            idx_prune = np.argmin(_candidate_scores)

            # update the termination relevant quantities,
            # assuming that we will prune that node
            self._curr_loss = _candidate_scores[idx_prune]
            self._curr_num_nodes = _candidate_models[idx_prune].reservoir_layer.nodes
            self._curr_loss_history.append(self._curr_loss)

            # check if we should actually prune the node, or if that would violate the termination criteria (no optimal design by now to do it here though)
            if not self._keep_pruning():

                # TODO: remove last entries from the history to not report
                # sth. that is acually not happening (i.e. pruning the current node)

                # exit the pruning loop
                break

            print(f"pruning node {idx_prune}, resulting in loss {self._curr_loss:.6f}")
            print(
                f"loss improvement by {((self._curr_loss_history[-2]-self._curr_loss)/self._curr_loss_history[-2]):+.3%}\n"
            )

            # prune the node that gives us the least performance drop. as we have already
            # pruned the node and stored the model, we only need to update the model. Saves # at least one training run and all the pruning logic
            model = _candidate_models[idx_prune]

            # update the pruning history object with all relevant quantities
            self._curr_metrics = model.evaluate(
                x=x_test, y=y_test, metrics=self.metrics
            )
            # self._update_pruning_history()
            _pruned_models.append(copy.deepcopy(model))

            # update counter
            iter_count += 1

        # in case we have a non-zero patience, we need to return the best model
        # instead of the last one
        if self.return_best_model:
            idx_best = np.argmin(self._curr_loss_history)
            model = copy.deepcopy(_pruned_models[idx_best])
            print(f"returning model {idx_best} as the best, i.e. with lowest loss")

        # we should fit the final model
        model.fit(x=x_train, y=y_train)
        final_loss = model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]
        final_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)
        print(f"\nfinal model has {model.reservoir_layer.nodes} nodes")
        print(f"final model loss: {final_loss:.6f}")
        print(f"final model metrics ({self.metrics}): {final_metrics}")
        return model, self.history

    def _keep_pruning(self):
        # Termination criteria for the pruning process

        # Keep pruning as long as all of the following conditions are met:
        # 1. The current score is below the target score
        # 2. The current number of nodes is above the minimum number of nodes
        # 3. The current loss is smaller than the previous loss

        if (
            self._loss_not_met()
            and self._num_nodes_not_met()
            and self._at_minimum_not_met()
        ):
            return True
        else:
            return False

    def _loss_not_met(self):
        # Checks the stopping condition based on the current and target loss
        # returns True if the criterion is not met, i.e. we should continue pruning
        if self._curr_loss >= self.target_score:
            print(
                f"Loss {self._curr_loss:.6f} is larger target score {self.target_score:.6f}. Continuing pruning"
            )
            return True
        else:
            print(
                f"Loss {self._curr_loss:.6f} is smaller target score {self.target_score:.6f}. Terminating pruning"
            )
            return False

    def _num_nodes_not_met(self):
        # checks if the number of nodes is above the minimum number of nodes
        # returns True if number of nodes is above minimum, i.e. we should continue pruning
        if self._curr_num_nodes > self.min_num_nodes:
            print(
                f"Number of nodes {self._curr_num_nodes} is larger than minimum number of nodes {self.min_num_nodes}. Continuing pruning"
            )
            return True
        else:
            print(
                f"Number of nodes {self._curr_num_nodes} is smaller/equal minimum number of nodes {self.min_num_nodes}. Terminating pruning"
            )
            return False

    def _at_minimum_not_met(self):
        # checks if the loss is at a minimum,
        # considering also patience.
        # returns True if loss is not at minimum, i.e. we should continue pruning
        if len(self._curr_loss_history) < 2:
            # we are just at the start of pruning, cannot
            # check for a minimum.
            return True

        if self.stop_at_minimum:
            if self._curr_loss_history[-2] > self._curr_loss_history[-1]:
                print(
                    f"Loss decreased from {self._curr_loss_history[-2]:.6f} to {self._curr_loss_history[-1]:.6f}. Continuing pruning"
                )
                self._patience_counter = 0
                return True
            else:  # current loss is larger than previous
                self._patience_counter += 1
                if self._patience_counter < self.patience:
                    print(
                        f"Loss increased, but {self._patience_counter} < {self.patience} Continuing pruning"
                    )
                    return True
                else:
                    # TODO: we need to recover the model that had the best score!
                    print(
                        f"Loss increased for {self.patience} consecutive iterations. Terminating pruning"
                    )
                    return False

    def _update_pruning_history(self):
        # this will keep track of all quantities that are relevant during the pruning iterations.

        # Pruning iteration
        self.history["iteration"].append(self._curr_iter)

        # history["num_nodes"] = [n]
        # history["score"] = [curr_score]
        # history["graph_props"] = []  # graph property extractor
        # history["node_props"] = []  # node property extractor
        # history["node_idx_pruned"] = []  # index of the node pruned per iteration
        # history["pruned_node_props"] = []  # node property of the node that was pruned

        """
        Candidate nodes: trying out different nodes at current iteration
        """

        #       # some logging
        # self._curr_idx_prune = idx
        # self._curr_nodes_before_pruning = _curr_total_nodes
        # self._curr_nodes_after_pruning = model.reservoir_layer.nodes
        # self._curr_candidate_nodes = _curr_candidate_nodes
        # self._curr_candidate_scores = _candidate_scores
        # self._curr_score = _candidate_scores[idx]
        # self._curr_node_props = _cand_node_props[idx]
        # self._curr_graph_props = _cand_graph_props[idx]

        # # Number of nodes that are being tried to at current iteration
        # self.history["num_candidate_nodes"].append(len(self._curr_candidate_nodes))

        # # Nodes indices of nodes that are being tried to at current iteration
        # self.history["candidate_nodes"].append(self._curr_candidate_nodes)

        # # Scores obtained when deleting a respective candidate node
        # self.history["candidate_node_scores"].append(self._curr_candidate_scores)

        """
        Final choice: node to prune is selected
        """
        # # Stats once a nodes is chosen to be deleted finally
        # self.history["node_idx_pruned"].append(self._curr_idx_prune)


if __name__ == "__main__":
    # test the pruning

    from pyreco.utils_data import sequence_to_sequence as seq_2_seq
    from pyreco.custom_models import RC as RC
    from pyreco.layers import InputLayer, ReadoutLayer
    from pyreco.layers import RandomReservoirLayer
    from pyreco.optimizers import RidgeSK

    # get some data
    X_train, X_test, y_train, y_test = seq_2_seq(
        name="sine_pred", n_batch=20, n_states=2, n_time=150
    )

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]

    # build a classical RC
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(
        RandomReservoirLayer(
            nodes=30,
            density=0.1,
            activation="tanh",
            leakage_rate=0.1,
            fraction_input=1.0,
        ),
    )
    model.add(ReadoutLayer(output_shape, fraction_out=0.9))

    # Compile the model
    optim = RidgeSK(alpha=0.5)
    model.compile(
        optimizer=optim,
        metrics=["mean_squared_error"],
    )

    # Train the model
    model.fit(X_train, y_train)

    print(f"score: \t\t\t{model.evaluate(x=X_test, y=y_test)[0]:.4f}")

    # prune the model
    pruner = NetworkPruner(
        stop_at_minimum=True,
        min_num_nodes=20,
        patience=5,
        candidate_fraction=0.1,
        remove_isolated_nodes=False,
        metrics=["mse"],
        maintain_spectral_radius=False,
    )

    model_pruned = pruner.prune(
        model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
    )

"""
Some left-over



def evaluate_node_removal(
        self, X, y, loss_fun, init_score, del_idx, current_num_nodes
    ):
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
        
            Determine if pruning should continue based on current performance.
          
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


        
    def remove_node(self, node_indices):

        Remove one or multiple nodes from all relevant layers of the reservoir computer.

        Args:
        node_indices (int or list or np.array): Index or indices of the nodes to be removed.
  
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


"""
