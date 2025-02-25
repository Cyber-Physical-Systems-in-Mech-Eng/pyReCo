"""
Capabilities to prune an existing RC model, i.e. try to cut reservoir nodes and improve 
performance while reducing the reservoir size
"""

import numpy as np
from pyreco.custom_models import RC
from pyreco.node_selector import NodeSelector
import networkx as nx
import math


class NetworkPruner:
    # implements a pruning object for pyreco objects.

    def __init__(
        self,
        target_score: float = None,
        stop_at_minimum: bool = True,
        min_num_nodes: int = 2,
        patience: int = None,
        candidate_fraction: float = 0.1,
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
        """

        # Sanity checks for the input parameter types and values
        if target_score is not None and not isinstance(target_score, float):
            raise TypeError("target_score must be a float")

        if not isinstance(stop_at_minimum, bool):
            raise TypeError("stop_at_minimum must be a boolean")

        if min_num_nodes is not None:
            if not isinstance(min_num_nodes, int):
                raise TypeError("min_num_nodes must be an integer")
            if min_num_nodes <= 2:
                raise ValueError("min_num_nodes must be larger than 2")

        if patience is not None and not isinstance(patience, int):
            raise TypeError("patience must be an integer")

        if not isinstance(candidate_fraction, float):
            raise TypeError("candidate_fraction must be a float in (0, 1]")

        if candidate_fraction <= 0 or candidate_fraction > 1:
            raise ValueError("candidate_fraction must be a float in (0, 1]")

        # Additional sanity checks: logical constraints on the input parameters
        if min_num_nodes is not None and stop_at_minimum:
            raise ValueError("min_num_nodes conflicts with stop_at_minimum set to True")

        # Assigning the parameters to instance variables
        self.target_score = target_score
        self.stop_at_minimum = stop_at_minimum
        self.min_num_nodes = min_num_nodes
        self.patience = patience
        self.candidate_fraction = candidate_fraction
        self.history = dict()

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
            if not isinstance(elem, list) or not isinstance(elem, np.ndarray):
                raise TypeError(f"data_train[{idx}] must be a list or numpy array")

        for idx, elem in enumerate(data_val):
            if not isinstance(elem, list) or not isinstance(elem, np.ndarray):
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

        # Assigning the parameters to instance variables that can not be set
        # in the initializer
        n = model.reservoir_layer.nodes  # number of initial reservoir nodes
        curr_score = model.evaluate(X=data_val[0], y=data_val[1])[0]

        if self.patience is None:
            self.patience = int(n / 10)

        # Initialize a dict that stores all relevant information during pruning
        history = dict()
        history["num_nodes"] = [n]
        history["score"] = [curr_score]
        history["graph_props"] = []  # graph property extractor
        history["node_props"] = []  # node property extractor
        history["node_idx_pruned"] = []  # index of the node pruned per iteration
        history["pruned_node_props"] = []  # node property of the node that was pruned

        # Initialize stopping criteria values
        self._curr_score = model.evaluate(X=data_val[0], y=data_val[1])

        iter = 0

        while self._keep_pruning():

            self._curr_iter = iter
            _curr_total_nodes = model.reservoir_layer.nodes

            # propose a list of nodes to prune using a random uniform distribution. If the user specified a candidate_fraction of 1.0, we will try out all nodes
            _num_nodes_to_prune = math.ceil(self.candidate_fraction * _curr_total_nodes)
            selector = NodeSelector(
                total_nodes=_curr_total_nodes, strategy="uniform_random_wo_repl"
            )
            _curr_candidate_nodes = selector.select_nodes(num=_num_nodes_to_prune)

            _node_scores = []
            _candidate_scores = []
            _cand_node_props = []
            _cand_graph_props = []

            for node in _curr_candidate_nodes:

                print(f"pruning iteration {iter}: deleting candidate node {node}")

                # get a copy of the original model
                _model = model

                # _model._remove_nodes(node)  # will also delete unconnected nodes
                # _model.set_spec_rad()
                # _model.fit(X_train,y_train)
                # _candidate_scores.append(_model.evaluate(X_test, y_test)[0])

                # _cand_node_props.append(NodePropsExtractor())
                # _cand_graph_props(GraphPropsExtractor())

                del _model

            # select the node to prune
            idx = np.argmax(_node_scores)

            # model._remove_nodes(node)  # will also delete unconnected nodes
            # model.set_spec_rad()
            # model.fit(X_train, y_train)

            # some logging
            self._curr_idx_prune = idx
            self._curr_nodes_before_pruning = _curr_total_nodes
            self._curr_nodes_after_pruning = model.reservoir_layer.nodes
            self._curr_candidate_nodes = _curr_candidate_nodes
            self._curr_candidate_scores = _candidate_scores
            self._curr_score = _candidate_scores[idx]
            self._curr_node_props = _cand_node_props[idx]
            self._curr_graph_props = _cand_graph_props[idx]

            self._update_pruning_history()

            # update counter
            iter += 1

        pass

    def _update_pruning_history(self):
        # this will keep track of all quantities that are relevant during the pruning iterations.

        # Pruning iteration
        self.history["iteration"].append(self._curr_iter)

        """
        Candidate nodes: trying out different nodes at current iteration
        """

        # Number of nodes that are being tried to at current iteration
        self.history["num_candidate_nodes"].append(len(self._curr_candidate_nodes))

        # Nodes indices of nodes that are being tried to at current iteration
        self.history["candidate_nodes"].append(self._curr_candidate_nodes)

        # Scores obtained when deleting a respective candidate node
        self.history["candidate_node_scores"].append(self._curr_candidate_scores)

        """
        Final choice: node to prune is selected
        """
        # Stats once a nodes is chosen to be deleted finally
        self.history["node_idx_pruned"].append(self._curr_idx_prune)

    def _keep_pruning(self):

        if score > self.target_score:
            return True

        if num_nodes > self.min_num_nodes:
            return True

        return False


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
