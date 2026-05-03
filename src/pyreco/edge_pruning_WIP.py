"""
Capabilities to prune an existing RC model, i.e. try to cut reservoir nodes and improve 
performance while reducing the reservoir size
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from pyreco.custom_models import RC
from pyreco.edge_selector import EdgeSelector
import math
from typing import Union
import copy
from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.node_analyzer import NodeAnalyzer


class EdgePruner:
    # implements a pruning object for pyreco objects.

    PRUNING_CRITERION = {
        'performance': '_performance_pruning',
    }

    STOPPING_CRITERION = {
        'patience': '_patience_stopping',
        'min_edges': '_min_num_edges_stopping',
        'min_nodes': '_min_num_nodes_stopping',
    }

    def __init__(
        self,
        #edge_selection_strat: str = None,
        candidate_fraction: float = 0.1,
        pruning_criterion: str = 'performance',
        #stopping_criterion: str = None,
        stop_at_minimum: bool = True,
        min_num_nodes: int = 3,
        patience: int = 0,
        criterion: str = "mse",
        metrics: Union[list, str] = ["mse"],
        node_props_extractor=None,
        graph_props_extractor=None,
        return_best_model: bool = True,
        graph_analyzer: GraphAnalyzer = None,
        node_analyzer: NodeAnalyzer = None,
        remove_isolated_nodes: bool = False,
        #parallel: bool = False,

    ):
        """
        Initializer for the pruning class.

        Parameters:

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

        """

        # Sanity checks for the input parameter types and values
        if not isinstance(stop_at_minimum, bool):
            raise TypeError("stop_at_minimum must be a boolean")

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

        if not isinstance(criterion, str):
            raise TypeError("criterion must be a string")

        if graph_analyzer is not None and not isinstance(graph_analyzer, GraphAnalyzer):
            raise TypeError("graph_analyzer must be an instance of GraphAnalyzer")
        if graph_analyzer is None:
            graph_analyzer = GraphAnalyzer()

        if node_analyzer is not None and not isinstance(node_analyzer, NodeAnalyzer):
            raise TypeError("node_analyzer must be an instance of NodeAnalyzer")
        if node_analyzer is None:
            node_analyzer = NodeAnalyzer()
        
        # TODO Maybe make function just validating all intialization params
        #self._validate_pruning_criterion(pruning_criterion)
        #self.scoring_strategy = pruning_criterion

        # Assigning the parameters to instance variables

        self.criterion = criterion
        self.stop_at_minimum = stop_at_minimum
        self.min_num_nodes = min_num_nodes
        self.patience = patience
        self.candidate_fraction = candidate_fraction
        self.metrics = metrics
        self.return_best_model = return_best_model
        self.graph_analyzer = graph_analyzer
        self.node_analyzer = node_analyzer
        #### TODO this is only for testingf
        self.pruning_criterion = pruning_criterion

        # TODO not implemented yet
        self.remove_isolated_nodes = remove_isolated_nodes

        # store the history of the pruning process in a nested dictionary
        self.history = {}

        # initialize attributes that will be used during pruning (and changed during the process)
        # needs to be attributes as the history updates depend on them
        self._curr_loss = None
        self._curr_num_nodes = None
        self._curr_num_edges = None
        self._curr_loss_history = []
        self._idx_prune = None
        self._patience_counter = 0
        self._curr_metrics = None

    def prune(self, model: RC, data_train: tuple, data_val: tuple):
        """
        Prune a given model by removing edges.

        Parameters:
        - model (RC): The reservoir computer model to prune.
        - data_train (tuple): Training data.
        - data_val (tuple): Validation data.
        """

        # Sanity checks for the input parameter types and values
        self._validate_model(model)
        self._validate_train_val_data(data_train, data_val)

        # Obtain training and testing data
        x_test, y_test = data_val[0], data_val[1]
        x_train, y_train = data_train[0], data_train[1]

        # Assigning the parameters to instance variables that can not be set
        # in the initializer, as they depend on the model and data
        self._curr_num_nodes = model.reservoir_layer.nodes

        # initialize the quantities that affect the stop condition
        self._curr_loss = model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]
        self._curr_loss_history = [self._curr_loss]

        # initialize quantities that we track for the pruning history
        # these do not affect the pruning process
        self._curr_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)

        # storing all pruned models during the pruning iteration
        # allows to recover models from previous iterations, e.g. when the best model is not the last one in the iteration (positive patience value)
        _pruned_models = [copy.deepcopy(model)]

        # initialize the pruning iterator
        self._iter_count = 0

        # Store all relevant information during pruning inside self.history
        # self._update_pruning_history(model=model)
        self.add_val_to_history(["loss"], self._curr_loss)
        self.add_val_to_history(["metrics"], self._curr_metrics)
        self.add_val_to_history(["num_nodes"], self._curr_num_nodes)
        self.add_val_to_history(["iteration"], self._iter_count)

        _graph = model.reservoir_layer.weights
        _graph_props = self.graph_analyzer.extract_properties(graph=_graph)
        self.add_dict_to_history(["graph_props"], _graph_props)

        # Save the starting reservoir
        self.history["starting_reservoir"] = {
            "input_nodes": list(model.reservoir_layer.input_receiving_nodes),
            "readout_nodes": list(model.readout_layer.readout_nodes),
            "initial_weights": sp.csr_matrix(_graph) if isinstance(_graph, np.ndarray) else nx.to_scipy_sparse_array(_graph),
        }

        while True:  # self._curr_num_nodes>self.min_num_nodes:

            print(f'Currently at pruning iteration {self._iter_count} ...')

            print(
                f'Current reservoir size: {self._curr_num_nodes} | Current loss: {self._curr_loss:.8f}'
            )
            # what it was is down below
            _graph = model.reservoir_layer.weights

            # Get candidates for pruning (TODO: add additional passable arguments, e.g. strategy and fraction or make those attributes)
            _curr_candidates = self._get_candidates(_graph)

            # Apply chosen pruning strategy on candidates to get scores for candidates
            _candidate_scores, _candidate_models, _cand_graph_props_after = \
                self._apply_pruning_strategy(model, _curr_candidates, x_train, y_train, x_test, y_test)

            # Select candidate with best score and get according model

            #after trying out all candidate nodes, we need to select the node to prune,
            #i.e. the one that has the smallest loss among all candidate nodes
            idx_prune = np.argmin(_candidate_scores)
            #idx_prune = self._performance_pruning(model, _curr_candidates, x_train, y_train, x_test, y_test)
            self._curr_idx_prune = idx_prune 
            pruned_candidate =  _curr_candidates[idx_prune]# just for history logging

            # update the termination relevant quantities,
            # assuming that we will prune that node
            self._curr_loss = _candidate_scores[idx_prune]
            self._curr_num_nodes = _candidate_models[idx_prune].reservoir_layer.nodes
            self._curr_loss_history.append(self._curr_loss)

            # check if we should actually prune the node, or if that would violate the termination criteria (no optimal design by now to do it here though)
            if not self._keep_pruning():

                # exit the pruning loop
                break

            print(f"pruning candidate {pruned_candidate}, resulting in loss {self._curr_loss:.6f}")
            print(
                f"loss improvement by {((self._curr_loss_history[-2]-self._curr_loss)/self._curr_loss_history[-2]):+.3%}\n"
            )

            # prune the node that gives us the least performance drop. as we have already
            # pruned the node and stored the model, we only need to update the model.
            # Saves at least one training run and all the pruning logic
            model = _candidate_models[idx_prune]

            # store the model for later use
            _pruned_models.append(copy.deepcopy(model))

            # compute things that are required for the history, but not for the
            # pruning loop termination criteria
            self._curr_metrics = model.evaluate(
                x=x_test, y=y_test, metrics=self.metrics
            )

            # self._update_pruning_history

            self.add_val_to_history(["loss"], self._curr_loss)
            self.add_val_to_history(["metrics"], self._curr_metrics)
            self.add_val_to_history(["num_nodes"], self._curr_num_nodes)
            self.add_val_to_history(["idx_prune"], self._curr_idx_prune)
            self.add_val_to_history(["iteration"], self._iter_count)

            self.add_dict_to_history(
                ["graph_props"], _cand_graph_props_after[idx_prune]
            )

            # Data I want to save pruning details and weight snapshot ---
            # Save winner candidate and full candidate list for analysis
            self.add_val_to_history(["pruning_analysis", "winner"], _curr_candidates[idx_prune])
            self.add_val_to_history(["pruning_analysis", "candidates"], _curr_candidates)
            self.add_val_to_history(["pruning_analysis", "candidate_scores"], _candidate_scores)
            # Save snapshot of reservoir weights/connections 
            _w_curr = model.reservoir_layer.weights
            snapshot = sp.csr_matrix(_w_curr) if isinstance(_w_curr, np.ndarray) else nx.to_scipy_sparse_array(_w_curr)
            self.add_val_to_history(["weight_snapshots"], snapshot)

            # update counter
            self._iter_count += 1

        # in case we have a non-zero patience, we need to return the best model
        # instead of the last one (i.e. when a positive patience value was given)
        if self.return_best_model:
            idx_best = np.argmin(self._curr_loss_history[:-1])
            model = copy.deepcopy(_pruned_models[idx_best])
            print(f"returning model {idx_best} as the best, i.e. with lowest loss")

        # we should fit the final model, and evaluate it
        model.fit(x=x_train, y=y_train)
        final_loss = model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]
        final_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)
        print(
            f"\ninitial loss{self._curr_loss_history[0]:.6f}, loss after pruning: {final_loss:.6f}"
        )
        print(f"final model has {model.reservoir_layer.nodes} nodes")
        print(f"final model loss {self.criterion}: {final_loss:.6f}")
        print(f"final model metrics ({self.metrics}): {final_metrics}")
        return model, self.history
    

    ######### PRUNING STEPS FUNCTIONS #########

    def _get_candidates(self, graph):
        ## print(type(graph).__name__)print(np.array_equal(graph, graph.T))print(graph)print(_graph)print(_graph.shape)
        selector = EdgeSelector(
            graph=graph,
            strategy='random_uniform_wo_repl',
            directed=True
            )
        # obtain nodes that are proposed for pruning
        _curr_candidates = selector.select_edges(fraction=self.candidate_fraction)
        #print(_curr_candidates)
        print(
            f'Proposing {selector.num_select_edges}/{selector.num_total_edges} edges for pruning ...'
            )
        return _curr_candidates

    def _apply_pruning_strategy(self, model, candidates, x_train, y_train, x_test, y_test):
        '''
        Returns list of candidates and how good they are (#TODO rethink if scoring is better term for this)
        '''
        method = getattr(self, self.PRUNING_CRITERION[self.pruning_criterion])
        return method(model, candidates, x_train, y_train, x_test, y_test)

    def _get_best_candidate(self):
        pass

    def _check_stopping_criterion(self):
        pass

    def _keep_pruning(self):
        # Termination criteria for the pruning process

        # Keep pruning as long as all of the following conditions are met:
        # 1. The current score is below the target score
        # 2. The current number of nodes is above the minimum number of nodes
        # 3. The current loss is smaller than the previous loss

        if (
            self._min_num_nodes_stopping()
            and self._patience_stopping()
        ):
            return True
        else:
            return False

    ######### MODEL TRAINING FUNCTIONS #########
    def _retrain_model(self, model, x_train, y_train):
        return model.fit(x=x_train, y=y_train)

    ######### PRUNING CRITERIONS FUNCTIONS #########
    def _performance_pruning(self, model, candidates, x_train, y_train, x_test, y_test):

        # Initialize lists to track different metrics during pruning
        # Scores and models with candidate removes
        _candidate_scores = []
        _candidate_models = []

        # TODO Properties of the to-be removed edge
        _cand_edge_props = []

        # Properties of the to-be removed node
        _cand_node_props = []
        _cand_node_input_receiving = ([])  # Whether the node is connected to input layer
        _cand_node_output_sending = ([])  # Whether the node is connected to output layer

        # Properties of the graph
        _cand_graph_props_before = []  # Before pruning
        _cand_graph_props_after = []  # After pruning

        # Iteratate over the candidate edges
        # Delete one-by-one, measure performance, and also track node/graph-level properties
        for candidate in candidates:
            # Copy model to try out candidate removal
            _model = copy.deepcopy(model)

            # Get info on candidate egde and graph before removal (TODO rethink what to score)
            _graph = _model.reservoir_layer.weights
            _graph_props = self.graph_analyzer.extract_properties(graph=_graph)
            _cand_graph_props_before.append(_graph_props)

            # Remove candidate edge from reservoir
            _model.remove_reservoir_edges(edges=[candidate])
            # For logging/analysis: define node as the 'target' of the edge or skip node-specific props
            current_id = candidate


            # TODO: remove isolated nodes using utility function from utils_networks
            # if self.remove_isolated_nodes:
            # iso_nodes = ...
            # _model.remove_reservoir_nodes(nodes=[iso_nodes])

            # TODO: maintain the spectral radius of the reservoir layer
            # if self.maintain_spectral_radius:
            # spec_rad = model.reservoir_layer.spectral_radius
            # _model.set_spec_rad(spec_rad)

            # Re-fit (retrain) model to see effect of removal
            _model.fit(x=x_train, y=y_train)

            # Evaluate model with removed candidate edge (pruned model)
            _score = _model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]

            # Extract graph properties after pruning
            _graph = _model.reservoir_layer.weights
            _graph_props = self.graph_analyzer.extract_properties(graph=_graph)
            _cand_graph_props_after.append(_graph_props)

            # Format candidate identifier cleanly
            if isinstance(candidate, tuple):
                # Formats np.int64() cleanly
                label = f"{int(candidate[0])}-{int(candidate[1])}"
            else:
                # Handles a single node ID
                label = int(candidate)

            # Print info on effect of candidate removal
            print(
                f'Possible deletion of edge {label:<10} loss: {_score:.6f}  ({(self._curr_loss-_score)/self._curr_loss:+.3%})'
            )

            # Store score and model for removed candidate
            _candidate_scores.append(_score)
            _candidate_models.append(_model)

            # Delete temporary variables created for candidate (just for safety)
            del (
                _model,
                _score,
                _graph,
                _graph_props,
                #_node_props,
            )

        # store the candidate properties in the history object
        self.add_val_to_history(
            ["candidate_scores"],
            _candidate_scores,
        )

        self.add_val_to_history(
            ["candidates"],
            candidates,
        )

        self.add_val_to_history(
            ["candidate_node_props"],
            dictlist_to_dict(_cand_node_props),
        )

        self.add_val_to_history(
            ["candidate_graph_props_before"],
            dictlist_to_dict(_cand_graph_props_before),
        )

        self.add_val_to_history(
            ["candidate_graph_props_after"],
            dictlist_to_dict(_cand_graph_props_after),
        )

        return _candidate_scores, _candidate_models, _cand_graph_props_after

    def _shortest_path_pruning(self, model):
        # possible other pruning strat 
        pass
        
    ######### STOPPING CRITERIONS FUNCTIONS #########

    def _patience_stopping(self):
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
                    f'Loss decreased from {self._curr_loss_history[-2]:.6f} to {self._curr_loss_history[-1]:.6f} \nContinuing pruning ...'
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
        else:
            return True

    def _min_num_edges_stopping(self):
        # min_num_edges == model.weights
        pass

    def _min_num_nodes_stopping(self):
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
        # min_num_nodes == model.nodes

    def add_val_to_history(self, keys, value):
        """
        Add a value to history dictionary based on a list of keys.

        Args:
            keys (list): A list of keys specifying the path in the nested dictionary.
            value: The value to add.
        """
        if len(keys) == 1:
            if keys[0] not in self.history:
                self.history[keys[0]] = []
            self.history[keys[0]].append(value)

        elif len(keys) == 2:
            if keys[0] not in self.history:
                self.history[keys[0]] = {}
            if keys[1] not in self.history[keys[0]]:
                self.history[keys[0]][keys[1]] = []
            self.history[keys[0]][keys[1]].append(value)

        # for key in keys[:-1]:
        #     if key not in self.history:
        #         self.history[key] = {}
        #     self.history = self.history[key]
        # if keys[-1] not in self.history:
        #     self.history[keys[-1]] = []
        # self.history[keys[-1]].append(value)

    def add_dict_to_history(self, keys, value_dict):
        """
        Add a dictionary to history dictionary based on a list of keys.

        Args:
        nested_dict (dict): The nested dictionary.
        keys (list): A list of keys specifying the path in the nested dictionary.
        value_dict (dict): The dictionary to add.
        """

        for key in keys[:-1]:
            if key not in self.history:
                self.history[key] = {}
            self.history = self.history[key]
        if keys[-1] not in self.history:
            self.history[keys[-1]] = {}
        for k, v in value_dict.items():
            if k not in self.history[keys[-1]]:
                self.history[keys[-1]][k] = []
            self.history[keys[-1]][k].append(v)

    ######### VALIDATION FUNCTIONS #########
    def _validate_pruning_criterion(self, pruning_criterion):

        if pruning_criterion not in self.PRUNING_CRITERION:
            raise ValueError(f'scoring_strategy must be one of {self.PRUNING_CRITERION}')
        
    def _validate_stopping_criterion(self, stopping_criterion):

        if stopping_criterion not in self.STOPPING_CRITERION_CRITERION:
            raise ValueError(f'scoring_strategy must be one of {self.STOPPING_CRITERION_CRITERION}')

    def _validate_model(self, model):

        if not isinstance(model, RC):
            raise TypeError("model must be an instance of RC")

    def _validate_train_val_data(self, data_train, data_val):

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


def dictlist_to_dict(dict_list):
    """
    Join dictionaries in a list into a common dictionary.

    Args:
        dict_list (list): A list of dictionaries to join.

    Returns:
        dict: A common dictionary containing all key-value pairs from the dictionaries in the list.
    """
    common_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key in common_dict:
                if isinstance(common_dict[key], list):
                    common_dict[key].append(value)
                else:
                    common_dict[key] = [common_dict[key], value]
            else:
                common_dict[key] = value
    return common_dict

    # def _update_pruning_history(self, model: RC):
    #     # this will keep track of all quantities that are relevant during the pruning iterations.

    #     # Pruning iteration
    #     # self.history["iteration"].append(self._curr_iter)

    #     if not self.history:
    #         # initialize the history object
    #         self.history["iteration"] = []
    #         self.history["loss"] = []
    #         self.history["metrics"] = []
    #         self.history["num_nodes"] = []

    #         # initialize the dicts for the graph and node properties with empty lists
    #         graph_keys = self.graph_analyzer.list_properties()
    #         node_keys = self.node_analyzer.list_properties()

    #         self.history["graph_props"] = {key: [] for key in graph_keys}
    #         # self.history["candidate_graph_props"] = {key: [] for key in graph_keys}

    #         self.history["del_node_props"] = {key: [] for key in graph_keys}
    #         # self.history["candidate_node_props"] = {key: [] for key in graph_keys}

    #     else:
    #         # store the most relevant information

    #         # we will extract properties from the reservoir network of the model
    #         graph = model.reservoir_layer.weights
    #         graph_props = self.graph_analyzer.extract_properties(graph)

    #         # # choose the node to extract properties from
    #         # node = int(self._curr_idx_prune)
    #         # node_props = self.node_analyzer.extract_properties(graph, node)

    #         # high-level properties
    #         self.add_val_to_history(
    #             ["num_nodes"],
    #             self._curr_num_nodes,
    #         )

    #         self.add_val_to_history(
    #             ["loss"],
    #             self._curr_loss,
    #         )

    #         self.add_val_to_history(
    #             ["metrics"],
    #             self._curr_metrics,
    #         )

    #         self.add_val_to_history(
    #             ["iteration"],
    #             self._iter_count,
    #         )
    #         self.add_val_to_history(
    #             ["graph_props"],
    #             graph_props,
    #         )


def append_to_dict(dict1, dict2):
    # appends entries in dict1 to existing dict 2

    for key in list(dict1.keys()):
        if key in list(dict2.keys()):
            # print(f"appending {key} to existing dict")
            dict2[key].append(dict1[key])

    return dict2


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
            nodes=50,
            density=0.1,
            activation="tanh",
            leakage_rate=0.1,
            fraction_input=0.5,
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
    pruner = EdgePruner(
        #stop_at_minimum=False,
        stop_at_minimum=True,
        #min_num_nodes=46,
        patience=2,
        candidate_fraction=0.9,
        remove_isolated_nodes=False,
        metrics=["mse"],
    )

    model_pruned, history = pruner.prune(
        model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
    )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history["num_nodes"], history["loss"], label="loss")
    plt.xlabel("number of nodes")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    for key in history["graph_props"].keys():
        plt.plot(history["num_nodes"], history["graph_props"][key], label=key)
    plt.xlabel("number of nodes")
    plt.yscale("log")
    plt.legend()
    plt.show()


    # 1. Extract Sparsity (Density) over time
    # We use .nnz (number of non-zero elements) from the sparse snapshots
    total_possible_edges = history["starting_reservoir"]["initial_weights"].shape[0] ** 2
    densities = [snap.nnz / total_possible_edges for snap in history["weight_snapshots"]]
    losses = history["loss"]

    # 2. Create the Plots
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Loss
    ax1.set_xlabel('Pruning Iteration')
    ax1.set_ylabel('Loss (MSE)', color='tab:red')
    ax1.plot(losses, color='tab:red', linewidth=2, label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot Density on a twin axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Network Density (%)', color='tab:blue')
    ax2.plot(densities, color='tab:blue', linestyle='--', label='Density')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Performance vs. Reservoir Sparsity during Edge Pruning')
    fig.tight_layout()
    plt.show()

    # 3. Visualize the Reservoir "Skeleton" (Spy Plot)
    # Compare the first snapshot vs the last snapshot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.spy(history["weight_snapshots"][0], markersize=1)
    plt.title("Initial Reservoir")

    plt.subplot(1, 2, 2)
    plt.spy(history["weight_snapshots"][-1], markersize=1)
    plt.title("Pruned Reservoir")
    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    def visualize_pruning_topology(history, iteration_idx=-1):
        """
        Visualizes the reservoir graph at a specific iteration, 
        highlighting external connection roles.
        """
        # 1. Setup the Graph from the snapshot
        weights = history["weight_snapshots"][iteration_idx]
        G = nx.from_scipy_sparse_array(weights, create_using=nx.DiGraph)
        
        # 2. Identify Node Roles from history
        # (Using the metadata you saved in your starting_reservoir dict)
        input_receiving = set(history["starting_reservoir"]["input_nodes"])
        readout_nodes = set(history["starting_reservoir"]["readout_nodes"])
        
        # 3. Define Colors and Sizes
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            is_in = node in input_receiving
            is_out = node in readout_nodes
            
            if is_in and is_out:
                node_colors.append("#9b59b6") # Purple: Shortcut Node
                node_sizes.append(100)
            elif is_in:
                node_colors.append("#2ecc71") # Green: Input Only
                node_sizes.append(70)
            elif is_out:
                node_colors.append("#e74c3c") # Red: Output Only
                node_sizes.append(70)
            else:
                node_colors.append("#3498db") # Blue: Internal Reservoir Node
                node_sizes.append(40)

        # 4. Plotting
        plt.figure(figsize=(10, 8))
        # Using circular layout to see the "density" of the ring clearly
        pos = nx.circular_layout(G) 
        
        # Draw edges with low alpha so we can see the density
        nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color="gray", arrows=True, arrowsize=8)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        
        # Legend for clarity
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Input Only', markerfacecolor='#2ecc71', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Output Only', markerfacecolor='#e74c3c', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Shortcut (In+Out)', markerfacecolor='#9b59b6', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Hidden/Internal', markerfacecolor='#3498db', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        title_type = "Final" if iteration_idx == -1 else "Initial"
        plt.title(f"{title_type} Reservoir Topology (Nodes: {len(G.nodes)}, Edges: {weights.nnz})")
        plt.axis('off')
        plt.show()

    # --- Call the function for comparison ---
    visualize_pruning_topology(history, iteration_idx=0)  # Initial
    visualize_pruning_topology(history, iteration_idx=-1) # Final (Pruned)s