"""
Capabilities to prune an existing RC model, i.e. try to cut reservoir nodes and improve 
performance while reducing the reservoir size
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from joblib import Parallel, delayed #TODO check if we need to declare somwhere that this now is a needed import
from tqdm import tqdm #TODO check if we need to declare somwhere that this now is a needed import
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
        #'structural_pruning': 'structural_pruning' TODO implement this
    }

    STOPPING_CRITERION = {
        'patience': '_patience_stopping',
        'min_nodes': '_min_num_nodes_stopping',
        'min_edges': '_min_num_edges_stopping'
    }

    def __init__(
        self,
        edge_selection_strat: str = 'random_uniform_wo_repl',
        candidate_fraction: float = 0.1,
        pruning_criterion: str = 'performance',
        stopping_criterion: list = ['patience'],
        min_num_nodes: int = 3,
        min_num_edges: int = 2,
        patience: int = 0,
        performance_criterion: str = "mse",
        metrics: Union[list, str] = ["mse"],
        node_props_extractor=None,
        graph_props_extractor=None,
        return_best_model: bool = True,
        graph_analyzer: GraphAnalyzer = None,
        node_analyzer: NodeAnalyzer = None,
        remove_isolated_nodes: bool = False,
        directed: bool = True,
        parallel: bool = False,

    ):
        """
        Initializer for the pruning class.

        Parameters:

        - min_num_nodes (int): Stop pruning when arriving at this number of nodes.

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
        self._validate_init_params(
            candidate_fraction,
            pruning_criterion,
            stopping_criterion,
            min_num_nodes,
            #min_num_edges,  TODO create validation function
            patience,
            performance_criterion,
            metrics,
            node_props_extractor,
            graph_props_extractor,
            return_best_model,
            graph_analyzer,
            node_analyzer,
            remove_isolated_nodes,
            #parallel,  TODO create validation function
            )

        if graph_analyzer is None:
            graph_analyzer = GraphAnalyzer()
        if node_analyzer is None:
            node_analyzer = NodeAnalyzer()

        # Assigning the parameters to instance variables
        # Parameters for pruning criterion
        self.criterion = performance_criterion
        self.pruning_criterion = pruning_criterion
        # Parameters for stopping criterion
        self.stopping_criterion = stopping_criterion
        self.min_num_nodes = min_num_nodes
        self.min_num_edges = min_num_edges
        self.patience = patience
        # Parameters for candidate selection
        self.candidate_fraction = candidate_fraction
        self.edge_selection_strat = edge_selection_strat
        self.directed = directed
        # Parameters for tracking metrics
        self.metrics = metrics
        self.graph_analyzer = graph_analyzer
        self.node_analyzer = node_analyzer
        # Parameter bools for extra functionalities
        self.return_best_model = return_best_model
        self.remove_isolated_nodes = remove_isolated_nodes
        self.parallel = parallel 

        # Initialize history dict to store the history of the pruning process in a
        #  nested dictionary
        self.history = {}

        # Initialize attributes that will be used during pruning (and changed during
        #  the process)
        #   needs to be attributes as the history updates depend on them
        self._curr_model = None  # TODO check this again
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
        self._validate_pruning_params(model, data_train, data_val)

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

        _graph = model.reservoir_layer.weights
        _graph_props = self.graph_analyzer.extract_properties(graph=_graph)

        # Initialize history with important data
        self._intialize_history(_graph_props)

        # self.add_dict_to_history(["graph_props"], _graph_props)

        # Save the starting reservoir
        self.history["starting_reservoir"] = {
            "input_nodes": list(model.reservoir_layer.input_receiving_nodes),
            "readout_nodes": list(model.readout_layer.readout_nodes),
            "initial_weights": sp.csr_matrix(_graph) if isinstance(_graph, np.ndarray) else nx.to_scipy_sparse_array(_graph),
        }

        while True:

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

            # Out of all candidates select candidate with best score (the one to prune)
            idx_prune, pruned_candidate = self._get_best_candidate(_curr_candidates, _candidate_scores)
            self._curr_idx_prune = idx_prune
            # self._get_best_candidate_model_properties(_candidate_scores, _candidate_models)

            # Get model properties of selected candidate and update the termination relevant quantities
            curr_model, curr_loss, curr_num_nodes, curr_num_edges = \
                self._get_best_candidate_model_properties(idx_prune, _candidate_scores, _candidate_models)
            # TODO rethink if we should do the setting of these props above already and rename function so we can use it more
            #  and then also do current amount of edges
            self._curr_model = curr_model
            self._curr_loss = curr_loss
            self._curr_num_nodes = curr_num_nodes
            self._curr_num_edges = curr_num_edges
            self._curr_loss_history.append(self._curr_loss)

            # Check for isolated nodes and remove TODO if no effect on performance
            # TODO: remove isolated nodes using utility function from utils_networks (follow up on this)
            if self.remove_isolated_nodes:
                isolated_nodes = self._get_isolated_nodes(self._curr_model)
                self._curr_model = self._remove_isolated_nodes(isolated_nodes, self._curr_model)
                self._curr_num_nodes = self._curr_model.reservoir_layer.nodes

            # Check stopping criterion on to be pruned candidate model properties
            # If termination criteria would be violated by pruning candidate we stop pruning
            # TODO (no optimal design by now to do it here though)
            if self._check_stopping_criterion():
                # Exit pruning loop if stopping criterion condition is met
                break

            if isinstance(pruned_candidate, tuple):
                # clean print of pruned_candidate
                pruned_candidate = (int(pruned_candidate[0]), int(pruned_candidate[1]))
            print(f'Pruning candidate {pruned_candidate}, resulting in loss {self._curr_loss:.6f}')
            print(
                f'Loss improvement by {((self._curr_loss_history[-2]-self._curr_loss)/self._curr_loss_history[-2]):+.3%}\n'
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

            # Store important data after pruning in history
            self._update_history_after_prune_iter(model, idx_prune, _curr_candidates, _candidate_scores, _cand_graph_props_after)

            # update counter
            self._iter_count += 1

        # in case we have a non-zero patience, we need to return the best model
        # instead of the last one (i.e. when a positive patience value was given)
        if self.return_best_model:
            idx_best = np.argmin(self._curr_loss_history[:-1])
            model = copy.deepcopy(_pruned_models[idx_best])
            print(f"Returning model {idx_best} as the best (with lowest loss)")

        # we should fit the final model, and evaluate it
        model.fit(x=x_train, y=y_train)
        final_loss = model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]
        final_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)
        print(
            f"\nInitial loss: {self._curr_loss_history[0]:.6f}, loss after pruning: {final_loss:.6f}"
        )
        print(f"Final model has {model.reservoir_layer.nodes} nodes")
        print(f"Final model loss {self.criterion}: {final_loss:.6f}")
        print(f"Final model metrics ({self.metrics}): {final_metrics}")
        return model, self.history


    ######### PRUNING STEPS FUNCTIONS #########

    def _intialize_history(self, graph_props):
        # Store all relevant information during pruning inside self.history
        # self._update_pruning_history(model=model)
        self.add_val_to_history(["loss"], self._curr_loss)
        self.add_val_to_history(["metrics"], self._curr_metrics)
        self.add_val_to_history(["num_nodes"], self._curr_num_nodes)
        self.add_val_to_history(["iteration"], self._iter_count)

        self.add_dict_to_history(["graph_props"], graph_props)

    def _get_candidates(self, graph):

        selector = EdgeSelector(
            graph=graph,
            strategy=self.edge_selection_strat,
            directed=self.directed
            )

        # obtain edges for pruning
        _curr_candidates = selector.select_edges(fraction=self.candidate_fraction)

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

    def _update_history_during_prune_iter(self,
                                          candidate_scores,
                                          candidates,
                                          cand_graph_props_before,
                                          cand_graph_props_after
                                          ):
        self.add_val_to_history(
            ["candidate_scores"],
            candidate_scores,
        )

        self.add_val_to_history(
            ["candidates"],
            candidates,
        )

        self.add_val_to_history(
            ["candidate_graph_props_before"],
            dictlist_to_dict(cand_graph_props_before),
        )

        self.add_val_to_history(
            ["candidate_graph_props_after"],
            dictlist_to_dict(cand_graph_props_after),
        )

    def _get_best_candidate(self, candidates, candidate_scores):
        idx_prune = np.argmin(candidate_scores)
        pruned_candidate = candidates[idx_prune]  # just for history logging

        return idx_prune, pruned_candidate

    def _get_best_candidate_model_properties(self, candidate_idx, candidate_scores, candidate_models):
        curr_loss = candidate_scores[candidate_idx]
        curr_num_nodes = candidate_models[candidate_idx].reservoir_layer.nodes
        curr_model = candidate_models[candidate_idx]
        curr_graph = curr_model.reservoir_layer.weights

        if isinstance(curr_graph, nx.Graph):
            edge_indices = list(curr_graph.edges())
        elif isinstance(curr_graph, np.ndarray):
            rows, cols = np.where(curr_graph != 0)  # where entries are not zero
            edge_indices = list(zip(rows, cols))
            if not self.directed:
                edge_indices = [(r, c) for r, c in edge_indices if r < c]

        curr_num_edges = len(edge_indices)

        return curr_model, curr_loss, curr_num_nodes, curr_num_edges

    def _get_isolated_nodes(self, model):
        """
        TODO confirm pyreco RC have same logic as nx objects
        Find nodes that are isolated in the reservoir graph.
        Returns a list of isolated node ids, and whether their input or output nodes.
        """
        graph = model.reservoir_layer.weights
        input_nodes = set(model.reservoir_layer.input_receiving_nodes)
        readout_nodes = set(model.readout_layer.readout_nodes)

        isolated_nodes = []
        if isinstance(graph, nx.Graph):
            for node in graph.nodes():
                if graph.degree(node) == 0:
                    isolated_nodes.append({
                        'id': node,
                        'is_input': node in input_nodes,
                        'is_readout': node in readout_nodes,
                    })
        elif isinstance(graph, np.ndarray):
            for node in range(graph.shape[0]):
                if np.count_nonzero(graph[node, :]) == 0 and np.count_nonzero(graph[:, node]) == 0:
                    isolated_nodes.append({
                        'id': node,
                        'is_input': node in input_nodes,
                        'is_readout': node in readout_nodes,
                    })

        return isolated_nodes

    def _remove_isolated_nodes(self, isolated_nodes, model):
        # Remove isolated nodes that are neither input-receiving nor readout nodes # TODO clear up question
        fully_isolated_nodes = [
            n['id'] for n in isolated_nodes
            if not n['is_input'] and not n['is_readout']
        ]
        if fully_isolated_nodes:
            print(f'Removing {len(fully_isolated_nodes)} isolated non-input/readout nodes: {fully_isolated_nodes}')
            model.remove_reservoir_nodes(nodes=fully_isolated_nodes)
        return model

    def _check_stopping_criterion(self):
        for criterion in self.stopping_criterion:
            method = getattr(self, self.STOPPING_CRITERION[criterion])
            if not method():
                # Criterion is met
                return True
        # Criterion is not met
        return False

    def _update_history_after_prune_iter(self, model, idx_prune, candidates, candidate_scores, graph_props_after):
        """Store all relevant state after each pruning iteration."""
        self.add_val_to_history(["loss"], self._curr_loss)
        self.add_val_to_history(["metrics"], self._curr_metrics)
        self.add_val_to_history(["num_nodes"], self._curr_num_nodes)
        self.add_val_to_history(["idx_prune"], self._curr_idx_prune)
        self.add_val_to_history(["iteration"], self._iter_count)

        self.add_dict_to_history(
            ["graph_props"], graph_props_after[idx_prune]
        )

        # Data I want to save pruning details and weight snapshot ---
        # Save winner candidate and full candidate list for analysis
        self.add_val_to_history(["pruning_analysis", "winner"], candidates[idx_prune])
        self.add_val_to_history(["pruning_analysis", "candidates"], candidates)
        self.add_val_to_history(["pruning_analysis", "candidate_scores"], candidate_scores)

        # Save snapshot of reservoir weights/connections
        _w_curr = model.reservoir_layer.weights
        snapshot = sp.csr_matrix(_w_curr) if isinstance(_w_curr, np.ndarray) else nx.to_scipy_sparse_array(_w_curr)
        self.add_val_to_history(["weight_snapshots"], snapshot)

    ######### MODEL TRAINING FUNCTIONS #########
    def _retrain_model(self, model, x_train, y_train):
        return model.fit(x=x_train, y=y_train)

    ######### PRUNING CRITERIONS FUNCTIONS #########
    def _performance_pruning(self, model, candidates, x_train, y_train, x_test, y_test):

        if self.parallel:
            # Parallelizing performance evaluation of candidates
            # Kicking off parallelization of going through all candidates
            # tqdm shows process in in bar chart
            # n_jobs is number of jobs to run in parallel (-1 uses all CPU cores)
            # backend -  loky is default TODO reevaluate is this is most suitable
            parallel = Parallel(n_jobs=-1, backend='loky')
            # tqdm shows process in in bar chart
            # generator returns results in order that they're given
            results = parallel(
                               delayed(self._evaluate_candidate_performance)
                               (model, c, x_train, y_train, x_test, y_test)
                               for c in tqdm(candidates, desc="Evaluating candidates")
                               )
            _candidate_scores, _candidate_models, _cand_graph_props_before, _cand_graph_props_after = zip(*results)
            _candidate_scores = list(_candidate_scores)
            _candidate_models = list(_candidate_models)
            _cand_graph_props_before = list(_cand_graph_props_before)
            _cand_graph_props_after = list(_cand_graph_props_after)
        else:
            # Go through candidates one by one in a single process
            # Initialize lists to track different metrics during pruning iteration
            _candidate_scores, _candidate_models, _cand_graph_props_before, _cand_graph_props_after = [], [], [], []
            for candidate in candidates:
                # Get model performance for removing candidate
                score, cand_model, props_before, props_after = \
                    self._evaluate_candidate_performance(model, candidate, x_train, y_train, x_test, y_test)
                # Collect score, model and properties of candiate
                _candidate_scores.append(score)
                _candidate_models.append(cand_model)
                _cand_graph_props_before.append(props_before)
                _cand_graph_props_after.append(props_after)

        # Store the candidate properties in the history object
        # TODO maybe add candidate so dictonary can be further nested by candidates in iteration here
        self._update_history_during_prune_iter(_candidate_scores,
                                               candidates,
                                               _cand_graph_props_before,
                                               _cand_graph_props_after)

        return _candidate_scores, _candidate_models, _cand_graph_props_after


    def _evaluate_candidate_performance(self, model, candidate, x_train, y_train, x_test, y_test):
        # Single candidate run (had to be broken down to this to enable parallelization)
        # Copy original model for candidate removal
        _model = copy.deepcopy(model)

        # Get info on candidate egde and graph before removal
        # TODO rethink what to store
        # TODO rethink if before history is necessary
        _graph = _model.reservoir_layer.weights
        _graph_props_before = self.graph_analyzer.extract_properties(graph=_graph)

        # Remove candidate edge from reservoir
        _model.remove_reservoir_edges(edges=[candidate])

        # Re-fit (retrain) pruned model
        _model.fit(x=x_train, y=y_train)

        # Evaluate pruned model regarding performance criterion
        _score = _model.evaluate(x=x_test, y=y_test, metrics=self.criterion)[0]

        # Extract graph properties after pruning
        # TODO rethink what to store, info regarding edge unneccesary as its removed
        # TODO think about how to nest dict here
        _graph = _model.reservoir_layer.weights
        _graph_props_after = self.graph_analyzer.extract_properties(graph=_graph)

        if not self.parallel:
            # Print candidate and score info if not parallelized
            # Format candidate tuple cleanly
            # TODO maybe make parameter whether this should be shown or not
            label = f"{int(candidate[0])}-{int(candidate[1])}" if isinstance(candidate, tuple) else int(candidate)
            print(f'Possible deletion of edge {label:<10} loss: {_score:.6f}  ({(self._curr_loss - _score) / self._curr_loss:+.3%})')

        # Return canidate score, model and properties
        return _score, _model, _graph_props_before, _graph_props_after

    def _shortest_path_pruning(self, model):
        # possible other pruning strategies (neglecting for now)
        pass

    ############## STOPPING CRITERIONS FUNCTIONS ##############

    def _patience_stopping(self):
        # Checks if the loss is at a minimum, considering also patience
        # Returns True if loss is not at minimum and we should continue pruning
        if len(self._curr_loss_history) < 2:
            # Just at the start of pruning, cannot really check for minimum
            return True

        if self._curr_loss_history[-2] > self._curr_loss_history[-1]:
            # Current loss is smaller than previous, continue pruning
            print(
                f'Loss decreased from {self._curr_loss_history[-2]:.6f} to {self._curr_loss_history[-1]:.6f} \nContinuing pruning ...'
            )
            self._patience_counter = 0
            return True

        else:
            # Current loss is larger than previous
            self._patience_counter += 1
            if self._patience_counter < self.patience:
                # Patience counter still below patience, contiue pruning
                print(
                    f'Loss increased, but {self._patience_counter} < {self.patience} Continuing pruning'
                )
                return True
            else:
                # TODO: we need to recover the model that had the best score!
                # Patience is reached, stop pruning
                print(
                    f'Loss increased for {self.patience} consecutive iterations. Terminating pruning'
                )
                return False

    def _min_num_nodes_stopping(self):
        # Checks if the number of nodes is above the minimum number of nodes
        # returns True if number of nodes is above minimum and we should continue pruning
        if self._curr_num_nodes > self.min_num_nodes:
            # TODO this logic doesn't make much sense, because reaching min_num_nodes should still continue pruning
            # TODO pruning another edge doesn't mean that a node will be removed
            print(
                f'Number of nodes {self._curr_num_nodes} is larger than minimum number of nodes {self.min_num_nodes}. Continuing pruning'
            )
            return True
        else:
            print(
                f'Number of nodes {self._curr_num_nodes} is smaller/equal minimum number of nodes {self.min_num_nodes}. Terminating pruning'
            )
            return False

    def _min_num_edges_stopping(self):
        # Checks if the number of edges is above the minimum number of edges
        # returns True if number of nodes is above minimum and we should continue pruning
        if self._curr_num_edges > self.min_num_edges:
            print(f'Number of edges {self._curr_num_edges} is larger than minimum number of edges {self.min_num_edges}. Continuing pruning')
            return True
        else:
            print(f'Number of edges {self._curr_num_edges} is smaller/equal minimum number of edges {self.min_num_edges}. Terminating pruning')
        return False

    ######## history helpers ########

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
    def _validate_init_params(
        self,
        candidate_fraction,
        pruning_criterion,
        stopping_criterion,
        min_num_nodes,
        patience,
        criterion,
        metrics,
        node_props_extractor,
        graph_props_extractor,
        return_best_model,
        graph_analyzer,
        node_analyzer,
        remove_isolated_nodes,
    ):
        # Validate candidate fraction
        if not isinstance(candidate_fraction, float):
            raise TypeError('candidate_fraction must be a float in (0, 1]')

        if candidate_fraction <= 0 or candidate_fraction > 1:
            raise ValueError('candidate_fraction must be a float in (0, 1]')

        # Validate pruning criterion
        if pruning_criterion not in self.PRUNING_CRITERION:
            raise NotImplementedError(
                f"Unknown strategy '{pruning_criterion}'. Available strategies: {list(self.PRUNING_CRITERION)}"
            )

        # Validate stopping criterion
        if not isinstance(stopping_criterion, list):
            raise TypeError('stopping_criterion must be a list')
        for sc in stopping_criterion:
            if sc not in self.STOPPING_CRITERION:
                raise NotImplementedError(
                    f"Unknown strategy '{sc}'. Available strategies: {list(self.STOPPING_CRITERION)}"
                )

        # Validate min num of nodes
        if not isinstance(min_num_nodes, int):
            raise TypeError('min_num_nodes must be an integer')
        if min_num_nodes <= 2:
            raise ValueError('min_num_nodes must be larger than 2')

        # Validate patience
        if patience is not None and not isinstance(patience, int):
            raise TypeError('patience must be an integer')

        # Validate criterion
        if not isinstance(criterion, str):
            raise TypeError('criterion must be a string')

        # Validate graph analyzer
        if graph_analyzer is not None and not isinstance(graph_analyzer, GraphAnalyzer):
            raise TypeError('graph_analyzer must be an instance of GraphAnalyzer')

        # Validate node analyzer
        if node_analyzer is not None and not isinstance(node_analyzer, NodeAnalyzer):
            raise TypeError('node_analyzer must be an instance of NodeAnalyzer')

        # Validate remove isolated nodes
        if not isinstance(remove_isolated_nodes, bool):
            raise TypeError('remove_isolated_nodes must be a boolean')

    def _validate_pruning_params(self, model, data_train, data_val):

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
        #min_num_nodes=46,
        #stopping_criterion=['patience'],
        stopping_criterion=['min_edges'],
        #patience=2,
        min_num_edges=0,
        candidate_fraction=0.9,
        remove_isolated_nodes=True,
        metrics=["mse"],
        parallel=True
    )

    model_pruned, history = pruner.prune(
        model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
    )
