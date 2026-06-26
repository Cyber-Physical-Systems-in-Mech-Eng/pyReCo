import numpy as np
import networkx as nx
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm
from pyreco.custom_models import RC
from pyreco.edge_selector import EdgeSelector
from typing import Union
import copy
from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.node_analyzer import NodeAnalyzer
from pyreco.edge_analyzer import EdgeAnalyzer, available_extractors
from pyreco.metrics import available_metrics


def _evaluate_candidate_standalone(model, candidate, x_train, y_train, x_test, y_test,
                                   criterion, graph_analyzer):
    """
    Evaluate a single candidate edge removal, independent of 'EdgePruner'.
    Standalone (module-level) counterpart to
    'EdgePruner._evaluate_candidate_performance', used for parallel execution so that
    'joblib' workers do not need to pickle the whole 'EdgePruner' instance
    (whose history grows over the course of pruning) for every candidate.

    Parameters
    ----------
    model : RC
        Current (not yet pruned) reservoir computer model.
    candidate : tuple
        Edge (u, v) to tentatively remove.
    x_train : np.ndarray
        Training input data.
    y_train : np.ndarray
        Training target data.
    x_test : np.ndarray
        Validation input data.
    y_test : np.ndarray
        Validation target data.
    criterion : str
        Performance metric used to score the candidate.
    graph_analyzer : GraphAnalyzer
        Analyzer used to extract graph-level properties after pruning.

    Returns
    -------
    score : float
        Loss of the refit model with 'candidate' removed, evaluated on
        (x_test, y_test) using 'criterion'.
    model : RC
        Deep copy of 'model' with 'candidate' removed and refit.
    graph_props_after : dict
        Graph-level properties of the reservoir after removing 'candidate'.
    """
    _model = copy.deepcopy(model)
    _model.remove_reservoir_edges(edges=[candidate])
    _model.fit(x=x_train, y=y_train)
    _score = _model.evaluate(x=x_test, y=y_test, metrics=criterion)[0]
    _graph = _model.reservoir_layer.weights
    _graph_props_after = graph_analyzer.extract_properties(graph=_graph)
    return _score, _model, _graph_props_after


class EdgePruner:
    # implements a pruning object for pyreco objects.

    PRUNING_CRITERION = {
        'performance': '_performance_pruning',
        'structure': '_structural_pruning'
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
        performance_criterion: str = 'mse',
        structural_criterion: str = 'betweenness',
        metrics: Union[list, str] = ['mse'],
        return_best_model: bool = True,
        graph_analyzer: GraphAnalyzer = None,
        node_analyzer: NodeAnalyzer = None,
        edge_analyzer: EdgeAnalyzer = None,
        remove_isolated_nodes: bool = False,
        directed: bool = True,
        parallel: bool = False,

    ):
        """
        Initializer for the edge pruning class.

        Parameters
        ----------
        edge_selection_strat : str, optional
            Strategy used by :class:`EdgeSelector` to propose candidate edges
            for pruning during every iteration.
            Default is "random_uniform_wo_repl".
        candidate_fraction : float, optional
            Number of randomly chosen reservoir edges during every pruning
            iteration that is a candidate for pruning. Refers to the
            fraction of edges w.r.t. current number of edges during pruning
            iteration. Must be in (0, 1]. Default is 0.1.
        pruning_criterion : str, optional
            The criterion used to score and select among candidate edges.
            Must be a key of ``PRUNING_CRITERION``. Default is "performance".
        stopping_criterion : list of str, optional
            One or more criteria used to decide when to stop pruning. Pruning
            is stopped as soon as one criteria is fullfilled. Each
            entry must be a key of ``STOPPING_CRITERION``.
            Default is ["patience"].
        min_num_nodes : int, optional
            Stop pruning when arriving at this number of nodes. Must be
            larger than 2. Default is 3.
        min_num_edges : int, optional
            Stop pruning when arriving at this number of edges. Default is 2.
        patience : int, optional
            We allow a patience, i.e. to keep pruning after we reached a (local)
            minimum of the test set loss. Default is 0.
        performance_criterion : str, optional
            Loss metric used to evaluate model performance, both for
            steering ``pruning_criterion='performance'`` and for tracking
            real loss (patience, history, ``return_best_model``) regardless
            of which ``pruning_criterion`` is active. Must be a key of
            :func:`pyreco.metrics.available_metrics`. Default is "mse".
        structural_criterion : str, optional
            Edge property used to score candidates when
            ``pruning_criterion='structure'``. Must be a key of
            :func:`pyreco.edge_analyzer.available_extractors` (e.g.
            "betweenness", "source_out_degree"). Unused for other pruning
            criteria. Default is "betweenness" Still needs to be implemented.
        metrics : list or str, optional
            Additional metrics to track throughout the pruning history,
            without influencing pruning decisions. Each entry must be a key
            of :func:`pyreco.metrics.available_metrics`. Default is ["mse"].
        return_best_model : bool, optional
            Whether to return the model with the lowest recorded loss
            instead of the last model produced before stopping. Default is
            True.
        graph_analyzer : GraphAnalyzer, optional
            Analyzer used to extract graph-level properties. A default
            instance is created if not provided.
        node_analyzer : NodeAnalyzer, optional
            Analyzer used to extract node-level properties. A default
            instance is created if not provided.
        edge_analyzer : EdgeAnalyzer, optional
            Analyzer used to extract edge-level properties. A default
            instance is created if not provided.
        remove_isolated_nodes : bool, optional
            Whether to remove isolated nodes during pruning. Default is
            False.
        directed : bool, optional
            Whether the reservoir graph is treated as directed. Default is
            True.
        parallel : bool, optional
            Whether to parallelize candidate evaluation across CPU cores
            using ``joblib``. Default is False.

        Raises
        ------
        TypeError
            If any parameter has an invalid type.
        ValueError
            If any parameter has an invalid value.
        NotImplementedError
            If ``pruning_criterion`` or an entry of ``stopping_criterion`` is
            not a recognized strategy.
        """

        # Sanity checks for the input parameter types and values
        self._validate_init_params(
            edge_selection_strat,
            candidate_fraction,
            pruning_criterion,
            stopping_criterion,
            min_num_nodes,
            min_num_edges,
            patience,
            performance_criterion,
            structural_criterion,
            metrics,
            return_best_model,
            graph_analyzer,
            node_analyzer,
            edge_analyzer,
            remove_isolated_nodes,
            directed,
            parallel,
            )

        # If not given create analyzer classes for properties to store
        if graph_analyzer is None:
            graph_analyzer = GraphAnalyzer()
        if node_analyzer is None:
            node_analyzer = NodeAnalyzer()
        if edge_analyzer is None:
            edge_analyzer = EdgeAnalyzer()

        # Assigning the parameters to instance variables
        # Parameters for pruning criterion
        self.performance_criterion = performance_criterion
        self.structural_criterion = structural_criterion
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
        self.edge_analyzer = edge_analyzer
        # Parameter bools for extra functionalities
        self.return_best_model = return_best_model
        self.remove_isolated_nodes = remove_isolated_nodes
        self.parallel = parallel

        # Initialize history dict to store the history of the pruning process in a
        #  nested dictionary
        self.history = {}

        # Initialize attributes that will be used during pruning (and changed during
        #  the process)
        # Needs to be attributes as the history updates depend on them
        self._curr_model = None
        self._curr_loss = None
        self._curr_num_nodes = None
        self._curr_num_edges = None
        self._curr_loss_history = []
        self._best_loss = None
        self._idx_prune = None
        self._patience_counter = 0
        self._curr_metrics = None

    def prune(self, model: RC, data_train: tuple, data_val: tuple):
        """
        Prune a given model by iteratively removing edges.

        Parameters
        ----------
        model : RC
            Reservoir computer model to prune.
        data_train : tuple
            Tuple (x_train, y_train) of training data.
        data_val : tuple
            Tuple (x_val, y_val) of validation data, used to score candidates
            and to evaluate the stopping criterion.

        Returns
        -------
        model : RC
            Pruned model (refit on ``data_train``). If
            ``return_best_model`` is True, this is the model with the lowest
            recorded loss during pruning, otherwise the last model produced
            before stopping.
        history : dict
            Nested dictionary recording, per pruning iteration, the model
            state before and after pruning, the evaluated candidates and
            their scores, and any nodes removed as a side effect.
        """

        # Sanity checks for the input parameter types and values
        self._validate_pruning_params(model, data_train, data_val)

        # Obtain training and testing data
        x_test, y_test = data_val[0], data_val[1]
        x_train, y_train = data_train[0], data_train[1]

        # Assigning the parameters to instance variables that can not be set
        #   in the initializer, as they depend on the model and data
        self._curr_num_nodes = model.reservoir_layer.nodes

        _graph = model.reservoir_layer.weights
        if isinstance(_graph, nx.Graph):
            edge_indices = list(_graph.edges())
        elif isinstance(_graph, np.ndarray):
            rows, cols = np.where(_graph != 0)  # where entries are not zero
            edge_indices = list(zip(rows, cols))
            if not self.directed:
                edge_indices = [(r, c) for r, c in edge_indices if r < c]

        self._curr_num_edges = len(edge_indices)

        # Initialize the quantities that affect the stop condition
        self._curr_loss = model.evaluate(x=x_test, y=y_test,
                                         metrics=self.performance_criterion)[0]
        self._curr_loss_history = [self._curr_loss]

        # Initialize quantities that we track for the pruning history
        # These do not affect the pruning process
        self._curr_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)

        # Storing all pruned models during the pruning iteration
        # Allows to recover models from previous iterations, e.g. when the best model
        #   is not the last one in the iteration (positive patience value)
        _pruned_models = [copy.deepcopy(model)]
        _pruned_models_losses = [self._curr_loss]

        # Initialize the pruning iterator
        self._iter_count = 0

        while True:

            # min_edges checked here (pre-pruning) so we avoid evaluating candidates
            #   when we already know we would violate the constraint
            if 'min_edges' in self.stopping_criterion:
                if not self._min_num_edges_stopping():
                    break

            # Store initial model and properties in history
            self._history_update_before_pruning(model, _graph)

            print(f'Currently at pruning iteration {self._iter_count} ...')

            print(
                f'Current reservoir size: {self._curr_num_nodes} | '
                f'Current loss: {self._curr_loss:.8f}'
            )

            # Get candidates for pruning
            _curr_candidates = self._get_candidates(_graph)

            # Apply chosen pruning strategy on candidates to get scores for candidates
            _candidate_scores, _candidate_models, _cand_graph_props_after = \
                self._apply_pruning_strategy(model, _curr_candidates, x_train, y_train,
                                             x_test, y_test)

            # Store candidates and properties evaluated during iteration in history
            self._history_update_candidate_iteration(_graph, _curr_candidates,
                                                     _candidate_scores,
                                                     _cand_graph_props_after)

            # Out of all candidates select candidate with best score (the one to prune)
            idx_prune, pruned_candidate = self._get_best_candidate(_curr_candidates,
                                                                   _candidate_scores)
            self._curr_idx_prune = idx_prune

            # Get model properties of selected candidate and update the termination
            #   relevant quantities
            curr_model, curr_loss, curr_num_nodes, curr_num_edges = \
                self._get_best_candidate_model_properties(idx_prune, _candidate_models,
                                                          x_test, y_test)
            # TODO rethink if we should do the setting of these props above already and
            #   rename function so we can use it more
            #   and then also do current amount of edges
            self._curr_model = curr_model
            self._curr_loss = curr_loss
            self._curr_num_nodes = curr_num_nodes
            self._curr_num_edges = curr_num_edges
            self._curr_loss_history.append(self._curr_loss)

            # Check for isolated nodes and remove if no effect on performance
            removed_nodes = {}
            if self.remove_isolated_nodes:
                isolated_nodes = self._get_isolated_nodes(self._curr_model)
                self._curr_model, removed_nodes = self._remove_isolated_nodes(
                                                                isolated_nodes,
                                                                self._curr_model,
                                                                x_train,
                                                                y_train,
                                                                x_test,
                                                                y_test)
                self._curr_num_nodes = self._curr_model.reservoir_layer.nodes
                # Isolated readout node removal may have updated self._curr_loss above
                #   (see _remove_isolated_nodes), keep appended history entry
                #   in sync so patience/_curr_loss_history represents true values
                self._curr_loss_history[-1] = self._curr_loss

            # Check stopping criterion on to be pruned candidate model properties
            # If termination criteria would be violated by pruning candidate we stop
            #   pruning
            if self._check_stopping_criterion():
                # Exit pruning loop if stopping criterion condition is met
                # Remove partial history entry for this iteration before exiting to
                #   return clean history
                del self.history[self._iter_count]
                break

            if isinstance(pruned_candidate, tuple):
                # Clean print of pruned_candidate
                pruned_candidate = (int(pruned_candidate[0]), int(pruned_candidate[1]))
            print(f'Pruning candidate {pruned_candidate}, '
                  f'resulting in loss {self._curr_loss:.6f}')
            print(
                f'Loss improvement to previous iteration by {(
                    (self._curr_loss_history[-2]-self._curr_loss) /
                    self._curr_loss_history[-2]):+.3%}\n'
            )

            # Prune edge that gives us the least performance drop
            # As we have already pruned edge and stored the model, we only need to
            #   update the model
            # Get it from self._curr_model to not discard isolated-node removal
            # model = _candidate_models[idx_prune]
            model = self._curr_model
            _graph = model.reservoir_layer.weights

            # Store the model and loss for later use
            _pruned_models.append(copy.deepcopy(model))
            _pruned_models_losses.append(self._curr_loss)

            # Compute things that are required for history, but not for
            #   pruning loop termination criteria
            self._curr_metrics = model.evaluate(
                x=x_test, y=y_test, metrics=self.metrics
            )

            # Store important data after pruning in history
            self._history_update_after_pruning(model, _graph, idx_prune,
                                               _curr_candidates, removed_nodes)

            # Update iteration counter
            self._iter_count += 1

        # In case we have a non-zero patience, we might want to return the best model
        #   instead of the last one (i.e. when a positive patience value was given)
        if self.return_best_model:
            idx_best = np.argmin(_pruned_models_losses)
            model = copy.deepcopy(_pruned_models[idx_best])
            print(f'\nReturning model form iteration {idx_best-1} as the best')

        # Fit the final model, and evaluate it
        model.fit(x=x_train, y=y_train)
        final_loss = model.evaluate(x=x_test, y=y_test,
                                    metrics=self.performance_criterion)[0]
        final_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)
        print(
            f'\nInitial loss: {self._curr_loss_history[0]:.6f}, '
            f'loss after pruning: {final_loss:.6f}'
        )
        print(f'Final model has {model.reservoir_layer.nodes} nodes')
        print(f'Final model loss {self.performance_criterion}: {final_loss:.6f}')
        print(f'Final model metrics ({self.metrics}): {final_metrics}')
        return model, self.history

    #   ######## PRUNING STEPS FUNCTIONS ##########

    def _history_update_before_pruning(self, model, graph):
        """
        Record the state of the model at the start of a pruning iteration.

        Parameters
        ----------
        model : RC
            Model as it stands before this iteration's candidate edges
            are evaluated.
        graph : nx.Graph or np.ndarray
            Reservoir's adjacency representation corresponding to
            ``model``.

        Returns
        -------
        None
            Updates ``self.history[self._iter_count]['starting_model']`` in
            place.
        """

        # Get graph properties of current model
        graph_props = self.graph_analyzer.extract_properties(graph=graph)
        # Store the properties of current model
        self.history[self._iter_count] = {
            'starting_model': {
                'weights': sp.csr_matrix(graph) if isinstance(graph, np.ndarray)
                else nx.to_scipy_sparse_array(graph),
                'input_nodes': list(model.reservoir_layer.input_receiving_nodes),
                'readout_nodes': list(model.readout_layer.readout_nodes),
                'loss': self._curr_loss,
                'num_nodes': self._curr_num_nodes,
                'num_edges': self._curr_num_edges,
                'metrics': self._curr_metrics,
                'graph_props': graph_props,
            }
        }

    def _get_candidates(self, graph):
        """
        Propose candidate edges for pruning in the current iteration.

        Parameters
        ----------
        graph : nx.Graph or np.ndarray
            Reservoir's current adjacency representation.

        Returns
        -------
        list of tuple
            Edge indices (u, v) proposed as pruning candidates, selected via
            ``self.edge_selection_strat`` and ``self.candidate_fraction``.
        """

        selector = EdgeSelector(
            graph=graph,
            strategy=self.edge_selection_strat,
            directed=self.directed
            )

        # Obtain edges for pruning
        _curr_candidates = selector.select_edges(fraction=self.candidate_fraction)

        print(
            f'Proposing {selector.num_select_edges}/{selector.num_total_edges} '
            'edges for pruning ...'
            )
        return _curr_candidates

    def _apply_pruning_strategy(self, model, candidates, x_train, y_train, x_test,
                                y_test):
        """
        Dispatch candidate scoring to the configured pruning criterion.

        Looks up ``self.pruning_criterion`` in ``PRUNING_CRITERION`` and
        calls the corresponding method (e.g. ``_performance_pruning``).

        Parameters
        ----------
        model : RC
            Current (not yet pruned) model.
        candidates : list of tuple
            Edge indices (u, v) to evaluate as pruning candidates.
        x_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Training target data.
        x_test : np.ndarray
            Validation input data.
        y_test : np.ndarray
            Validation target data.

        Returns
        -------
        candidate_scores : list of float
            Score for each candidate (lower is better, i.e. a better pruning
            choice).
        candidate_models : list of RC
            Model resulting from removing each candidate edge.
        cand_graph_props_after : list of dict
            Graph-level properties of the reservoir after removing each
            candidate.
        """
        method = getattr(self, self.PRUNING_CRITERION[self.pruning_criterion])
        return method(model, candidates, x_train, y_train, x_test, y_test)

    def _history_update_candidate_iteration(self, graph, candidates, candidate_scores,
                                            cand_graph_props_after):
        """
        Record all evaluated candidates and their scores for this iteration.

        Parameters
        ----------
        graph : nx.Graph or np.ndarray
            Reservoir's weight matrix before pruning, used to
            extract edge-level properties of the candidates.
        candidates : list of tuple
            Edge indices (u, v) that were evaluated as pruning candidates.
        candidate_scores : list of float
            Score for each candidate, in the same order as ``candidates``.
        cand_graph_props_after : list of dict
            Graph-level properties of the reservoir after removing each
            candidate, in the same order as ``candidates``.

        Returns
        -------
        None
            Updates ``self.history[self._iter_count]['candidates']`` in
            place.
        """
        # Get scores, edge properties and potential graph properties of candidates
        cand_edge_props = self.edge_analyzer.extract_properties_batch(graph, candidates)

        self.history[self._iter_count]['candidates'] = {
            (int(c[0]), int(c[1])): {
                'score': score,
                'edge_props': edge_props,
                'graph_props_after': graph_props,
            }
            for c, score, edge_props, graph_props in zip(
                candidates, candidate_scores, cand_edge_props, cand_graph_props_after
            )
        }

    def _get_best_candidate(self, candidates, candidate_scores):
        """
        Select the candidate edge with the best (lowest) score.

        Parameters
        ----------
        candidates : list of tuple
            Edge indices (u, v) that were evaluated as pruning candidates.
        candidate_scores : list of float
            Score for each candidate, in the same order as ``candidates``.

        Returns
        -------
        idx_prune : int
            Index, into ``candidates`` and ``candidate_scores``, of the
            selected candidate.
        pruned_candidate : tuple
            Edge (u, v)  to be pruned.
        """
        idx_prune = np.argmin(candidate_scores)
        pruned_candidate = candidates[idx_prune]  # Just for history logging

        return idx_prune, pruned_candidate

    def _get_best_candidate_model_properties(self, candidate_idx, candidate_models,
                                             x_test, y_test):
        """
        Extract the model and bookkeeping quantities for the selected candidate.

        The loss is computed here directly from the selected model, rather
        than reused from the candidate's selection score. The two coincide
        for ``pruning_criterion='performance'`` (the score is the loss), but
        not in general (e.g. a structural criterion's score is a graph
        property, not a loss). Computing it explicitly keeps
        ``self._curr_loss``/patience/history meaningful regardless of which
        criterion picked the candidate.

        Parameters
        ----------
        candidate_idx : int
            Index of the selected candidate, as returned by
            ``_get_best_candidate``.
        candidate_models : list of RC
            Model resulting from removing each evaluated candidate edge.
        x_test : np.ndarray
            Validation input data.
        y_test : np.ndarray
            Validation target data.

        Returns
        -------
        curr_model : RC
            Model resulting from removing the selected candidate edge.
        curr_loss : float
            Loss of ``curr_model`` on (x_test, y_test), evaluated using
            ``self.performance_criterion``.
        curr_num_nodes : int
            Number of reservoir nodes in ``curr_model``.
        curr_num_edges : int
            Number of reservoir edges in ``curr_model``.
        """
        curr_model = candidate_models[candidate_idx]
        curr_loss = curr_model.evaluate(x=x_test, y=y_test,
                                        metrics=self.performance_criterion)[0]
        curr_num_nodes = curr_model.reservoir_layer.nodes
        curr_graph = curr_model.reservoir_layer.weights

        if isinstance(curr_graph, nx.Graph):
            edge_indices = list(curr_graph.edges())
        elif isinstance(curr_graph, np.ndarray):
            rows, cols = np.where(curr_graph != 0)  # Where entries are not zero
            edge_indices = list(zip(rows, cols))
            if not self.directed:
                edge_indices = [(r, c) for r, c in edge_indices if r < c]

        curr_num_edges = len(edge_indices)

        return curr_model, curr_loss, curr_num_nodes, curr_num_edges

    def _get_isolated_nodes(self, model):
        """
        Find nodes that are isolated (degree 0) in the reservoir graph.

        Parameters
        ----------
        model : RC
            Model whose reservoir graph is inspected.

        Returns
        -------
        list of dict
            One entry per isolated node, each with keys ``'id'`` (node id),
            ``'is_input'`` (whether the node receives input), and
            ``'is_readout'`` (whether the node feeds the readout layer).
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
                if np.count_nonzero(graph[node, :]) == 0 and \
                                    np.count_nonzero(graph[:, node]) == 0:
                    isolated_nodes.append({
                        'id': node,
                        'is_input': node in input_nodes,
                        'is_readout': node in readout_nodes,
                    })

        return isolated_nodes

    def _remove_isolated_nodes(self, isolated_nodes, model,
                               x_train, y_train, x_test, y_test):
        """
        Remove isolated nodes from the model where safe to do so.

        Non-input, non-readout isolated nodes are removed unconditionally.
        Isolated readout nodes are removed only if doing so does not
        increase the loss; input-receiving isolated nodes are never removed.

        Parameters
        ----------
        isolated_nodes : list of dict
            Isolated nodes as returned by ``_get_isolated_nodes``.
        model : RC
            Model to remove isolated nodes from.
        x_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Training target data.
        x_test : np.ndarray
            Validation input data.
        y_test : np.ndarray
            Validation target data.

        Returns
        -------
        model : RC
            Model with eligible isolated nodes removed.
        removed : dict
            Mapping of removed node id to a dict with keys ``'loss_after'``
            (loss after removal, or None for unconditionally removed nodes)
            and ``'is_readout'``.
        """
        # Dict to store isolated nodes
        removed = {}
        # Remove isolated nodes that are neither input-receiving nor readout nodes
        fully_isolated_node_ids = [
            n['id'] for n in isolated_nodes
            if not n['is_input'] and not n['is_readout']
        ]
        if fully_isolated_node_ids:
            print(f'Removing {len(fully_isolated_node_ids)} isolated non-input/readout '
                  f'nodes: {fully_isolated_node_ids}')
            model.remove_reservoir_nodes(nodes=fully_isolated_node_ids)
            # Store removed node in dict for tracking
            for node_id in fully_isolated_node_ids:
                removed[node_id] = {'loss_after': None, 'is_readout': False}

        # Remove isolated readout nodes if removal doesn't affect performance negatively
        isolated_readout_nodes = [
            n for n in isolated_nodes
            if not n['is_input'] and n['is_readout']
        ]

        # Track original IDs of all removed nodes to adjust indices after renumbering
        # Each removal shifts down the indices of all higher-numbered nodes by 1
        removed_original_ids = list(fully_isolated_node_ids)
        # Check effect on performance performance
        for node in isolated_readout_nodes:
            # Compute how many previously removed nodes had a smaller original ID,
            #   since each such removal shifted this node's current index down by 1
            adjusted_id = node['id'] - sum(1 for r in removed_original_ids
                                           if r < node['id'])
            _model = copy.deepcopy(model)
            _model.remove_reservoir_nodes(nodes=[adjusted_id])
            _model.fit(x=x_train, y=y_train)
            _score = _model.evaluate(x=x_test, y=y_test,
                                     metrics=self.performance_criterion)[0]

            if _score <= self._curr_loss:
                print(f'Removing isolated readout node {node["id"]:>3}: loss '
                      f'{self._curr_loss:.6f} -> {_score:.6f}')
                # Note that node has been removed
                removed_original_ids.append(node['id'])
                # Store removed node in dict for tracking
                removed[node['id']] = {'loss_after': _score, 'is_readout': True}
                model = _model
                self._curr_loss = _score
            else:
                print(f'Keeping isolated readout node  {node["id"]:>3}: removal would '
                      f'increase loss {self._curr_loss:.6f} -> {_score:.6f}')

        return model, removed

    def _check_stopping_criterion(self):
        """
        Check whether any configured stopping criterion has been met.

        Iterates over ``self.stopping_criterion`` and calls the
        corresponding method (e.g. ``_patience_stopping``) for each.
        ``'min_edges'`` is skipped here since it is checked separately, at
        the top of the pruning loop, before candidates are evaluated.

        Returns
        -------
        bool
            True if pruning should stop (any criterion's "continue" check
            returned False), False if pruning should continue.
        """
        for criterion in self.stopping_criterion:
            if criterion == 'min_edges':
                # Handled at the top of the pruning loop (pre-pruning check)
                continue
            method = getattr(self, self.STOPPING_CRITERION[criterion])
            if not method():
                # Criterion is met
                return True
        # Criterion is not met
        return False

    def _history_update_after_pruning(self, model, graph, idx_prune,
                                      candidates, removed_nodes):
        """
        Record the outcome of a pruning iteration.

        Parameters
        ----------
        model : RC
            Model after the selected candidate edge has been pruned.
        graph : nx.Graph or np.ndarray
            ``model``'s reservoir adjacency representation.
        idx_prune : int
            Index, into ``candidates``, of the edge that was pruned.
        candidates : list of tuple
            Edge indices (u, v) that were evaluated this iteration.
        removed_nodes : dict
            Isolated nodes removed this iteration, as returned by
            ``_remove_isolated_nodes``.

        Returns
        -------
        None
            Updates ``self.history[self._iter_count]`` in place with the
            ``'winner'``, ``'removed_nodes'``, and ``'final_model'`` entries.
        """
        # Get final graph properties and winner
        graph_props = self.graph_analyzer.extract_properties(graph=graph)
        winner = candidates[idx_prune]
        # Store winner, removed nodes and properties of winner model
        self.history[self._iter_count]['winner'] = (int(winner[0]), int(winner[1]))
        self.history[self._iter_count]['removed_nodes'] = removed_nodes
        self.history[self._iter_count]['final_model'] = {
            'weights': sp.csr_matrix(graph) if isinstance(graph, np.ndarray) else
            nx.to_scipy_sparse_array(graph),
            'input_nodes': list(model.reservoir_layer.input_receiving_nodes),
            'readout_nodes': list(model.readout_layer.readout_nodes),
            'loss': self._curr_loss,
            'num_nodes': self._curr_num_nodes,
            'num_edges': self._curr_num_edges,
            'metrics': self._curr_metrics,
            'graph_props': graph_props,
        }

    #   ######## MODEL TRAINING FUNCTIONS #########

    def _retrain_model(self, model, x_train, y_train):
        """
        Fit a model on training data.

        Parameters
        ----------
        model : RC
            Model to fit.
        x_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Training target data.

        Returns
        -------
        Whatever ``model.fit`` returns.
        """
        return model.fit(x=x_train, y=y_train)

    #   ######## PRUNING CRITERIONS FUNCTIONS #########

    def _performance_pruning(self, model, candidates, x_train, y_train, x_test, y_test):
        """
        Score candidates by the model's loss after removing and refitting.

        For each candidate edge, removes it from a copy of ``model``, refits
        the copy, and evaluates it on the validation data.
        Runs serially or in parallel (across CPU cores via
        ``joblib``) depending on ``self.parallel``.

        Parameters
        ----------
        model : RC
            Current (not yet pruned) model.
        candidates : list of tuple
            Edge indices (u, v) to evaluate as pruning candidates.
        x_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Training target data.
        x_test : np.ndarray
            Validation input data.
        y_test : np.ndarray
            Validation target data.

        Returns
        -------
        candidate_scores : list of float
            Loss of the refit model for each candidate, in the same order as
            ``candidates``.
        candidate_models : list of RC
            Refit model resulting from removing each candidate edge.
        cand_graph_props_after : list of dict
            Graph-level properties of the reservoir after removing each
            candidate.
        """

        if self.parallel:
            # Parallelizing performance evaluation of candidates
            # Kicking off parallelization of going through all candidates
            # tqdm shows process in in bar chart
            # n_jobs is number of jobs to run in parallel (-1 uses all CPU cores)
            # backend -  loky is default
            parallel = Parallel(n_jobs=-1, backend='loky')
            # tqdm shows process in bar chart
            # Generator returns results in order that they're given
            # Call seperate evaluation function that doesn't pass the whole self object
            #   (history gets larger with each iteration and therefore slows down when
            #   it's spawned across multiple CPUs)
            results = parallel(
                delayed(_evaluate_candidate_standalone)
                (model, c, x_train, y_train, x_test, y_test,
                 self.performance_criterion, self.graph_analyzer)
                for c in tqdm(candidates, desc='Evaluating candidates')
                )
            _candidate_scores, _candidate_models, _cand_graph_props_after = \
                zip(*results)
            _candidate_scores = list(_candidate_scores)
            _candidate_models = list(_candidate_models)
            _cand_graph_props_after = list(_cand_graph_props_after)
        else:
            # Go through candidates one by one in a single process
            # Initialize lists to track different metrics during pruning iteration
            _candidate_scores, _candidate_models, _cand_graph_props_after = [], [], []
            for candidate in candidates:
                # Get model performance for removing candidate
                score, cand_model, props_after = \
                    self._evaluate_candidate_performance(model, candidate,
                                                         x_train, y_train,
                                                         x_test, y_test)
                # Collect score, model and properties of candiate
                _candidate_scores.append(score)
                _candidate_models.append(cand_model)
                _cand_graph_props_after.append(props_after)

        return _candidate_scores, _candidate_models, _cand_graph_props_after

    def _evaluate_candidate_performance(self, model, candidate,
                                        x_train, y_train, x_test, y_test):
        """
        Evaluate a single candidate edge removal (serial path).

        Parameters
        ----------
        model : RC
            Current (not yet pruned) model.
        candidate : tuple
            Edge (u, v) to tentatively remove.
        x_train : np.ndarray
            Training input data.
        y_train : np.ndarray
            Training target data.
        x_test : np.ndarray
            Validation input data.
        y_test : np.ndarray
            Validation target data.

        Returns
        -------
        score : float
            Loss of the refit model with ``candidate`` removed, evaluated on
            (x_test, y_test) using ``self.performance_criterion``.
        model : RC
            Deep copy of ``model`` with ``candidate`` removed and refit.
        graph_props_after : dict
            Graph-level properties of the reservoir after removing
            ``candidate``.
        """
        # Single candidate run (had to be broken down to this to enable parallelization)
        # Copy original model for candidate removal
        _model = copy.deepcopy(model)

        # Get info on candidate egde and graph before removal
        _graph = _model.reservoir_layer.weights

        # Remove candidate edge from reservoir
        _model.remove_reservoir_edges(edges=[candidate])

        # Re-fit (retrain) pruned model
        _model.fit(x=x_train, y=y_train)

        # Evaluate pruned model regarding performance criterion
        _score = _model.evaluate(x=x_test, y=y_test,
                                 metrics=self.performance_criterion)[0]

        # Extract graph properties after pruning
        _graph = _model.reservoir_layer.weights
        _graph_props_after = self.graph_analyzer.extract_properties(graph=_graph)

        if not self.parallel:
            # Print candidate and score info if not parallelized
            # Format candidate tuple cleanly
            label = f"{int(candidate[0])}-{int(candidate[1])}" if \
                    isinstance(candidate, tuple) else int(candidate)
            print(f'Possible deletion of edge {label:<10} loss: {_score:.6f}  '
                  f'({(self._curr_loss - _score) / self._curr_loss:+.3%})')

        # Return canidate score, model and properties
        return _score, _model, _graph_props_after

    def _shortest_path_pruning(self, model):
        """
        Placeholder for a shortest-path-based pruning criterion.

        Not implemented yet.

        Parameters
        ----------
        model : RC
            The current (not yet pruned) model.
        """
        # possible other pruning strategies (neglecting for now)
        pass

    #   ############# STOPPING CRITERIONS FUNCTIONS ##############

    def _patience_stopping(self):
        """
        Check the patience-based stopping criterion.

        Patience counts iterations since the all-time best loss was
        achieved, not just since the last improvement over the previous
        iteration.

        Returns
        -------
        bool
            True if we should continue pruning, False to stop.
        """
        # Checks if the loss is at a minimum, considering also patience.
        # Patience counts steps since the all-time best loss was achieved,
        #   not just since the last improvement over the previous step
        # Returns True if we should continue pruning, False to stop.
        if len(self._curr_loss_history) < 1:
            return True

        current_loss = self._curr_loss_history[-1]

        if self._best_loss is None or current_loss <= self._best_loss:
            # New best loss — reset patience
            print(
                f'Loss improved to new best {current_loss:.6f}. '
                '\nContinuing pruning ...'
            )
            self._best_loss = current_loss
            self._patience_counter = 0
            return True
        else:
            # No improvement over best loss
            self._patience_counter += 1
            if self._patience_counter <= self.patience:
                print(
                    f'No improvement over best loss ({self._best_loss:.6f}), patience '
                    f'{self._patience_counter}/{self.patience}. '
                    '\nContinuing pruning ...'
                )
                return True
            else:
                print(
                    f'No improvement over best loss ({self._best_loss:.6f}) for '
                    f'{self.patience} consecutive iterations. \nTerminating pruning!'
                )
                return False

    def _min_num_nodes_stopping(self):
        """
        Check the minimum-number-of-nodes stopping criterion.

        Returns
        -------
        bool
            True if the number of nodes is above ``self.min_num_nodes`` and
            we should continue pruning, False to stop.
        """
        # Checks if the number of nodes is above the minimum number of nodes
        # Returns True if number of nodes is above minimum and we should continue
        #   pruning
        if self._curr_num_nodes >= self.min_num_nodes:
            # When reaching min_num_nodes we should still continue pruning
            #   pruning another edge doesn't mean that a node will be removed
            print(
                f'Number of nodes {self._curr_num_nodes} is larger than minimum number '
                f'of nodes {self.min_num_nodes}. \nContinuing pruning ...'
            )
            return True
        else:
            print(
                f'Number of nodes {self._curr_num_nodes} is smaller/equal minimum '
                f'number of nodes {self.min_num_nodes}. \nTerminating pruning!'
            )
            return False

    def _min_num_edges_stopping(self):
        """
        Check the minimum-number-of-edges stopping criterion.

        Unlike the other stopping criteria, this is a pre-pruning check
        called at the top of the pruning loop, before candidates are
        evaluated. This avoids scoring candidates we already know we cannot
        prune.

        Returns
        -------
        bool
            True if pruning one more edge would not violate
            ``self.min_num_edges`` and we should continue, False to stop.
        """
        # Pre-pruning check: called at the top of the while loop before candidate
        #   evaluation.
        # Stops if pruning one more edge would violate the minimum.
        if self._curr_num_edges <= self.min_num_edges:
            print(f'Number of edges {self._curr_num_edges} is minimum number of edges '
                  f'{self.min_num_edges}. \nTerminating pruning!')
            return False
        print(f'Number of edges {self._curr_num_edges} is larger than minimum number '
              f'of edges {self.min_num_edges}. \nContinuing pruning ...')
        return True

    #   ######## VALIDATION FUNCTIONS #########

    def _validate_init_params(
        self,
        edge_selection_strat,
        candidate_fraction,
        pruning_criterion,
        stopping_criterion,
        min_num_nodes,
        min_num_edges,
        patience,
        performance_criterion,
        structural_criterion,
        metrics,
        return_best_model,
        graph_analyzer,
        node_analyzer,
        edge_analyzer,
        remove_isolated_nodes,
        directed,
        parallel,
    ):
        """
        Validate the types and values of parameters passed to ``__init__``.

        Parameters
        ----------
        edge_selection_strat : str
            See ``__init__``.
        candidate_fraction : float
            See ``__init__``.
        pruning_criterion : str
            See ``__init__``.
        stopping_criterion : list of str
            See ``__init__``.
        min_num_nodes : int
            See ``__init__``.
        min_num_edges : int
            See ``__init__``.
        patience : int
            See ``__init__``.
        performance_criterion : str
            See ``__init__``.
        structural_criterion : str
            See ``__init__``.
        metrics : list or str
            See ``__init__``.
        return_best_model : bool
            See ``__init__``.
        graph_analyzer : GraphAnalyzer or None
            See ``__init__``.
        node_analyzer : NodeAnalyzer or None
            See ``__init__``.
        edge_analyzer : EdgeAnalyzer or None
            See ``__init__``.
        remove_isolated_nodes : bool
            See ``__init__``.
        directed : bool
            See ``__init__``.
        parallel : bool
            See ``__init__``.

        Raises
        ------
        TypeError
            If any parameter has an invalid type.
        ValueError
            If any parameter has an invalid value.
        NotImplementedError
            If ``pruning_criterion`` or an entry of ``stopping_criterion`` is
            not a recognized strategy.
        """
        # Validate edge selection strategy
        from pyreco.edge_selector import EdgeSelector
        if not isinstance(edge_selection_strat, str):
            raise TypeError('edge_selection_strat must be a string')
        if edge_selection_strat not in EdgeSelector.STRATEGIES:
            raise NotImplementedError(
                f"Unknown edge selection strategy '{edge_selection_strat}'. "
                f'Available strategies: {list(EdgeSelector.STRATEGIES)}'
            )

        # Validate candidate fraction
        if not isinstance(candidate_fraction, float):
            raise TypeError('candidate_fraction must be a float in (0, 1]')

        if candidate_fraction <= 0 or candidate_fraction > 1:
            raise ValueError('candidate_fraction must be a float in (0, 1]')

        # Validate pruning criterion
        if not isinstance(pruning_criterion, str):
            raise TypeError('pruning_criterion must be a string')
        if pruning_criterion not in self.PRUNING_CRITERION:
            raise NotImplementedError(
                f"Unknown strategy '{pruning_criterion}'. "
                f'Available strategies: {list(self.PRUNING_CRITERION)}'
            )
        # Structural pruning not imlemented yet
        if pruning_criterion == 'structure':
            raise NotImplementedError(
                'Structural pruning criterion is not implemented yet'
            )

        # Validate stopping criterion
        if not isinstance(stopping_criterion, list):
            raise TypeError('stopping_criterion must be a list')
        if len(stopping_criterion) == 0:
            raise ValueError(
                'stopping_criterion must contain at least one criterion, otherwise '
                'pruning has no way to stop and will run until it errors out on an '
                'edgeless graph'
            )
        for sc in stopping_criterion:
            if sc not in self.STOPPING_CRITERION:
                raise NotImplementedError(
                    f"Unknown strategy '{sc}'. "
                    f'Available strategies: {list(self.STOPPING_CRITERION)}'
                )

        # Validate min num of nodes
        if not isinstance(min_num_nodes, int):
            raise TypeError('min_num_nodes must be an integer')
        if min_num_nodes <= 2:
            raise ValueError('min_num_nodes must be larger than 2')

        # Validate min num of edges
        if not isinstance(min_num_edges, int):
            raise TypeError('min_num_edges must be an integer')
        if min_num_edges < 0:
            raise ValueError('min_num_edges must be larger than or equal to 0')

        # Validate patience
        if not isinstance(patience, int):
            raise TypeError('patience must be an integer')
        if patience < 0:
            raise ValueError('patience must be >= 0')

        # Validate performance criterion
        if not isinstance(performance_criterion, str):
            raise TypeError('performance_criterion must be a string')
        if performance_criterion not in available_metrics():
            raise ValueError(
                f"Unknown metric '{performance_criterion}'. "
                f'Available metrics: {available_metrics()}'
            )

        # Validate structural criterion
        if not isinstance(structural_criterion, str):
            raise TypeError('structural_criterion must be a string')
        if structural_criterion not in available_extractors():
            raise ValueError(
                f"Unknown structural criterion '{structural_criterion}'. "
                f'Available structural criteria: {list(available_extractors())}'
            )

        # Validate metrics
        if not isinstance(metrics, (list, str)):
            raise TypeError('metrics must be a list or a string')
        if isinstance(metrics, list) and not all(isinstance(m, str) for m in metrics):
            raise TypeError('metrics must be a list of strings')
        _metrics_list = metrics if isinstance(metrics, list) else [metrics]
        if not all(m in available_metrics() for m in _metrics_list):
            raise ValueError(
                f'Unknown metric in {metrics}. '
                f'Available metrics: {available_metrics()}'
            )

        # Validate return_best_model
        if not isinstance(return_best_model, bool):
            raise TypeError('return_best_model must be a boolean')

        # Validate graph analyzer
        if graph_analyzer is not None and not isinstance(graph_analyzer, GraphAnalyzer):
            raise TypeError('graph_analyzer must be an instance of GraphAnalyzer')

        # Validate node analyzer
        if node_analyzer is not None and not isinstance(node_analyzer, NodeAnalyzer):
            raise TypeError('node_analyzer must be an instance of NodeAnalyzer')

        # Validate edge analyzer
        if edge_analyzer is not None and not isinstance(edge_analyzer, EdgeAnalyzer):
            raise TypeError('edge_analyzer must be an instance of EdgeAnalyzer')

        # Validate remove isolated nodes
        if not isinstance(remove_isolated_nodes, bool):
            raise TypeError('remove_isolated_nodes must be a boolean')

        # Validate directed
        if not isinstance(directed, bool):
            raise TypeError('directed must be a boolean')

        # Validate parallel
        if not isinstance(parallel, bool):
            raise TypeError('parallel must be a boolean')

    def _validate_pruning_params(self, model, data_train, data_val):
        """
        Validate the types and values of parameters passed to ``prune``.

        Parameters
        ----------
        model : RC
            See ``prune``.
        data_train : tuple
            See ``prune``.
        data_val : tuple
            See ``prune``.

        Raises
        ------
        TypeError
            If ``model`` is not an ``RC`` instance, ``data_train``/
            ``data_val`` are not tuples, or their elements are not lists or
            numpy arrays.
        ValueError
            If ``data_train``/``data_val`` do not have exactly 2 elements,
            or if the inputs and targets within either do not have matching
            lengths.
        """

        if not isinstance(model, RC):
            raise TypeError('model must be an instance of RC')

        if not isinstance(data_train, tuple) or not isinstance(data_val, tuple):
            raise TypeError('data_train and data_val must be tuples')

        if len(data_train) != 2 or len(data_val) != 2:
            raise ValueError('data_train and data_val must have 2 elements each')

        for idx, elem in enumerate(data_train):
            if not isinstance(elem, list):
                if not isinstance(elem, np.ndarray):
                    raise TypeError(f'data_train[{idx}] must be a list or numpy array')

        for idx, elem in enumerate(data_val):
            if not isinstance(elem, list):
                if not isinstance(elem, np.ndarray):
                    raise TypeError(f'data_val[{idx}] must be a list or numpy array')

        if len(data_train[0]) != len(data_train[1]):
            raise ValueError(
                'data_train[0] and data_train[1] must have the same length, '
                'i.e. same number of samples'
            )

        if len(data_val[0]) != len(data_val[1]):
            raise ValueError(
                'data_val[0] and data_val[1] must have the same length, '
                'i.e. same number of samples'
            )


if __name__ == "__main__":
    # Test the pruning

    from pyreco.utils_data import sequence_to_sequence as seq_2_seq
    from pyreco.custom_models import RC as RC
    from pyreco.layers import InputLayer, ReadoutLayer
    from pyreco.layers import RandomReservoirLayer
    from pyreco.optimizers import RidgeSK

    # Get some data
    X_train, X_test, y_train, y_test = seq_2_seq(
        name='sine_pred', n_batch=20, n_states=2, n_time=150
    )

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]

    # Build a classical RC
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(
        RandomReservoirLayer(
            nodes=50,
            density=0.1,
            activation='tanh',
            leakage_rate=0.1,
            fraction_input=0.5,
        ),
    )
    model.add(ReadoutLayer(output_shape, fraction_out=0.9))

    # Compile the model
    optim = RidgeSK(alpha=0.5)
    model.compile(
        optimizer=optim,
        metrics=['mean_squared_error'],
    )

    # Train the model
    model.fit(X_train, y_train)

    print(f'Ccore: \t\t\t{model.evaluate(x=X_test, y=y_test)[0]:.4f}')

    # Prune the model
    pruner = EdgePruner(
        # min_num_nodes=46,
        # stopping_criterion=['patience'],
        # stopping_criterion=['min_edges', 'patience'],
        stopping_criterion=['min_edges'],
        patience=2,
        min_num_edges=0,
        candidate_fraction=0.9,
        remove_isolated_nodes=True,
        metrics=["mse"],
        parallel=True
    )

    model_pruned, history = pruner.prune(
        model=model, data_train=(X_train, y_train), data_val=(X_test, y_test)
    )
