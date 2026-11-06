import warnings

import numpy as np
import networkx as nx
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm
from pyreco.custom_models import RC
from pyreco.edge_selector import EdgeSelector
from pyreco.utils_random import check_seed
from typing import NamedTuple, Optional, Union
import copy
from pyreco.edge_analyzer import available_extractors
from pyreco.features import (
    FeatureContext,
    evaluate_edge_features,
    evaluate_network_features,
    evaluate_node_features,
    resolve_features,
)
from pyreco.metrics import available_metrics, is_loss


def _remove_edge(model, candidate, target_spec_rad=None):
    """
    Remove one edge from the reservoir of ``model`` and, if ``target_spec_rad`` is given, rescale the
    remaining weights to that spectral radius. A reservoir without a directed cycle has spectral radius 0
    and cannot be rescaled: it is left as it is.
    """
    model.remove_reservoir_edges(edges=[candidate])
    layer = model.reservoir_layer
    if target_spec_rad is not None and layer.spec_rad > 0:
        try:
            layer.set_spec_rad(target_spec_rad)
        except ValueError:  # the radius is numerically zero
            pass


def _check_rescale_spec_rad(value):
    """Raise for a ``rescale_spec_rad`` that is not None, "initial" or a positive finite number."""
    if value is None or (isinstance(value, str) and value == 'initial'):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f'rescale_spec_rad must be None, "initial" or a number, got {value!r}')
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'rescale_spec_rad must be positive and finite, got {value!r}')


class _CandidateResult(NamedTuple):
    """Outcome of removing one edge: the refit model with its validation loss, and what is recorded."""

    score: float  # loss on the validation data
    model: RC  # copy of the model with the edge removed, refit
    train_score: Optional[float] = None  # loss on the training data, only if recorded
    network_features: Optional[dict] = None  # name -> float, features of the candidate model, only if recorded


def _evaluate_candidate_standalone(model, candidate, x_train, y_train, x_test, y_test,
                                   criterion, target_spec_rad=None, record=False,
                                   network_features=None):
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
    target_spec_rad : float, optional
        Spectral radius that the reservoir is rescaled to after removing the edge. None (default) does
        not rescale.
    record : bool, optional
        Also compute the training loss and the network features of the candidate. Default is False.
    network_features : dict, optional
        Network features (name -> function, see 'pyreco.features.resolve_features') to evaluate on the
        refit candidate if 'record' is True.

    Returns
    -------
    _CandidateResult
        'score': loss of the refit model with 'candidate' removed, evaluated on (x_test, y_test) using
        'criterion'; 'model': deep copy of 'model' with 'candidate' removed and refit; if 'record' is
        True also 'train_score' (loss on (x_train, y_train)) and 'network_features'.
    """
    _model = copy.deepcopy(model)
    _remove_edge(_model, candidate, target_spec_rad)
    _model.fit(x=x_train, y=y_train)
    _score = _model.evaluate(x=x_test, y=y_test, metrics=criterion)[0]
    if not record:
        return _CandidateResult(_score, _model)
    _train_score = _model.evaluate(x=x_train, y=y_train, metrics=criterion)[0]
    _features = evaluate_network_features(network_features or {}, FeatureContext(_model, x_train))
    return _CandidateResult(_score, _model, _train_score, _features)


class EdgePruner:
    r"""
    Prune the edges of the reservoir network one after the other.

    In iteration :math:`k` the network has the edge set :math:`E_k`. A random fraction
    :math:`\varphi` (``candidate_fraction``) of the edges is chosen as candidates,
    :math:`C_k \subset E_k` with :math:`|C_k| = \max(1, \mathrm{round}(\varphi |E_k|))`. For every
    candidate :math:`e` the edge is removed tentatively, the readout is retrained, and the loss
    :math:`L_\mathrm{val}(e)` on the validation data is computed. The candidate that hurts the
    performance least is removed for good,

    .. math::

        e^*_k = \arg\min_{e \in C_k} L_\mathrm{val}\!\left( A_k \setminus e \right), \qquad
        E_{k+1} = E_k \setminus \{ e^*_k \} ,

    where :math:`A_k \setminus e` is the adjacency matrix with the entry of :math:`e` set to 0. The pruning
    stops when a stopping criterion is met: no improvement of the loss for ``patience`` iterations, a
    minimum number of nodes or edges. The history records the loss and the properties of the network
    (density, spectral radius, degrees, ...) per iteration. The history holds the initial reservoir and,
    per iteration, the reservoir that survives it, with its performance and its named ``features`` (network,
    edge and node properties, see :mod:`pyreco.features`). With ``record_candidates=True`` it also holds a
    compact table of all candidates of every iteration. See :meth:`prune` for the layout.

    Use :meth:`prune` to run it; the constructor arguments are documented below.
    """

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
        remove_isolated_nodes: bool = False,
        directed: bool = True,
        parallel: bool = False,
        seed: int = None,
        rescale_spec_rad: Union[None, float, str] = None,
        record_candidates: bool = False,
        features: Optional[list] = None,
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
            of which ``pruning_criterion`` is active. Must be a loss (lower is
            better) from :func:`pyreco.metrics.available_metrics`, i.e. not
            "r2". Default is "mse".
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
        remove_isolated_nodes : bool, optional
            Whether to remove isolated nodes during pruning. Default is
            False.
        directed : bool, optional
            Whether the reservoir graph is treated as directed. Default is
            True.
        parallel : bool, optional
            Whether to parallelize candidate evaluation across CPU cores
            using ``joblib``. Default is False.
        seed : int, optional
            Seed for the random choice of pruning candidates. Every call of
            ``prune`` restarts from this seed, so pruning the same model twice
            proposes the same candidates. The global random state is not used.
            None draws fresh entropy. Default is None.
        rescale_spec_rad : None, float or "initial", optional
            Keep the spectral radius of the reservoir fixed. After an edge is removed (also for every
            candidate, before it is scored) the remaining weights are rescaled to this radius. A float is
            the target radius (the model that is pruned is rescaled to it and its readout refit first),
            ``"initial"`` is the radius of the model that is pruned. None (default) does not rescale:
            the radius then changes as edges are removed. A reservoir without a directed cycle has
            radius 0 and cannot be rescaled; the rescaling stops there, with a warning.
        record_candidates : bool, optional
            Keep a table of all candidates of every iteration, ``history[i]['candidates']`` (see
            :meth:`prune`): the loss on the validation and on the training data and the network features of
            every candidate reservoir. Costs one more forward pass through the training data per
            candidate, plus one for the features that need the activity of the nodes. Default is False:
            the candidates are evaluated for their validation loss only, and only the reservoir that wins
            is described.
        features : list of str or callable, optional
            The named features to measure (see :mod:`pyreco.features`): names of registered features, or
            functions decorated with ``@network_feature(name)``, ``@edge_feature(name)`` or
            ``@node_feature(name)``, all levels mixed. They describe the initial reservoir and the reservoir
            that survives each iteration (all three levels), and, with ``record_candidates``, every
            candidate (network features). None (default) selects ``pyreco.features.DEFAULT_NETWORK_FEATURES``,
            ``DEFAULT_EDGE_FEATURES`` and ``DEFAULT_NODE_FEATURES``; an empty list measures nothing. Features
            that use the activity of the nodes (e.g. ``'mean_sq_activity'``) need one pass of the model
            through the training data per reservoir that is described.

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
            remove_isolated_nodes,
            directed,
            parallel,
            record_candidates,
            )

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
        # Parameter bools for extra functionalities
        self.return_best_model = return_best_model
        self.remove_isolated_nodes = remove_isolated_nodes
        self.parallel = parallel
        self.record_candidates = record_candidates
        # Named features: name -> function, for the network, the edges and the nodes
        self.features = features
        self._network_features, self._edge_features, self._node_features = resolve_features(features)
        # Spectral radius that is kept fixed (resolved in prune for "initial")
        _check_rescale_spec_rad(rescale_spec_rad)
        self.rescale_spec_rad = rescale_spec_rad
        self._target_spec_rad = None
        # Random stream for the candidate selection
        check_seed(seed)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Initialize history dict to store the history of the pruning process in a
        #  nested dictionary
        self.history = {}

        # Initialize attributes that will be used during pruning (and changed during
        #  the process)
        # Needs to be attributes as the history updates depend on them
        self._curr_model = None
        self._curr_loss = None
        self._curr_train_loss = None
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
            The record of the pruning. ``history['initial']`` describes the reservoir before the
            pruning, ``history[i]`` (``i = 0, 1, ...``) iteration ``i``, and ``history['features']`` lists
            the names of the features that were measured, ``{'network': [...], 'edge': [...], 'node': [...]}``
            (the keys of the ``'network'``, ``'edge'`` and ``'node'`` entries of the states).

            A *state* describes one reservoir (the initial one, or the one that survives an iteration):

            * ``'weights'`` (``scipy.sparse.csr_matrix``), ``'input_nodes'``, ``'readout_nodes'``: the
              reservoir, enough to rebuild it;
            * ``'val_loss'``, ``'train_loss'``: performance (``performance_criterion``) on the validation
              and on the training data; ``'metrics'``: ``{name: value}`` of ``metrics`` on the validation data;
            * ``'edges'``: array of shape (E, 2) with all edges ``(u, v)`` of the reservoir;
            * ``'network'``: ``{name: float}``, the network features (structure and activity);
            * ``'edge'``: ``{name: array of shape (E,)}``, the edge features of every edge, in the order of
              ``'edges'``;
            * ``'node'``: ``{name: array of shape (N,)}``, the node features of every node (e.g. the
              activity of each node).

            ``history[i]`` has

            * ``'state'``: the state of the reservoir that results from iteration ``i`` (after the removal of
              isolated nodes, if requested);
            * ``'winner_edge'``: the edge ``(u, v)`` that was removed, ``'winner_row'``: its position among
              the candidates of the iteration (the index into the arrays of ``'candidates'``);
            * ``'removed_edge'``: ``{name: float}``, the edge features of the removed edge, measured in the
              reservoir before the removal (the entry of the previous state's ``'edge'`` table, or of the
              initial one);
            * ``'removed_nodes'``: the isolated nodes that were removed in this iteration;
            * ``'candidates'``, only with ``record_candidates=True``: ``{'edges': array of shape (n, 2),
              'val_loss': array (n,), 'train_loss': array (n,), 'network': {name: array (n,)},
              'removed_edge': {name: array (n,)}}``, one entry per candidate in the same order: its removed
              edge, its performance, the network features of the candidate reservoir and the edge features of
              the edge it removes. The winner is one of the rows (``'winner_row'``).

            The winner is the candidate with the lowest validation loss. The state at the start of
            iteration ``i + 1`` is ``history[i]['state']``. The training and validation data must be aligned
            like in ``fit`` (targets shorter than the inputs by ``discard_transients``). The states of the
            iterations in the patience window stay in the history: cutting it at the best state is up to the
            caller.

        Raises
        ------
        ValueError
            If ``'min_edges'`` is a stopping criterion and ``min_num_edges`` is larger than the number of
            edges of the model, or for invalid models and data.
        """

        # Sanity checks for the input parameter types and values
        self._validate_pruning_params(model, data_train, data_val)

        # Restart the random stream of the candidate selection from the seed
        self._rng = np.random.default_rng(self.seed)
        self._warned_zero_spec_rad = False

        # a pruner can be used for several models: start with an empty history and no patience used up
        self.history = {}
        self._best_loss = None
        self._patience_counter = 0

        # Obtain training and testing data
        x_test, y_test = data_val[0], data_val[1]
        x_train, y_train = data_train[0], data_train[1]

        # Keep the spectral radius fixed: resolve the target and bring the starting model to it
        self._target_spec_rad = None
        if self.rescale_spec_rad is not None:
            _start_spec_rad = model.reservoir_layer.spec_rad
            if _start_spec_rad <= 0:
                raise ValueError('rescale_spec_rad needs a reservoir with a directed cycle: '
                                 'the spectral radius of the model is 0.')
            if self.rescale_spec_rad == 'initial':
                self._target_spec_rad = float(_start_spec_rad)
            else:
                self._target_spec_rad = float(self.rescale_spec_rad)
                if not np.isclose(_start_spec_rad, self._target_spec_rad, rtol=1e-6):
                    # the pruned model starts at the target radius, with a readout trained for it
                    model = copy.deepcopy(model)
                    model.reservoir_layer.set_spec_rad(self._target_spec_rad)
                    model.fit(x=x_train, y=y_train)

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

        # a minimum above the edge count could never be reached: nothing would be pruned, silently
        if 'min_edges' in self.stopping_criterion and self.min_num_edges > self._curr_num_edges:
            raise ValueError(
                f'min_num_edges ({self.min_num_edges}) is larger than the number of edges of the '
                f'model ({self._curr_num_edges}): there is nothing to prune. Use a smaller '
                f'min_num_edges.'
            )

        # Initialize the quantities that affect the stop condition
        self._curr_loss = model.evaluate(x=x_test, y=y_test,
                                         metrics=self.performance_criterion)[0]
        self._curr_loss_history = [self._curr_loss]
        self._curr_train_loss = model.evaluate(x=x_train, y=y_train,
                                               metrics=self.performance_criterion)[0]

        # Initialize quantities that we track for the pruning history
        # These do not affect the pruning process
        self._curr_metrics = model.evaluate(x=x_test, y=y_test, metrics=self.metrics)

        # The state of the reservoir before the pruning. The state of the previous iteration is the parent
        #   of the candidates of the next one: their removed edges are rows of its edge table
        parent_state = self._make_state(model, x_train)
        self.history['initial'] = parent_state
        self.history['features'] = {'network': list(self._network_features),
                                    'edge': list(self._edge_features),
                                    'node': list(self._node_features)}

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

            print(f'Currently at pruning iteration {self._iter_count} ...')

            print(
                f'Current reservoir size: {self._curr_num_nodes} | '
                f'Current loss: {self._curr_loss:.8f}'
            )

            # Get candidates for pruning
            _curr_candidates = self._get_candidates(_graph)

            # Apply chosen pruning strategy on candidates to get scores for candidates
            _candidate_results = self._apply_pruning_strategy(model, _curr_candidates, x_train,
                                                              y_train, x_test, y_test)
            _candidate_scores = [result.score for result in _candidate_results]
            _candidate_models = [result.model for result in _candidate_results]

            # Where the removed edges are in the edge table of the parent
            _parent_rows = self._rows_in_edge_table(parent_state, _curr_candidates)

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
                break

            if isinstance(pruned_candidate, tuple):
                # Clean print of pruned_candidate
                pruned_candidate = (int(pruned_candidate[0]), int(pruned_candidate[1]))
            print(f'Pruning candidate {pruned_candidate}, '
                  f'resulting in loss {self._curr_loss:.6f}')

            prev_loss = self._curr_loss_history[-2]
            loss_improvement = (prev_loss-self._curr_loss) / prev_loss if prev_loss != 0 else 0.0
            print(
                'Loss improvement to previous iteration by '
                f'{loss_improvement:+.3%}\n'
            )

            # Prune edge that gives us the least performance drop
            # As we have already pruned edge and stored the model, we only need to
            #   update the model
            # Get it from self._curr_model to not discard isolated-node removal
            # model = _candidate_models[idx_prune]
            model = self._curr_model
            _graph = model.reservoir_layer.weights

            # Pruning can remove every directed cycle. The reservoir is then a
            #   feed-forward network: it still runs, but it can no longer be rescaled
            #   to a target spectral radius (set_hp(spec_rad=...) raises an error).
            if not self._warned_zero_spec_rad and model.reservoir_layer.spec_rad == 0:
                warnings.warn(
                    f'Pruning iteration {self._iter_count} removed the last directed '
                    'cycle from the reservoir: its spectral radius is now 0'
                    + (' and it can no longer be rescaled.' if self._target_spec_rad is not None else '.'),
                    RuntimeWarning,
                )
                self._warned_zero_spec_rad = True

            # Store the model and loss for later use
            _pruned_models.append(copy.deepcopy(model))
            _pruned_models_losses.append(self._curr_loss)

            # Compute things that are required for history, but not for
            #   pruning loop termination criteria
            self._curr_metrics = model.evaluate(
                x=x_test, y=y_test, metrics=self.metrics
            )
            # (after the isolated-node removal, which can refit the model)
            self._curr_train_loss = model.evaluate(x=x_train, y=y_train,
                                                   metrics=self.performance_criterion)[0]

            # The record of this iteration: the reservoir that survived, the edge that was removed
            state = self._make_state(model, x_train)
            winner_row = _parent_rows[idx_prune]
            entry = {
                'state': state,
                'winner_edge': (int(_curr_candidates[idx_prune][0]), int(_curr_candidates[idx_prune][1])),
                'winner_row': int(idx_prune),
                'removed_edge': {name: float(values[winner_row])
                                 for name, values in parent_state['edge'].items()},
                'removed_nodes': removed_nodes,
            }
            if self.record_candidates:
                entry['candidates'] = self._candidate_table(_curr_candidates, _candidate_results,
                                                            _parent_rows, parent_state)
            self._store_iteration(model, entry, _candidate_scores)
            parent_state = state

            # Update iteration counter
            self._iter_count += 1

        # In case we have a non-zero patience, we might want to return the best model
        #   instead of the last one (i.e. when a positive patience value was given)
        if self.return_best_model:
            idx_best = np.argmin(_pruned_models_losses)
            model = copy.deepcopy(_pruned_models[idx_best])
            print(f'\nReturning the model after {idx_best} pruned edges as the best')

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

    def _make_state(self, model, x_train):
        """
        Describe one reservoir (the initial one, or the one that survives an iteration), with the loss
        and the metrics that are current in ``self``. See :meth:`prune` for the entries.
        """
        weights = model.reservoir_layer.weights
        edges = np.argwhere(weights != 0)  # row-major, the order of the edge table
        context = FeatureContext(model, x_train)
        metric_names = self.metrics if isinstance(self.metrics, list) else [self.metrics]
        return {
            'weights': sp.csr_matrix(weights),
            'input_nodes': list(model.reservoir_layer.input_receiving_nodes),
            'readout_nodes': list(model.readout_layer.readout_nodes),
            'val_loss': self._curr_loss,
            'train_loss': self._curr_train_loss,
            'metrics': dict(zip(metric_names, self._curr_metrics)),
            'edges': edges,
            'network': evaluate_network_features(self._network_features, context),
            'edge': evaluate_edge_features(self._edge_features, context, edges),
            'node': evaluate_node_features(self._node_features, context),
        }

    @staticmethod
    def _rows_in_edge_table(state, edges):
        """Row of every edge ``(u, v)`` of ``edges`` in the edge table of ``state``."""
        row_of = {(int(u), int(v)): row for row, (u, v) in enumerate(state['edges'])}
        return np.array([row_of[(int(u), int(v))] for u, v in edges], dtype=int)

    def _candidate_table(self, candidates, results, rows, parent_state):
        """
        Table of all candidates of an iteration: arrays with one entry per candidate, in the order of
        ``candidates``. ``rows`` are the positions of the removed edges in the edge table of the parent.
        """
        return {
            'edges': np.array([(int(u), int(v)) for u, v in candidates], dtype=int).reshape(-1, 2),
            'val_loss': np.array([result.score for result in results], dtype=float),
            'train_loss': np.array([result.train_score for result in results], dtype=float),
            'network': {name: np.array([result.network_features[name] for result in results], dtype=float)
                        for name in self._network_features},
            'removed_edge': {name: values[rows] for name, values in parent_state['edge'].items()},
        }

    def _store_iteration(self, model, entry, candidate_val_losses):
        """
        Put the record of the iteration that has just finished into the history.

        This is the place to override in a subclass that wants to keep less (or write the record to disk): it
        gets the model that survived, the record (see :meth:`prune`) and the validation loss of every
        candidate, also if ``record_candidates`` is False.
        """
        self.history[self._iter_count] = entry

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
            directed=self.directed,
            rng=self._rng,
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
        list of _CandidateResult
            One per candidate, in the order of ``candidates``: the score (lower is better, i.e. a better
            pruning choice) and the model that results from removing the edge, and with
            ``record_candidates`` its training loss and network features.
        """
        method = getattr(self, self.PRUNING_CRITERION[self.pruning_criterion])
        return method(model, candidates, x_train, y_train, x_test, y_test)

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
        list of _CandidateResult
            One per candidate, in the same order as ``candidates``: the validation loss of the refit model and
            the refit model, and with ``record_candidates`` its training loss and network features.
        """

        if self.parallel:
            # n_jobs=-1 uses all CPU cores, 'loky' is the default backend; tqdm shows a progress bar.
            # The results come back in the order of the candidates. A separate evaluation function that
            #   does not pass the whole self object is called: the history gets larger with each
            #   iteration and would slow down the transfer to the workers.
            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(_evaluate_candidate_standalone)
                (model, c, x_train, y_train, x_test, y_test, self.performance_criterion,
                 self._target_spec_rad, self.record_candidates, self._network_features)
                for c in tqdm(candidates, desc='Evaluating candidates')
                )
        else:
            # Go through candidates one by one in a single process
            results = [
                self._evaluate_candidate_performance(model, candidate, x_train, y_train,
                                                     x_test, y_test)
                for candidate in candidates
            ]

        return list(results)

    def _evaluate_candidate_performance(self, model, candidate,
                                        x_train, y_train, x_test, y_test):
        """
        Evaluate a single candidate edge removal (serial path).

        Calls :func:`_evaluate_candidate_standalone` with the settings of the pruner and prints the
        result.

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
        _CandidateResult
            See :func:`_evaluate_candidate_standalone`; ``score`` is the loss of the refit model with
            ``candidate`` removed, evaluated on (x_test, y_test) using ``self.performance_criterion``.
        """
        result = _evaluate_candidate_standalone(
            model, candidate, x_train, y_train, x_test, y_test, self.performance_criterion,
            self._target_spec_rad, self.record_candidates, self._network_features)
        score = result.score

        label = f"{int(candidate[0])}-{int(candidate[1])}" if \
                isinstance(candidate, tuple) else int(candidate)
        print(f'Possible deletion of edge {label:<10} loss: {score:.6f}  '
              f'({(self._curr_loss - score) / self._curr_loss:+.3%})')

        return result

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
        remove_isolated_nodes,
        directed,
        parallel,
        record_candidates=False,
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
        remove_isolated_nodes : bool
            See ``__init__``.
        directed : bool
            See ``__init__``.
        parallel : bool
            See ``__init__``.
        record_candidates : bool
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
        if not is_loss(performance_criterion):
            # candidates are ranked by the lowest value, so a score for which higher is
            #   better (e.g. r2) would select the worst candidate
            raise ValueError(
                f"performance_criterion '{performance_criterion}' is a score for which "
                'higher is better, but pruning minimizes the criterion. Use a loss '
                "(e.g. 'mse', 'mae', 'rmse', 'nrmse'); a score such as 'r2' can be "
                'tracked through the metrics argument.'
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

        # Validate remove isolated nodes
        if not isinstance(remove_isolated_nodes, bool):
            raise TypeError('remove_isolated_nodes must be a boolean')

        # Validate directed
        if not isinstance(directed, bool):
            raise TypeError('directed must be a boolean')

        # Validate parallel
        if not isinstance(parallel, bool):
            raise TypeError('parallel must be a boolean')

        # Validate record_candidates
        if not isinstance(record_candidates, bool):
            raise TypeError('record_candidates must be a boolean')

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
