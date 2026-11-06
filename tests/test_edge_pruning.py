"""
EdgePruner: constructor validation, validation of the data passed to prune(), the pruning loop and its
history, the stopping criteria, best-model selection, isolated nodes, parallel candidate evaluation.
"""

import copy

import numpy as np
import pytest
import scipy.sparse as sp

from pyreco.custom_models import RC
from pyreco.edge_pruning import EdgePruner
from pyreco.features import DEFAULT_EDGE_FEATURES, DEFAULT_NETWORK_FEATURES, DEFAULT_NODE_FEATURES
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.optimizers import RidgeSK
from pyreco.utils_data import sequence_to_sequence as seq_2_seq


def iteration_ids(history):
    """The keys of the iterations, in order (the history also holds 'initial')."""
    return sorted(key for key in history if isinstance(key, int))


def n_edges_of(state):
    return len(state['edges'])


def parent_of(history, iteration):
    """The state at the start of an iteration: the previous state, or the initial one."""
    return history['initial'] if iteration == 0 else history[iteration - 1]['state']


# Fixtures, to reuse for testing

# Small dataset for testing
@pytest.fixture
def small_data():
    X_train, X_test, y_train, y_test = seq_2_seq(
        name="sine_pred", n_batch=10, n_states=2, n_time=50
    )
    return (X_train, y_train), (X_test, y_test)


# Small trained RC with randomly generated reservoir
@pytest.fixture
def small_model(small_data):
    data_train, _ = small_data
    X_train, y_train = data_train

    model = RC()
    model.add(InputLayer(input_shape=X_train.shape[1:]))
    model.add(RandomReservoirLayer(
        nodes=10,
        density=0.4,
        activation="tanh",
        leakage_rate=0.1,
        fraction_input=0.5,
    ))
    model.add(ReadoutLayer(y_train.shape[1:], fraction_out=0.9))
    model.compile(optimizer=RidgeSK(alpha=0.5), metrics=["mean_squared_error"])
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def isolatable_data():
    X_train, X_test, y_train, y_test = seq_2_seq(
        name="sine_pred", n_batch=8, n_states=2, n_time=30
    )
    return (X_train, y_train), (X_test, y_test)


def _build_fixed_topology_model(x_train, y_train, mat, input_nodes, readout_nodes):
    '''
    Build an RC model with a hand-specified reservoir adjacency matrix and
    fixed input-receiving/readout node assignments.
    '''
    model = RC()
    model.add(InputLayer(input_shape=x_train.shape[1:]))
    model.add(RandomReservoirLayer(
        nodes=20, density=0.3, activation="tanh", leakage_rate=0.1, fraction_input=0.3,
    ))
    model.reservoir_layer.set_weights(mat)
    model.add(ReadoutLayer(y_train.shape[1:], fraction_out=0.3))
    model.compile(optimizer=RidgeSK(alpha=0.5), metrics=["mean_squared_error"])
    model.reservoir_layer.input_receiving_nodes = input_nodes
    model.readout_layer.readout_nodes = readout_nodes
    model.fit(x_train, y_train)
    return model


@pytest.fixture
def isolatable_model(isolatable_data):
    '''
    Small RC model.
    Nodes 0-4 form reservoir with input node 0 and readout  nodes 1 and 2.
    Node 5 is isolated, not input-receiving, not readout.
    '''
    data_train, _ = isolatable_data
    X_train, y_train = data_train

    mat = np.zeros((6, 6))
    mat[0, 1] = 0.5
    mat[1, 2] = 0.5
    mat[2, 0] = 0.5
    mat[1, 3] = 0.5
    mat[3, 1] = 0.5
    mat[0, 4] = 0.5
    mat[4, 2] = 0.5
    # node 5: all-zero row and column -> isolated
    return _build_fixed_topology_model(X_train, y_train, mat,
                                       input_nodes=[0], readout_nodes=[1, 2])


def _count_edges(model):
    graph = model.reservoir_layer.weights
    if isinstance(graph, np.ndarray):
        return int(np.count_nonzero(graph))
    return graph.number_of_edges()


class TestInit:
    """Defaults and argument validation of the constructor."""

    def test_default_values(self):
        # Constructor defaults match the documented values
        pruner = EdgePruner()
        assert pruner.candidate_fraction == 0.1
        assert pruner.pruning_criterion == 'performance'
        assert pruner.stopping_criterion == ['patience']
        assert pruner.min_num_nodes == 3
        assert pruner.min_num_edges == 2
        assert pruner.patience == 0
        assert pruner.performance_criterion == 'mse'
        assert pruner.structural_criterion == 'betweenness'
        assert pruner.metrics == ['mse']
        assert pruner.return_best_model is True
        assert pruner.remove_isolated_nodes is False
        assert pruner.directed is True
        assert pruner.parallel is False
        assert pruner.record_candidates is False
        assert pruner.features is None
        assert pruner.edge_selection_strat == 'random_uniform_wo_repl'

    def test_the_analyzer_arguments_are_gone(self):
        # the features replace the analyzers
        for name in ('graph_analyzer', 'node_analyzer', 'edge_analyzer'):
            with pytest.raises(TypeError):
                EdgePruner(**{name: None})

    def test_history_starts_empty(self):
        # History starts out as empty dict
        assert EdgePruner().history == {}

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            # strategy names: wrong type, unknown name, and 'structure' that is registered but not implemented
            (dict(edge_selection_strat=42), TypeError),
            (dict(edge_selection_strat="not_a_strategy"), NotImplementedError),
            (dict(pruning_criterion=["performance"]), TypeError),
            (dict(pruning_criterion="unknown"), NotImplementedError),
            (dict(pruning_criterion="structure"), NotImplementedError),
            # candidate_fraction is a float in (0, 1]
            (dict(candidate_fraction=1), TypeError),
            (dict(candidate_fraction=0.0), ValueError),
            (dict(candidate_fraction=1.5), ValueError),
            # stopping criteria: a non-empty list of known names
            (dict(stopping_criterion="patience"), TypeError),
            (dict(stopping_criterion=[]), ValueError),
            (dict(stopping_criterion=["unknown"]), NotImplementedError),
            # integer limits
            (dict(min_num_nodes=3.0), TypeError),
            (dict(min_num_nodes=2), ValueError),  # larger than 2
            (dict(min_num_edges=2.0), TypeError),
            (dict(min_num_edges=-1), ValueError),  # at least 0
            (dict(patience=1.0), TypeError),
            (dict(patience=-1), ValueError),
            # criteria and metrics: strings that name a known loss / edge property / metric
            (dict(performance_criterion=1), TypeError),
            (dict(performance_criterion="not_a_metric"), ValueError),
            (dict(structural_criterion=1), TypeError),
            (dict(structural_criterion="not_a_property"), ValueError),
            (dict(metrics=1), TypeError),
            (dict(metrics=["mse", 1]), TypeError),
            (dict(metrics=["not_a_metric"]), ValueError),
            # flags are booleans
            (dict(return_best_model="yes"), TypeError),
            (dict(remove_isolated_nodes="yes"), TypeError),
            (dict(directed="yes"), TypeError),
            (dict(parallel="yes"), TypeError),
            (dict(record_candidates="yes"), TypeError),
            (dict(features="density"), TypeError),
            (dict(features=["not_a_feature"]), ValueError),
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs, error):
        """Every constructor argument is validated, wrong types (TypeError) and values (ValueError)."""
        with pytest.raises(error):
            EdgePruner(**kwargs)

    def test_structural_pruning_is_not_implemented_yet(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            EdgePruner(pruning_criterion="structure")

    def test_min_num_edges_zero_allowed(self):
        """0 is a valid boundary value for min_num_edges."""
        assert EdgePruner(min_num_edges=0).min_num_edges == 0


# Test prune() parameter validation

class TestPruningParams:
    """prune() validates the model and the (X, y) tuples before it does anything."""


    def test_invalid_model_type(self, small_data):
        # model must be RC instance
        pruner = EdgePruner()
        data_train, data_val = small_data
        with pytest.raises(TypeError):
            pruner.prune(model="not a model", data_train=data_train, data_val=data_val)

    def test_invalid_data_train_type(self, small_model, small_data):
        # data_train must be tuple
        pruner = EdgePruner()
        _, data_val = small_data
        with pytest.raises(TypeError):
            pruner.prune(model=small_model, data_train=[1, 2], data_val=data_val)

    def test_invalid_data_val_type(self, small_model, small_data):
        # data_val must be tuple
        pruner = EdgePruner()
        data_train, _ = small_data
        with pytest.raises(TypeError):
            pruner.prune(model=small_model, data_train=data_train, data_val=[1, 2])

    def test_data_train_wrong_length(self, small_model, small_data):
        # data_train must have 2 elements (X, y)
        pruner = EdgePruner()
        data_train, data_val = small_data
        with pytest.raises(ValueError):
            pruner.prune(model=small_model, data_train=(data_train[0],),
                         data_val=data_val)

    def test_data_val_wrong_length(self, small_model, small_data):
        # data_val must also have 2 elements (X, y)
        pruner = EdgePruner()
        data_train, data_val = small_data
        with pytest.raises(ValueError):
            pruner.prune(model=small_model, data_train=data_train,
                         data_val=(data_val[0],))

    def test_data_train_mismatched_samples(self, small_model, small_data):
        # X and y in data_train must have same number of samples
        pruner = EdgePruner()
        data_train, data_val = small_data
        X_train, y_train = data_train
        with pytest.raises(ValueError):
            pruner.prune(model=small_model, data_train=(X_train, y_train[:-1]),
                         data_val=data_val)

    def test_data_val_mismatched_samples(self, small_model, small_data):
        # X and y in data_val must have same number of samples
        pruner = EdgePruner()
        data_train, data_val = small_data
        X_test, y_test = data_val
        with pytest.raises(ValueError):
            pruner.prune(model=small_model, data_train=data_train,
                         data_val=(X_test, y_test[:-1]))


# Tes pruning loop and history structure

class TestPruningLoop:
    """One edge is removed per iteration; the history records candidates, winner and snapshots."""


    def test_min_num_edges_above_the_edge_count_is_an_error(self, small_model, small_data):
        # a minimum that can never be reached would silently prune nothing: it is an error instead
        data_train, data_val = small_data
        n_edges = int(np.count_nonzero(small_model.reservoir_layer.weights))
        pruner = EdgePruner(stopping_criterion=['min_edges'], min_num_edges=n_edges + 1)
        with pytest.raises(ValueError, match="min_num_edges"):
            pruner.prune(model=small_model, data_train=data_train, data_val=data_val)
        # the model is untouched, and a minimum equal to the edge count is allowed (nothing to prune)
        assert int(np.count_nonzero(small_model.reservoir_layer.weights)) == n_edges
        EdgePruner(stopping_criterion=['min_edges'], min_num_edges=n_edges).prune(
            model=small_model, data_train=data_train, data_val=data_val)

    def test_min_num_edges_is_only_checked_for_the_min_edges_criterion(self, small_model, small_data):
        # with another stopping criterion the minimum plays no role, whatever its value
        data_train, data_val = small_data
        n_edges = int(np.count_nonzero(small_model.reservoir_layer.weights))
        EdgePruner(stopping_criterion=['patience'], patience=0, min_num_edges=n_edges + 10).prune(
            model=small_model, data_train=data_train, data_val=data_val)

    def test_a_pruner_can_be_used_again(self, small_model, small_data):
        # history, best loss and patience start afresh in every prune() call
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'], min_num_edges=6)
        _, first = pruner.prune(model=copy.deepcopy(small_model), data_train=data_train, data_val=data_val)
        n_first = len(first)
        pruner.min_num_edges = int(np.count_nonzero(small_model.reservoir_layer.weights)) - 1
        _, second = pruner.prune(model=copy.deepcopy(small_model), data_train=data_train, data_val=data_val)
        assert n_first > 3 and len(second) == 3  # no entries of the first run are left over
        assert set(second) == {'initial', 'features', 0}

    def test_returns_model_and_history(self, small_model, small_data):
        # prune() returns a fitted RC model and a non-empty history dict
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        model, history = pruner.prune(model=small_model, data_train=data_train,
                                      data_val=data_val)
        assert isinstance(model, RC)
        assert isinstance(history, dict)
        assert len(history) > 0

    def test_history_keyed_by_consecutive_iteration_ints(self, small_model, small_data):
        # history keys are consecutive integers (0 to N-1/ one per completed iteration)
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        assert iteration_ids(history) == list(range(len(history) - 2))
        assert set(history) == {'initial', 'features', *iteration_ids(history)}

    def test_history_entry_has_expected_top_level_keys(self, small_model, small_data):
        # Iteration's history entry exposes exactly these top-level keys; the candidate table only on request
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        assert set(history[0].keys()) == {
            'state', 'winner_edge', 'winner_row', 'removed_edge', 'removed_nodes'
        }

    def test_the_state_of_the_initial_reservoir_and_of_the_survivors_have_the_same_entries(
            self, small_model, small_data):
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        expected = {'weights', 'input_nodes', 'readout_nodes', 'val_loss', 'train_loss', 'metrics', 'edges',
                    'network', 'edge', 'node'}
        assert set(history['initial']) == expected
        assert set(history[0]['state']) == expected
        assert list(history['initial']['metrics']) == ['mse']

    def test_the_history_lists_the_names_of_the_features(self, small_model, small_data):
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'], min_num_edges=8,
                            features=['density', 'abs_weight', 'weight', 'node_mean_sq_activity', 'spectral_radius'])
        _, history = pruner.prune(model=small_model, data_train=data_train, data_val=data_val)
        assert history['features'] == {'network': ['density', 'spectral_radius'], 'edge': ['abs_weight', 'weight'],
                                       'node': ['node_mean_sq_activity']}
        for key in ('network', 'edge', 'node'):
            assert list(history[0]['state'][key]) == history['features'][key]
        _, history = EdgePruner(features=[], stopping_criterion=['min_edges'], min_num_edges=8).prune(
            copy.deepcopy(small_model), data_train, data_val)
        assert history['features'] == {'network': [], 'edge': [], 'node': []}

    def test_the_old_keys_are_gone(self, small_model, small_data):
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for old in ('starting_model', 'final_model', 'candidates', 'winner', 'candidate_features'):
            assert old not in history[0]

    def test_the_state_describes_its_reservoir(self, small_model, small_data):
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for key in ['initial'] + iteration_ids(history):
            state = history[key] if key == 'initial' else history[key]['state']
            weights = state['weights'].toarray()
            n_nodes, n_edges = weights.shape[0], n_edges_of(state)
            assert (np.argwhere(weights != 0) == state['edges']).all()
            assert all(column.shape == (n_edges,) for column in state['edge'].values())
            assert all(column.shape == (n_nodes,) for column in state['node'].values())
            assert state['network']['density'] == pytest.approx(n_edges / (n_nodes * (n_nodes - 1)))
            assert np.isfinite(state['val_loss']) and np.isfinite(state['train_loss'])

    def test_the_initial_state_is_the_model_that_was_passed(self, small_model, small_data):
        data_train, data_val = small_data
        _, history = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                                min_num_edges=8, seed=0).prune(copy.deepcopy(small_model), data_train, data_val)
        initial = history['initial']
        assert (initial['weights'].toarray() == small_model.reservoir_layer.weights).all()
        assert initial['val_loss'] == pytest.approx(small_model.evaluate(data_val[0], data_val[1], metrics='mse')[0])
        assert initial['train_loss'] == pytest.approx(
            small_model.evaluate(data_train[0], data_train[1], metrics='mse')[0])

    def test_winner_edge_is_zeroed_in_the_state(self, small_model, small_data):
        # Comparing the parent and the new weights for iteration, the winner edge
        #   must go from nonzero to zero
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for i in iteration_ids(history):
            u, v = history[i]['winner_edge']
            assert parent_of(history, i)['weights'][u, v] != 0
            assert history[i]['state']['weights'][u, v] == 0

    def test_edge_count_decreases_by_one_per_iteration(self, small_model, small_data):
        # One edge is removed per completed iteration
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for i in iteration_ids(history):
            assert n_edges_of(history[i]['state']) == n_edges_of(parent_of(history, i)) - 1

    def test_stops_exactly_at_min_edges(self, small_model, small_data):
        # Pruning stops when edge count reaches min_num_edges
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        last_iter = max(iteration_ids(history))
        assert n_edges_of(history[last_iter]['state']) == 8

    def test_returned_model_has_fewer_edges_than_original(self, small_model,
                                                          small_data):
        # Returned model has strictly fewer edges than original
        data_train, data_val = small_data
        edges_before = _count_edges(small_model)
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8, return_best_model=False)
        model, _ = pruner.prune(model=small_model, data_train=data_train,
                                data_val=data_val)
        assert _count_edges(model) < edges_before


# Test stopping criteria

class TestStoppingCriteria:
    """Patience, minimum number of edges and their combination end the loop at the defined moment."""


    def test_patience_zero_loss_never_worsens_across_kept_iterations(self, small_model,
                                                                     small_data):
        # With patience=0, iteration is discarded and pruning stops
        #   the moment loss fails to improve
        # -> iteration's losses before must be lower than previous one
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['patience'],
                            patience=0)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        losses = [history[it]['state']['val_loss'] for it in iteration_ids(history)]
        assert all(losses[i + 1] <= losses[i] + 1e-9 for i in range(len(losses) - 1))

    def test_patience_n_tolerates_at_most_n_trailing_non_improvements(self, small_model,
                                                                      small_data):
        # patience N allows N consecutive non-improving iterations before pruning stops
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['patience'],
                            patience=2)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        losses = [history[it]['state']['val_loss'] for it in iteration_ids(history)]
        running_best = float('inf')
        trailing_non_improvements = 0
        for loss in losses:
            if loss <= running_best:
                running_best = loss
                trailing_non_improvements = 0
            else:
                trailing_non_improvements += 1
        assert trailing_non_improvements <= 2

    def test_min_edges_stops_exactly_at_boundary(self, small_model, small_data):
        # min_edges stops pruning at specified edge count
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=6)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        last_iter = max(iteration_ids(history))
        assert n_edges_of(history[last_iter]['state']) == 6

    def test_multiple_stopping_criteria_combined(self, small_model, small_data):
        # Pruning stops as soon as one of specified criteria triggers
        data_train, data_val = small_data
        pruner = EdgePruner(
            candidate_fraction=1.0,
            stopping_criterion=['patience', 'min_edges'],
            patience=0,
            min_num_edges=2,
        )
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        assert len(history) > 1
        last_iter = max(iteration_ids(history))
        assert n_edges_of(history[last_iter]['state']) >= 2


# Test return_best_model

class TestReturnBestModel:
    """return_best_model chooses between the best and the last iteration."""


    def test_true_picks_lowest_recorded_loss(self, small_model, small_data):
        # return_best_model=True returns model with the lowest recorded loss
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['patience'],
                            patience=2, return_best_model=True)
        model, history = pruner.prune(model=small_model, data_train=data_train,
                                      data_val=data_val)
        final_loss = model.evaluate(x=data_val[0], y=data_val[1], metrics='mse')[0]
        all_losses = [history['initial']['val_loss']] + \
            [history[it]['state']['val_loss'] for it in iteration_ids(history)]
        assert final_loss <= min(all_losses) + 1e-6

    def test_false_returns_last_iteration_model(self, small_model, small_data):
        # return_best_model=False returns the last iteration's model
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['patience'],
                            patience=2, return_best_model=False)
        model, history = pruner.prune(model=small_model, data_train=data_train,
                                      data_val=data_val)
        last_iter = max(iteration_ids(history))
        expected_edges = n_edges_of(history[last_iter]['state'])
        assert _count_edges(model) == expected_edges


# Test isolated node handling

class TestIsolatedNodes:
    """Nodes without edges are detected and removed, except input and readout nodes where that would break the model."""


    def test_get_isolated_nodes_detects_isolated_node(self, isolatable_model):
        # Detects node with zero edges (node 5) in fixture
        pruner = EdgePruner()
        isolated = pruner._get_isolated_nodes(isolatable_model)
        assert [n['id'] for n in isolated] == [5]

    def test_get_isolated_nodes_flags_input_and_readout_correctly(self,
                                                                  isolatable_model):
        # Isolated node is correctly flagged as neither input-receiving
        #   nor a readout node
        pruner = EdgePruner()
        isolated = pruner._get_isolated_nodes(isolatable_model)
        node5 = isolated[0]
        assert node5['is_input'] is False
        assert node5['is_readout'] is False

    def test_remove_isolated_nodes_removes_non_input_non_readout(self, isolatable_model,
                                                                 isolatable_data):
        # Fully isolated, non-input, non-readout node is removed unconditionally,
        #   and survivors' readout indices are corrected
        data_train, data_val = isolatable_data
        pruner = EdgePruner()
        pruner._curr_loss = isolatable_model.evaluate(
            x=data_val[0], y=data_val[1], metrics='mse'
        )[0]
        isolated = pruner._get_isolated_nodes(isolatable_model)
        new_model, removed = pruner._remove_isolated_nodes(
            isolated, isolatable_model, data_train[0], data_train[1],
            data_val[0], data_val[1]
        )
        assert 5 in removed
        assert removed[5]['is_readout'] is False
        assert removed[5]['loss_after'] is None
        assert new_model.reservoir_layer.nodes == 5
        # readout nodes [1, 2] are both below removed index (5)
        #   -> ids must be unaffected by the removal
        assert list(new_model.readout_layer.readout_nodes) == [1, 2]

    def test_remove_isolated_nodes_never_touches_input_node(self, isolatable_data):
        # Isolated node is never removed if it's input-receiving node
        data_train, data_val = isolatable_data
        nodes = 6

        mat = np.zeros((nodes, nodes))
        mat[1, 2] = 0.5
        mat[2, 1] = 0.5
        mat[2, 3] = 0.5
        mat[3, 4] = 0.5
        mat[4, 5] = 0.5
        # node 0 is isolated, and explicitly only input-receiving node
        model = _build_fixed_topology_model(data_train[0], data_train[1], mat,
                                            input_nodes=[0], readout_nodes=[1, 3])

        pruner = EdgePruner()
        pruner._curr_loss = model.evaluate(x=data_val[0], y=data_val[1],
                                           metrics='mse')[0]
        isolated = pruner._get_isolated_nodes(model)
        assert [n['id'] for n in isolated] == [0]
        assert isolated[0]['is_input'] is True

        new_model, removed = pruner._remove_isolated_nodes(
            isolated, model, data_train[0], data_train[1], data_val[0], data_val[1]
        )
        assert removed == {}
        assert new_model.reservoir_layer.nodes == nodes

    def test_remove_isolated_nodes_readout_node_handled_consistently(self,
                                                                     isolatable_data):
        # Isolated readout node is removed only if doing so doesn't increase loss
        data_train, data_val = isolatable_data
        nodes = 6

        mat = np.zeros((nodes, nodes))
        mat[0, 1] = 0.5
        mat[1, 0] = 0.5
        mat[1, 2] = 0.5
        mat[2, 3] = 0.5
        mat[3, 4] = 0.5
        # node 5 is isolated, and readout node
        model = _build_fixed_topology_model(data_train[0], data_train[1], mat,
                                            input_nodes=[0], readout_nodes=[1, 5])

        pruner = EdgePruner()
        pruner._curr_loss = model.evaluate(x=data_val[0], y=data_val[1],
                                           metrics='mse')[0]
        isolated = pruner._get_isolated_nodes(model)
        assert [n['id'] for n in isolated] == [5]
        assert isolated[0]['is_readout'] is True

        new_model, removed = pruner._remove_isolated_nodes(
            isolated, model, data_train[0], data_train[1], data_val[0], data_val[1]
        )
        if removed:
            assert removed[5]['is_readout'] is True
            assert removed[5]['loss_after'] is not None
            assert removed[5]['loss_after'] <= pruner._curr_loss
            assert new_model.reservoir_layer.nodes == nodes - 1
        else:
            assert new_model.reservoir_layer.nodes == nodes

    def test_isolated_node_removal_persists_across_iterations(self, isolatable_model,
                                                              isolatable_data):
        # Node 5 is isolated from start -> must be detected and removed at iteration 0
        #   -> reduced node count must persist into every later iteration
        data_train, data_val = isolatable_data
        pruner = EdgePruner(
            candidate_fraction=1.0,
            stopping_criterion=['min_edges'],
            min_num_edges=2,
            remove_isolated_nodes=True,
            patience=0,
            return_best_model=False,
        )
        pruned_model, history = pruner.prune(
            model=isolatable_model, data_train=data_train, data_val=data_val
        )

        assert 5 in history[0]['removed_nodes']
        assert history[0]['state']['weights'].shape[0] == 5
        assert history['initial']['weights'].shape[0] == 6
        # readout nodes [1, 2] are both below removed index (5)
        #   -> ids must be unaffected by the removal
        assert list(history[0]['state']['readout_nodes']) == [1, 2]

        # Node count never goes up once reduced
        node_counts = [history[0]['state']['weights'].shape[0]]
        for it in iteration_ids(history)[1:]:
            assert parent_of(history, it)['weights'].shape[0] == node_counts[-1]
            node_counts.append(history[it]['state']['weights'].shape[0])
        assert all(node_counts[i + 1] <= node_counts[i] for i in
                   range(len(node_counts) - 1))
        assert node_counts[-1] <= 5

        assert pruned_model.reservoir_layer.nodes == node_counts[-1]


# Test parallel vs serial

class TestParallel:
    """Candidates evaluated in worker processes give the same winner as the serial loop."""


    def test_parallel_runs_without_error(self, small_model, small_data):
        # parallel=True produces valid model and history
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8, parallel=True)
        model, history = pruner.prune(model=small_model, data_train=data_train,
                                      data_val=data_val)
        assert isinstance(model, RC)
        assert len(history) > 0

    def test_parallel_and_serial_agree_on_first_winner(self, small_model, small_data):
        # With candidate_fraction=1.0 parallel and serial scoring must agree on winning
        #   edge and resulting loss
        data_train, data_val = small_data

        pruner_serial = EdgePruner(candidate_fraction=1.0,
                                   stopping_criterion=['min_edges'],
                                   min_num_edges=8, parallel=False)
        _, history_serial = pruner_serial.prune(
            model=copy.deepcopy(small_model), data_train=data_train, data_val=data_val
        )

        pruner_parallel = EdgePruner(candidate_fraction=1.0,
                                     stopping_criterion=['min_edges'],
                                     min_num_edges=8, parallel=True)
        _, history_parallel = pruner_parallel.prune(
            model=copy.deepcopy(small_model), data_train=data_train, data_val=data_val
        )

        assert history_serial[0]['winner_edge'] == history_parallel[0]['winner_edge']
        assert history_serial[0]['state']['val_loss'] == \
            pytest.approx(history_parallel[0]['state']['val_loss'])


class TestCandidates:
    """record_candidates keeps a table of all candidates; without it only the winner is described."""

    @staticmethod
    def run(model, data, removals=3, **kwargs):
        n_edges = _count_edges(model)
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=n_edges - removals, return_best_model=False, seed=0,
                            **kwargs)
        return pruner.prune(model=copy.deepcopy(model), data_train=data[0], data_val=data[1])[1]

    def test_the_default_keeps_no_candidates(self, small_model, small_data):
        history = self.run(small_model, small_data)
        assert all('candidates' not in history[i] for i in iteration_ids(history))

    def test_the_table(self, small_model, small_data):
        history = self.run(small_model, small_data, record_candidates=True)
        for i in iteration_ids(history):
            table = history[i]['candidates']
            n = n_edges_of(parent_of(history, i))
            assert set(table) == {'edges', 'val_loss', 'train_loss', 'network', 'removed_edge'}
            assert table['edges'].shape == (n, 2)  # candidate_fraction=1: every edge
            for column in [table['val_loss'], table['train_loss'], *table['network'].values(),
                           *table['removed_edge'].values()]:
                assert column.shape == (n,) and column.dtype == float and np.isfinite(column).all()
            assert list(table['network']) == DEFAULT_NETWORK_FEATURES
            assert list(table['removed_edge']) == DEFAULT_EDGE_FEATURES

    def test_the_winner_is_a_row_of_the_table_with_the_lowest_validation_loss(self, small_model, small_data):
        history = self.run(small_model, small_data, record_candidates=True)
        for i in iteration_ids(history):
            table, row = history[i]['candidates'], history[i]['winner_row']
            assert tuple(table['edges'][row]) == history[i]['winner_edge']
            assert table['val_loss'][row] == table['val_loss'].min()
            # the surviving reservoir is the winning candidate
            assert history[i]['state']['val_loss'] == pytest.approx(table['val_loss'][row])
            assert history[i]['state']['train_loss'] == pytest.approx(table['train_loss'][row])
            for name in DEFAULT_NETWORK_FEATURES:
                assert history[i]['state']['network'][name] == pytest.approx(table['network'][name][row])

    def test_the_removed_edge_of_a_candidate_is_its_row_in_the_parents_edge_table(self, small_model, small_data):
        history = self.run(small_model, small_data, record_candidates=True)
        for i in iteration_ids(history):
            parent, table = parent_of(history, i), history[i]['candidates']
            for row, (u, v) in enumerate(table['edges']):
                parent_row = int(np.flatnonzero((parent['edges'] == (u, v)).all(axis=1))[0])
                for name, column in parent['edge'].items():
                    assert table['removed_edge'][name][row] == column[parent_row]
                assert table['removed_edge']['weight'][row] == parent['weights'][u, v]

    def test_the_removed_edge_of_the_winner(self, small_model, small_data):
        for record in (False, True):
            history = self.run(small_model, small_data, record_candidates=record)
            for i in iteration_ids(history):
                parent, (u, v) = parent_of(history, i), history[i]['winner_edge']
                row = int(np.flatnonzero((parent['edges'] == (u, v)).all(axis=1))[0])
                assert history[i]['removed_edge'] == {name: column[row] for name, column in parent['edge'].items()}

    def test_the_network_features_of_a_candidate_are_those_of_its_reservoir(self, small_model, small_data):
        from pyreco.features import compute_network_features
        history = self.run(small_model, small_data, removals=1, record_candidates=True)
        table, parent = history[0]['candidates'], history['initial']
        for row in (0, len(table['edges']) // 2, len(table['edges']) - 1):
            u, v = table['edges'][row]
            candidate = copy.deepcopy(small_model)
            candidate.remove_reservoir_edges(edges=[(int(u), int(v))])
            candidate.fit(*small_data[0])
            expected = compute_network_features(candidate, small_data[0][0], DEFAULT_NETWORK_FEATURES)
            for name, value in expected.items():
                assert table['network'][name][row] == pytest.approx(value)
            assert table['val_loss'][row] == pytest.approx(candidate.evaluate(*small_data[1], metrics='mse')[0])
            assert table['train_loss'][row] == pytest.approx(candidate.evaluate(*small_data[0], metrics='mse')[0])

    def test_candidate_features_have_the_fixed_spectral_radius(self, small_data):
        X_train, y_train = small_data[0]
        model = RC(seed=0)
        model.add(InputLayer(input_shape=X_train.shape[1:]))
        model.add(RandomReservoirLayer(nodes=10, density=0.5, spec_rad=0.8, leakage_rate=0.3, seed=1))
        model.add(ReadoutLayer(y_train.shape[1:], fraction_out=0.9))
        model.compile(optimizer=RidgeSK(alpha=0.5), metrics=['mean_squared_error'])
        model.fit(X_train, y_train)

        history = self.run(model, small_data, removals=1, record_candidates=True, rescale_spec_rad='initial')
        assert history[0]['candidates']['network']['spectral_radius'] == pytest.approx(
            model.reservoir_layer.spec_rad, rel=1e-6)
        assert history[0]['state']['network']['spectral_radius'] == pytest.approx(
            model.reservoir_layer.spec_rad, rel=1e-6)

    def test_parallel_and_serial_record_the_same(self, small_model, small_data):
        serial = self.run(small_model, small_data, record_candidates=True, parallel=False)
        parallel = self.run(small_model, small_data, record_candidates=True, parallel=True)
        assert serial.keys() == parallel.keys()
        for i in iteration_ids(serial):
            a, b = serial[i]['candidates'], parallel[i]['candidates']
            assert (a['edges'] == b['edges']).all()
            for name in ('val_loss', 'train_loss'):
                assert a[name] == pytest.approx(b[name])
            for level in ('network', 'removed_edge'):
                for name in a[level]:
                    assert a[level][name] == pytest.approx(b[level][name])

    def test_the_table_stays_small(self, small_model, small_data):
        import pickle
        history = self.run(small_model, small_data, record_candidates=True)
        table = pickle.dumps([history[i]['candidates'] for i in iteration_ids(history)])
        n_candidates = sum(len(history[i]['candidates']['edges']) for i in iteration_ids(history))
        assert len(table) < 600 * n_candidates  # bytes per candidate: a few floats, not a matrix


class TestFeatures:
    """The named features of the initial reservoir and of the survivors."""

    @staticmethod
    def run(model, data, removals=3, **kwargs):
        return TestCandidates.run(model, data, removals, **kwargs)

    def test_the_default_features(self, small_model, small_data):
        history = self.run(small_model, small_data)
        for key in ['initial'] + iteration_ids(history):
            state = history[key] if key == 'initial' else history[key]['state']
            assert list(state['network']) == DEFAULT_NETWORK_FEATURES
            assert list(state['edge']) == DEFAULT_EDGE_FEATURES
            assert list(state['node']) == DEFAULT_NODE_FEATURES
        assert list(history[0]['removed_edge']) == DEFAULT_EDGE_FEATURES

    def test_the_activity_is_measured_on_the_training_inputs(self, small_model, small_data):
        from pyreco.features import FeatureContext
        history = self.run(small_model, small_data)
        activity = FeatureContext(small_model, small_data[0][0]).node_mean_sq_activity
        assert history['initial']['node']['node_mean_sq_activity'] == pytest.approx(activity)
        assert history['initial']['network']['mean_sq_activity'] == pytest.approx(activity.mean())
        assert (history['initial']['edge']['source_mean_sq_activity'] > 0).all()

    def test_the_features_of_the_survivor_are_measured_after_isolated_nodes_are_removed(
            self, isolatable_model, isolatable_data):
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'], min_num_edges=3,
                            remove_isolated_nodes=True, return_best_model=False, seed=0)
        history = pruner.prune(isolatable_model, *isolatable_data)[1]
        state = history[0]['state']
        assert state['weights'].shape[0] == 5 and history['initial']['weights'].shape[0] == 6
        assert state['node']['node_mean_sq_activity'].shape == (5,)
        assert history['initial']['node']['node_mean_sq_activity'].shape == (6,)
        assert state['network']['density'] == pytest.approx(n_edges_of(state) / (5 * 4))
        # the removed edge belongs to the parent reservoir with six nodes
        assert history[0]['winner_edge'] in {tuple(e) for e in history['initial']['edges']}

    def test_an_empty_list_measures_nothing(self, small_model, small_data):
        history = self.run(small_model, small_data, features=[], record_candidates=True)
        for key in ['initial'] + iteration_ids(history):
            state = history[key] if key == 'initial' else history[key]['state']
            assert state['network'] == {} and state['edge'] == {} and state['node'] == {}
        assert history[0]['removed_edge'] == {}
        assert history[0]['candidates']['network'] == {} and history[0]['candidates']['removed_edge'] == {}
        assert np.isfinite(history[0]['candidates']['val_loss']).all()

    def test_the_selection_decides_what_is_stored(self, small_model, small_data):
        history = self.run(small_model, small_data, features=['abs_weight', 'density', 'node_mean_sq_activity'],
                           record_candidates=True)
        state = history[0]['state']
        assert list(state['network']) == ['density'] and list(state['edge']) == ['abs_weight']
        assert list(state['node']) == ['node_mean_sq_activity']
        assert list(history[0]['candidates']['network']) == ['density']
        assert list(history[0]['candidates']['removed_edge']) == ['abs_weight']

    def test_a_custom_feature(self, small_model, small_data):
        from pyreco import features as F

        @F.network_feature('test_n_edges')
        def n_edges(ctx):
            return np.count_nonzero(ctx.weights)

        @F.node_feature('test_in_degree')
        def in_degree(ctx, v):
            return np.count_nonzero(ctx.weights[:, v])

        try:
            history = self.run(small_model, small_data, features=[n_edges, in_degree, 'weight'],
                               record_candidates=True)
        finally:
            del F._NETWORK_FEATURES['test_n_edges'], F._NODE_FEATURES['test_in_degree']
        assert history['initial']['network']['test_n_edges'] == n_edges_of(history['initial'])
        assert (history[0]['candidates']['network']['test_n_edges'] == n_edges_of(history['initial']) - 1).all()
        assert history[0]['state']['node']['test_in_degree'].sum() == n_edges_of(history[0]['state'])

    def test_the_activity_of_the_candidates_differs(self, small_model, small_data):
        history = self.run(small_model, small_data, features=['mean_sq_activity'], record_candidates=True)
        assert len(np.unique(history[0]['candidates']['network']['mean_sq_activity'])) > 1

    def test_the_default_run_is_small(self, small_model, small_data):
        import pickle
        history = self.run(small_model, small_data)
        with_candidates = self.run(small_model, small_data, record_candidates=True)
        assert len(pickle.dumps(history)) < len(pickle.dumps(with_candidates))

    def test_the_store_hook_gets_the_record_and_the_candidate_losses(self, small_model, small_data):
        seen = []

        class Light(EdgePruner):
            def _store_iteration(self, model, entry, candidate_val_losses):
                seen.append((entry['winner_edge'], len(candidate_val_losses), model.reservoir_layer.nodes))

        n_edges = _count_edges(small_model)
        pruner = Light(candidate_fraction=1.0, stopping_criterion=['min_edges'], min_num_edges=n_edges - 2,
                       return_best_model=False, seed=0)
        _, history = pruner.prune(copy.deepcopy(small_model), *small_data)
        assert set(history) == {'initial', 'features'}  # no iteration was kept
        assert [count for _, count, _ in seen] == [n_edges, n_edges - 1]


# Test edge cases

class TestEdgeCases:
    """Corner cases of the candidate selection."""


    def test_tiny_candidate_fraction_always_proposes_at_least_one_edge(self,
                                                                       small_model,
                                                                       small_data):
        # Even tiny candidate_fraction proposes at least one candidate per iteration
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=0.01, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        pruner.record_candidates = True
        _, history = pruner.prune(model=small_model, data_train=data_train, data_val=data_val)
        assert all(len(history[i]['candidates']['edges']) >= 1 for i in iteration_ids(history))
