import copy

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.edge_analyzer import EdgeAnalyzer
from pyreco.edge_pruning import EdgePruner
from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.node_analyzer import NodeAnalyzer
from pyreco.optimizers import RidgeSK
from pyreco.utils_data import sequence_to_sequence as seq_2_seq


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
    """
    Build an RC model with a hand-specified reservoir adjacency matrix and
    fixed input-receiving/readout node assignments.

    Uses a large, "safe" placeholder reservoir and overwrites its weights
    *before* calling compile(), rather than building a RandomReservoirLayer
    at the small target size directly: gen_ER_graph occasionally produces a
    near-zero spectral radius on tiny graphs, which blows up to NaN/inf when
    normalizing -- irrelevant here since the placeholder weights are
    immediately discarded, but it crashes construction before that happens.
    Pre-compile, set_weights() accepts any square shape, so this also
    resizes the reservoir down to mat's shape.
    """
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
    """
    Small (6-node), hand-built RC model. Nodes 0-4 form a connected,
    dynamically stable reservoir (spectral radius < 1) with input only at
    node 0 and readout only at nodes 1 and 2. Node 5 has zero edges from
    the start: isolated, not input-receiving, not readout.

    Built this way (rather than relying on edge pruning to organically
    create an isolated node) so isolated-node removal is exercised
    deterministically, independent of which edge the performance criterion
    happens to prune first.
    """
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


# Test __init__ function

class TestInit:

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
        assert pruner.edge_selection_strat == 'random_uniform_wo_repl'

    def test_default_analyzers_created(self):
        # Graph/node/edge analyzers are auto-created when not provided
        pruner = EdgePruner()
        assert isinstance(pruner.graph_analyzer, GraphAnalyzer)
        assert isinstance(pruner.node_analyzer, NodeAnalyzer)
        assert isinstance(pruner.edge_analyzer, EdgeAnalyzer)

    def test_custom_analyzers_stored(self):
        # Analyzers by users are stored (same object references)
        ga, na, ea = GraphAnalyzer(), NodeAnalyzer(), EdgeAnalyzer()
        pruner = EdgePruner(graph_analyzer=ga, node_analyzer=na, edge_analyzer=ea)
        assert pruner.graph_analyzer is ga
        assert pruner.node_analyzer is na
        assert pruner.edge_analyzer is ea

    def test_history_starts_empty(self):
        # History starts out as empty dict
        assert EdgePruner().history == {}

    def test_invalid_candidate_fraction_type(self):
        # candidate_fraction must be a float, not an int
        with pytest.raises(TypeError):
            EdgePruner(candidate_fraction=1)

    def test_invalid_candidate_fraction_too_low(self):
        # candidate_fraction must be > 0
        with pytest.raises(ValueError):
            EdgePruner(candidate_fraction=0.0)

    def test_invalid_candidate_fraction_too_high(self):
        # candidate_fraction must be <= 1
        with pytest.raises(ValueError):
            EdgePruner(candidate_fraction=1.5)

    def test_invalid_pruning_criterion_type(self):
        # pruning_criterion must be a string
        with pytest.raises(TypeError):
            EdgePruner(pruning_criterion=['performance'])

    def test_invalid_pruning_criterion_unknown(self):
        # pruning_criterion must be a recognized strategy name
        with pytest.raises(NotImplementedError):
            EdgePruner(pruning_criterion='unknown')

    def test_structural_pruning_blocked(self):
        # 'structure' is registered but not implemented yet 
        with pytest.raises(NotImplementedError, match="not implemented"):
            EdgePruner(pruning_criterion='structure')

    def test_invalid_stopping_criterion_type(self):
        # stopping_criterion must be a list
        with pytest.raises(TypeError):
            EdgePruner(stopping_criterion='patience')

    def test_empty_stopping_criterion_raises(self):
        # stopping_criterion must not be empty
        with pytest.raises(ValueError):
            EdgePruner(stopping_criterion=[])

    def test_invalid_stopping_criterion_value(self):
        # Each entry must be a recognized stopping criterion name
        with pytest.raises(NotImplementedError):
            EdgePruner(stopping_criterion=['unknown'])

    def test_invalid_min_num_nodes_type(self):
        # min_num_nodes must be an int
        with pytest.raises(TypeError):
            EdgePruner(min_num_nodes=3.0)

    def test_invalid_min_num_nodes_value(self):
        # min_num_nodes must be larger than 2
        with pytest.raises(ValueError):
            EdgePruner(min_num_nodes=2)

    def test_invalid_min_num_edges_type(self):
        # min_num_edges must be an int
        with pytest.raises(TypeError):
            EdgePruner(min_num_edges=2.0)

    def test_invalid_min_num_edges_value(self):
        # min_num_edges must be >= 0
        with pytest.raises(ValueError):
            EdgePruner(min_num_edges=-1)

    def test_min_num_edges_zero_allowed(self):
        # 0 is a valid (boundary) value for min_num_edges
        assert EdgePruner(min_num_edges=0).min_num_edges == 0

    def test_invalid_patience_type(self):
        # patience must be an int
        with pytest.raises(TypeError):
            EdgePruner(patience=1.0)

    def test_invalid_performance_criterion_type(self):
        # performance_criterion must be a string
        with pytest.raises(TypeError):
            EdgePruner(performance_criterion=1)

    def test_invalid_performance_criterion_unknown(self):
        # performance_criterion must be a recognized loss metric name
        with pytest.raises(ValueError):
            EdgePruner(performance_criterion='not_a_metric')

    def test_invalid_structural_criterion_type(self):
        # structural_criterion must be a string
        with pytest.raises(TypeError):
            EdgePruner(structural_criterion=1)

    def test_invalid_structural_criterion_unknown(self):
        # structural_criterion must be a recognized edge property name
        with pytest.raises(ValueError):
            EdgePruner(structural_criterion='not_a_property')

    def test_invalid_metrics_type(self):
        # metrics must be a list or a string
        with pytest.raises(TypeError):
            EdgePruner(metrics=1)

    def test_invalid_metrics_list_element_type(self):
        # Every entry in metrics list must be string
        with pytest.raises(TypeError):
            EdgePruner(metrics=['mse', 1])

    def test_invalid_metrics_unknown_value(self):
        # Every metric name in list must be recognized
        with pytest.raises(ValueError):
            EdgePruner(metrics=['not_a_metric'])

    def test_invalid_return_best_model_type(self):
        # return_best_model must be a boolean
        with pytest.raises(TypeError):
            EdgePruner(return_best_model='yes')

    def test_invalid_graph_analyzer_type(self):
        # graph_analyzer, when given, must be GraphAnalyzer instance
        with pytest.raises(TypeError):
            EdgePruner(graph_analyzer='not an analyzer')

    def test_invalid_node_analyzer_type(self):
        # node_analyzer, when given, must be NodeAnalyzer instance
        with pytest.raises(TypeError):
            EdgePruner(node_analyzer='not an analyzer')

    def test_invalid_edge_analyzer_type(self):
        # edge_analyzer, when given, must be EdgeAnalyzer instance
        with pytest.raises(TypeError):
            EdgePruner(edge_analyzer='not an analyzer')

    def test_invalid_remove_isolated_nodes_type(self):
        # remove_isolated_nodes must be boolean
        with pytest.raises(TypeError):
            EdgePruner(remove_isolated_nodes='yes')

    def test_invalid_parallel_type(self):
        # parallel must be  boolean
        with pytest.raises(TypeError):
            EdgePruner(parallel='yes')


# Test prune() parameter validation 

class TestPruningParams:

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
        assert sorted(history.keys()) == list(range(len(history)))

    def test_history_entry_has_expected_top_level_keys(self, small_model, small_data):
        # Iteration's history entry exposes exactly these top-level keys
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        assert set(history[0].keys()) == {
            'starting_model', 'candidates', 'winner', 'removed_nodes', 'final_model'
        }

    def test_starting_and_final_model_have_expected_keys(self, small_model, small_data):
        # starting_model/final_model snapshots both expose the same set of fields
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        expected = {
            'weights', 'input_nodes', 'readout_nodes', 'loss', 'num_nodes',
            'num_edges', 'metrics', 'graph_props',
        }
        assert set(history[0]['starting_model'].keys()) == expected
        assert set(history[0]['final_model'].keys()) == expected

    def test_candidates_entry_structure(self, small_model, small_data):
        # Each candidate is keyed by (u, v) edge tuple, with its score,
        #   edge_props, graph_props_after 
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        candidates = history[0]['candidates']
        assert len(candidates) > 0
        for edge, props in candidates.items():
            assert isinstance(edge, tuple) and len(edge) == 2
            assert set(props.keys()) == {'score', 'edge_props', 'graph_props_after'}

    def test_winner_is_among_that_iterations_candidates(self, small_model, small_data):
        # Pruned edge (winner candidate) is always among iteration's candidates
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for entry in history.values():
            assert entry['winner'] in entry['candidates']

    def test_winner_has_lowest_score_among_candidates(self, small_model, small_data):
        # Winner is candidate with lowest score in iteration
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for entry in history.values():
            candidates = entry['candidates']
            winner_score = candidates[entry['winner']]['score']
            assert winner_score == min(c['score'] for c in candidates.values())

    def test_winner_edge_is_zeroed_in_final_snapshot(self, small_model, small_data):
        # Comparing starting and final weight snapshot for iteration, the winner edge
        #   must go from nonzero to zero
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for entry in history.values():
            u, v = entry['winner']
            assert entry['starting_model']['weights'][u, v] != 0
            assert entry['final_model']['weights'][u, v] == 0

    def test_edge_count_decreases_by_one_per_iteration(self, small_model, small_data):
        # One edge is removed per completed iteration
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for entry in history.values():
            assert entry['final_model']['num_edges'] == \
                entry['starting_model']['num_edges'] - 1

    def test_loss_continuity_between_iterations(self, small_model, small_data):
        # Iteration's final loss is next iteration's starting loss
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        for it in range(len(history) - 1):
            assert history[it]['final_model']['loss'] == \
                pytest.approx(history[it + 1]['starting_model']['loss'])

    def test_stops_exactly_at_min_edges(self, small_model, small_data):
        # Pruning stops when edge count reaches min_num_edges
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        last_iter = max(history.keys())
        assert history[last_iter]['final_model']['num_edges'] == 8

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
        losses = [history[it]['final_model']['loss'] for it in sorted(history.keys())]
        assert all(losses[i + 1] <= losses[i] + 1e-9 for i in range(len(losses) - 1))

    def test_patience_n_tolerates_at_most_n_trailing_non_improvements(self, small_model,
                                                                      small_data):
        # patience N allows N consecutive non-improving iterations before pruning stops
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['patience'],
                            patience=2)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        losses = [history[it]['final_model']['loss'] for it in sorted(history.keys())]
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
        last_iter = max(history.keys())
        assert history[last_iter]['final_model']['num_edges'] == 6

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
        assert len(history) > 0
        last_iter = max(history.keys())
        assert history[last_iter]['final_model']['num_edges'] >= 2


# Test return_best_model 

class TestReturnBestModel:

    def test_true_picks_lowest_recorded_loss(self, small_model, small_data):
        # return_best_model=True returns model with the lowest recorded loss
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['patience'],
                            patience=2, return_best_model=True)
        model, history = pruner.prune(model=small_model, data_train=data_train,
                                      data_val=data_val)
        final_loss = model.evaluate(x=data_val[0], y=data_val[1], metrics='mse')[0]
        all_losses = [history[0]['starting_model']['loss']] + \
            [history[it]['final_model']['loss'] for it in sorted(history.keys())]
        assert final_loss <= min(all_losses) + 1e-6

    def test_false_returns_last_iteration_model(self, small_model, small_data):
        # return_best_model=False returns the last iteration's model
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=1.0, stopping_criterion=['patience'],
                            patience=2, return_best_model=False)
        model, history = pruner.prune(model=small_model, data_train=data_train,
                                      data_val=data_val)
        last_iter = max(history.keys())
        expected_edges = history[last_iter]['final_model']['num_edges']
        assert _count_edges(model) == expected_edges


# Test isolated node handling

class TestIsolatedNodes:

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
        assert history[0]['final_model']['num_nodes'] == 5
        # readout nodes [1, 2] are both below removed index (5)
        #   -> ids must be unaffected by the removal
        assert list(history[0]['final_model']['readout_nodes']) == [1, 2]

        # Node count never goes up once reduced 
        node_counts = [history[0]['final_model']['num_nodes']]
        for it in sorted(history.keys())[1:]:
            assert history[it]['starting_model']['num_nodes'] == node_counts[-1]
            node_counts.append(history[it]['final_model']['num_nodes'])
        assert all(node_counts[i + 1] <= node_counts[i] for i in
                   range(len(node_counts) - 1))
        assert node_counts[-1] <= 5

        assert pruned_model.reservoir_layer.nodes == node_counts[-1]


# Test parallel vs serial

class TestParallel:

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

        assert history_serial[0]['winner'] == history_parallel[0]['winner']
        assert history_serial[0]['final_model']['loss'] == \
            pytest.approx(history_parallel[0]['final_model']['loss'])


# Test edge cases

class TestEdgeCases:

    def test_tiny_candidate_fraction_always_proposes_at_least_one_edge(self,
                                                                       small_model,
                                                                       small_data):
        # Even tiny candidate_fraction proposes at least one candidate per iteration
        data_train, data_val = small_data
        pruner = EdgePruner(candidate_fraction=0.01, stopping_criterion=['min_edges'],
                            min_num_edges=8)
        _, history = pruner.prune(model=small_model, data_train=data_train,
                                  data_val=data_val)
        assert all(len(entry['candidates']) >= 1 for entry in history.values())
