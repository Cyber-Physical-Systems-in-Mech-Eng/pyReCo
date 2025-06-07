import pytest
from pyreco.pruning import NetworkPruner


def test_network_pruner_initialization():
    # Test conflicting min_num_nodes and stop_at_minimum FIRST
    with pytest.raises(ValueError):
        NetworkPruner(min_num_nodes=10, stop_at_minimum=True)

    # Test valid initialization (min_num_nodes=3 is valid now)
    pruner = NetworkPruner(
        target_score=0.9, stop_at_minimum=False, min_num_nodes=3, patience=5
    )
    assert pruner.target_score == 0.9
    assert pruner.stop_at_minimum is False
    assert pruner.min_num_nodes == 3
    assert pruner.patience == 5

    # Test invalid target_score type
    with pytest.raises(TypeError):
        NetworkPruner(target_score="0.9", stop_at_minimum=False, min_num_nodes=3)

    # Test invalid stop_at_minimum type
    with pytest.raises(TypeError):
        NetworkPruner(stop_at_minimum="True", min_num_nodes=3)

    # Test invalid min_num_nodes type
    with pytest.raises(TypeError):
        NetworkPruner(min_num_nodes="10", stop_at_minimum=False)

    # Test invalid min_num_nodes value (should be >2)
    with pytest.raises(ValueError):
        NetworkPruner(min_num_nodes=2, stop_at_minimum=False)

    # Test invalid patience type
    with pytest.raises(TypeError):
        NetworkPruner(patience="5", stop_at_minimum=False, min_num_nodes=3)


if __name__ == "__main__":
    pytest.main()
