"""
Capabilities to prune an existing RC model, i.e. try to cut reservoir nodes and improve 
performance while reducing the reservoir size
"""

from pyreco.custom_models import RC


class NetworkPruner:
    # implements a pruning object for pyreco objects.

    def __init__(
        self,
        target_score: float = None,
        stop_at_minimum: bool = True,
        min_num_nodes: int = None,
        patience: int = None,
    ):
        # initializer for the pruning class.

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

        # Additional sanity checks: logical constraints on the input parameters
        if self.min_num_nodes is not None and self.stop_at_minimum:
            raise ValueError("min_num_nodes conflicts with stop_at_minimum set to True")

        # Assigning the parameters to instance variables
        self.target_score = target_score
        self.stop_at_minimum = stop_at_minimum
        self.min_num_nodes = min_num_nodes
        self.patience = patience

    def prune(self, model: RC, data_train: tuple, data_val: tuple):
        # prune a given model by removing nodes.

        N = model.reservoir_layer.nodes  # number of initial reservoir nodes

        if self.patience is None:
            self.patience = int(N / 10)

        pass
