"""
Capabilities to prune an existing RC model, i.e. try to cut reservoir nodes and improve 
performance while reducing the reservoir size
"""

import numpy as np
from pyreco.custom_models import RC


class NetworkPruner:
    # implements a pruning object for pyreco objects.

    def __init__(
        self,
        target_score: float = None,
        stop_at_minimum: bool = True,
        min_num_nodes: int = None,
        patience: int = None,
        prune_fraction: float = 0.1,
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

        - prune_fraction (float): number of randomly chosen reservoir nodes during
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

        if not isinstance(prune_fraction, float):
            raise TypeError("prune_fraction must be a float in (0, 1]")

        if prune_fraction <= 0 or prune_fraction > 1:
            raise ValueError("prune_fraction must be a float in (0, 1]")

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

        if self.patience is None:
            self.patience = int(n / 10)

        # Initialize stopping criteria values
        curr_score = model.evaluate(X=data_val[0], y=data_val[1])  # baseline score
        curr_num_nodes = model.reservoir_layer.nodes

        # Initialize a dict that stores all relevant information during pruning
        history = dict()
        history["num_nodes"] = [curr_num_nodes]
        history["score"] = [curr_score]
        history["graph_props"] = None  # graph property extractor
        history["node_props"] = None  # node property extractor
        history["pruned_node_props"] = None  # node property of the node that was pruned

        iter = 0
        while self._keep_pruning(
            iteration=iter,
            score=curr_score,
            num_nodes=curr_num_nodes,
        ):

            # propose_node_to_prune()
            # mask_input_layer(idx)
            # mask_output_layer(idx)

            # fit_model()
            # evaluate_model()

            iter += 1

        pass

    def _keep_pruning(self, iteration: int, score: float, num_nodes: int):

        if score > self.target_score:
            return True

        if num_nodes > self.min_num_nodes:
            return True

        return False
