from abc import ABC, abstractmethod
import numpy as np


class NetworkInitializer:
    """
    network initializers.
    """

    def __init__(self, method: str = "random"):
        self.method = method

    def gen_initial_states(self, num_nodes: int) -> np.ndarray:
        """
        Generate initial states for the reservoir.

        Parameters:
        - num_nodes (int): The number of nodes in the reservoir.

        Returns:
        - np.ndarray: The initialized states.
        """
        # returns an array of length <num_nodes>
        # creates the entries based on different sampling methods
        # when not setting specific values, the range is normalized to abs(1)

        if self.method == "random":
            init_states = np.random.random(num_nodes)
        elif self.method == "random_normal":
            init_states = np.random.randn(num_nodes)
        elif self.method == "ones":
            init_states = np.ones(num_nodes)
        elif self.method == "zeros":
            init_states = np.zeros(num_nodes)
        else:
            raise ValueError(
                f"Sampling method {self.method} is unknown for generating initial reservoir states"
            )

        # normalize to max. absolute value of 1
        if self.method != "zeros":
            init_states = init_states / np.max(np.abs(init_states))

        return init_states
