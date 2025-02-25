from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class NetworkInitializer:
    """
    network initializers.
    """

    def __init__(self, method: str = "random"):
        self.method = method

    def gen_initial_states(self, shape: Union[tuple, list]) -> np.ndarray:
        """
        Generate initial states for the reservoir.

        Parameters:
        - shape (tuple, list): The shape of array to generate

        Returns:
        - np.ndarray: The initialized array
        """
        # returns an array of shape <shape>
        # creates the entries based on different sampling methods
        # when not setting specific values, the range is normalized to abs(1)

        # Convert shape to tuple if it is a scalar or list
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)

        if self.method == "random":
            init_states = np.random.random(*shape)
        elif self.method == "random_normal":
            init_states = np.random.randn(*shape)
        elif self.method == "ones":
            init_states = np.ones(*shape)
        elif self.method == "zeros":
            init_states = np.zeros(*shape)
        else:
            raise ValueError(
                f"Sampling method {self.method} is unknown for generating initial reservoir states"
            )

        # normalize to max. absolute value of 1
        if self.method != "zeros":
            init_states = init_states / np.max(np.abs(init_states))

        return init_states
