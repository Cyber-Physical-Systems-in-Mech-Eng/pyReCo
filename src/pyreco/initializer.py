from abc import ABC, abstractmethod
import numpy as np
from typing import Union

from pyreco.utils_random import RngLike, make_rng


class NetworkInitializer:
    r"""
    Initializer of the reservoir state :math:`r(0)` (and other arrays of random numbers).

    All methods except ``"zeros"`` are divided by their largest absolute value, so that
    :math:`\max_u |r_u(0)| = 1`.

    Parameters
    ----------
    method : {"random", "random_normal", "ones", "zeros"}
        Entries uniform on :math:`[0, 1)`, standard normal, all ones, or all zeros.
    rng : Generator, int or None
        Source of randomness. A Generator is used as is, an int seeds a new one, None draws fresh entropy.
    """

    def __init__(self, method: str = "random", rng: RngLike = None):
        self.method = method
        self.rng = make_rng(rng)

    def gen_initial_states(self, shape: Union[tuple, list]) -> np.ndarray:
        """
        Generate initial states for the reservoir.

        Parameters
        ----------
        shape : int, tuple or list
            The shape of the array to generate.

        Returns
        -------
        ndarray
            The initialized array of the given shape.

        Raises
        ------
        ValueError
            For an unknown method.
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
            init_states = self.rng.random(shape)
        elif self.method == "random_normal":
            init_states = self.rng.standard_normal(shape)
        elif self.method == "ones":
            init_states = np.ones(shape)
        elif self.method == "zeros":
            init_states = np.zeros(shape)
        else:
            raise ValueError(
                f"Sampling method {self.method} is unknown for generating initial reservoir states"
            )

        # normalize to max. absolute value of 1
        if self.method != "zeros":
            init_states = init_states / np.max(np.abs(init_states))

        return init_states
