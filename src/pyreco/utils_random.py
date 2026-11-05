"""
Helpers for random number generation.

pyReCo never uses the global random state (neither ``np.random.*`` nor the stdlib
``random`` module). Every random draw comes from an explicit ``numpy.random.Generator``,
which is created from a ``seed`` argument (``None`` draws fresh entropy).
"""

from typing import Union

import numpy as np

RngLike = Union[None, int, np.integer, np.random.SeedSequence, np.random.Generator]


def make_rng(seed: RngLike = None) -> np.random.Generator:
    """
    Return a Generator for the given seed.

    A Generator is passed through unchanged (so that callers can share one stream); an
    int, a SeedSequence or None creates a new Generator.
    """
    return np.random.default_rng(seed)


def check_seed(seed) -> None:
    """Seeds given to models must be None or a non-negative integer."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be None or a non-negative integer.")
    if seed < 0:
        raise ValueError("seed must be None or a non-negative integer.")
