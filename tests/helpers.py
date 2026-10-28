"""
Builders shared by the tests, so that a small reservoir model and some data do not have to be
set up in every file.

Import them with ``from helpers import build_rc, make_xy`` (pytest puts the test directory on the
path). A test that needs an unusual layout builds its model by hand instead of adding options here.
"""

import numpy as np

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer


def make_xy(n=6, T=40, n_in=2, K=None, n_out=1, seed=0, loc=0.0, scale=1.0):
    """
    Inputs x of shape (n, T, n_in) and the instantaneous nonlinear targets y of shape (n, K, n_out).

    With f(t) = tanh(x_0(t)) + 0.3 x_1(t) (just tanh(x_0) for one input channel), the target
    channel c is (1 + c / n_out) f(t) on the last K time steps, i.e. y is a function of the input
    at the same time step and needs no memory. `loc` and `scale` shift and scale the normal inputs.
    """
    K = T if K is None else K
    x = np.random.default_rng(seed).normal(loc=loc, scale=scale, size=(n, T, n_in))
    f = np.tanh(x[:, -K:, :1]) + (0.3 * x[:, -K:, 1:2] if n_in > 1 else 0.0)
    return x, np.concatenate([f] * n_out, axis=-1) * np.linspace(1.0, 2.0, n_out)


def build_rc(
    T=40,
    K=None,
    n_in=2,
    n_out=1,
    nodes=40,
    density=0.3,
    seed=None,
    layer_seed=None,
    input_layer=None,
    layer=None,
    readout=None,
    compiled=True,
    **compile_kwargs,
):
    """
    A reservoir computer for inputs (T, n_in) and targets (K, n_out), K defaulting to T.

    `seed` is the seed of the model, `layer_seed` the one of the reservoir network (both None: random).
    `input_layer`, `layer` and `readout` are extra keyword arguments of the three layers,
    `compile_kwargs` go to `compile()`, which is skipped for ``compiled=False``.
    """
    K = T if K is None else K
    model = RC(seed=seed)
    model.add(InputLayer(input_shape=(T, n_in), **(input_layer or {})))
    model.add(RandomReservoirLayer(nodes=nodes, density=density, seed=layer_seed, **(layer or {})))
    model.add(ReadoutLayer(output_shape=(K, n_out), **(readout or {})))
    if compiled:
        model.compile(**compile_kwargs)
    return model
