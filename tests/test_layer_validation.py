"""
Layers refuse invalid arguments when they are created, instead of building a model that silently
does something else (no recurrence, a frozen or unstable reservoir, an unknown activation, ...).
"""

import numpy as np
import pytest

from pyreco.layers import RandomReservoirLayer, ReadoutLayer


def _layer(**kwargs):
    options = dict(nodes=10, density=0.3, seed=1)
    options.update(kwargs)
    return RandomReservoirLayer(**options)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(nodes=2),
        dict(nodes=np.int64(10)),
        dict(density=1.0),
        dict(density=1),  # an integer is a number as well
        dict(density=np.float64(0.2)),
        dict(spec_rad=0.01),
        dict(spec_rad=1.5),  # larger than 1 is unusual, but valid
        dict(leakage_rate=1.0),
        dict(leakage_rate=1),
        dict(activation="sigmoid"),
        dict(fraction_input=1.0),
        dict(fraction_input=1),
        dict(init_res_sampling="ones"),
    ],
)
def test_valid_arguments_are_accepted(kwargs):
    _layer(**kwargs)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(nodes=10.5), TypeError, "nodes must be an integer"),
        (dict(nodes="10"), TypeError, "nodes must be an integer"),
        (dict(nodes=True), TypeError, "nodes must be an integer"),
        (dict(nodes=1), ValueError, "at least 2"),
        (dict(nodes=0), ValueError, "at least 2"),
        (dict(nodes=-5), ValueError, "at least 2"),
        (dict(density=0.0), ValueError, "density"),
        (dict(density=-0.1), ValueError, "density"),
        (dict(density=1.5), ValueError, "density"),
        (dict(density=np.nan), ValueError, "density"),
        (dict(density="0.1"), TypeError, "density"),
        (dict(spec_rad=0.0), ValueError, "spec_rad"),
        (dict(spec_rad=-0.5), ValueError, "spec_rad"),
        (dict(spec_rad=np.inf), ValueError, "spec_rad"),
        (dict(spec_rad=None), TypeError, "spec_rad"),
        (dict(leakage_rate=0.0), ValueError, "leakage_rate"),
        (dict(leakage_rate=2.5), ValueError, "leakage_rate"),
        (dict(leakage_rate=-0.1), ValueError, "leakage_rate"),
        (dict(activation="relu"), ValueError, "activation"),
        (dict(activation=None), ValueError, "activation"),
        (dict(fraction_input=0.0), ValueError, "fraction_input"),
        (dict(fraction_input=1.5), ValueError, "fraction_input"),
        (dict(fraction_input=-0.2), ValueError, "fraction_input"),
        (dict(fraction_input="all"), TypeError, "fraction_input"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, error, match):
    with pytest.raises(error, match=match):
        _layer(**kwargs)


@pytest.mark.parametrize("fraction", [0.0, 1.5, -0.1, np.nan])
def test_readout_fraction_must_be_in_the_unit_interval(fraction):
    with pytest.raises(ValueError, match="fraction_out"):
        ReadoutLayer((10, 1), fraction_out=fraction)
    with pytest.raises(TypeError, match="fraction_out"):
        ReadoutLayer((10, 1), fraction_out="half")


def test_readout_fractions_are_stored_as_floats():
    assert ReadoutLayer((10, 1), fraction_out=1).fraction_out == 1.0
    assert isinstance(ReadoutLayer((10, 1), fraction_out=np.float32(0.5)).fraction_out, float)
    layer = _layer(fraction_input=1, density=1, leakage_rate=1)
    assert all(isinstance(v, float) for v in (layer.fraction_input, layer.density, layer.leakage_rate))
    assert isinstance(_layer(nodes=np.int64(10)).nodes, int)


def test_a_valid_layer_is_unchanged_by_the_validation():
    # the generated network does not depend on the type of the arguments
    a = _layer(nodes=12, density=0.25)
    b = _layer(nodes=np.int64(12), density=np.float64(0.25))
    assert np.array_equal(a.weights, b.weights)
    assert np.isclose(a.spec_rad, 0.9)  # the default


def test_the_setters_use_the_same_activations():
    layer = _layer()
    layer.set_activation("sigmoid")
    with pytest.raises(ValueError):
        layer.set_activation("relu")
    with pytest.raises(ValueError):
        layer.set_leakage_rate(0.0)
