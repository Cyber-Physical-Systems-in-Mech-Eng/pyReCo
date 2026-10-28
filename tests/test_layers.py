"""
Layers: arguments are checked when a layer is created (no silent models without recurrence, with a
frozen or unstable reservoir, or an unknown activation), and the reservoir weights can be replaced
from outside.
"""

import numpy as np
import pytest

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.utils_networks import compute_density, compute_spec_rad


# ------------------------------------------------------------------------- validation


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
    """Boundary values and numpy / integer numbers are valid arguments."""
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
    """Each invalid argument raises at construction, with the name of the argument in the message."""
    with pytest.raises(error, match=match):
        _layer(**kwargs)


@pytest.mark.parametrize("fraction", [0.0, 1.5, -0.1, np.nan])
def test_readout_fraction_must_be_in_the_unit_interval(fraction):
    with pytest.raises(ValueError, match="fraction_out"):
        ReadoutLayer((10, 1), fraction_out=fraction)
    with pytest.raises(TypeError, match="fraction_out"):
        ReadoutLayer((10, 1), fraction_out="half")


def test_numbers_are_stored_as_python_numbers():
    """Integers and numpy scalars are converted, so that later arithmetic and saving behave."""
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



# --------------------------------------------------------------------- set_weights


def _compiled_model(nodes=100):
    model = RC()
    model.add(InputLayer(input_shape=(1, nodes)))
    model.add(RandomReservoirLayer(nodes=nodes, density=0.1))
    model.add(ReadoutLayer(output_shape=(1, nodes)))
    model.compile()
    return model


def test_set_weights_updates_the_layer_properties():
    """density and spectral radius of the layer always describe the current weight matrix."""
    layer = _compiled_model().reservoir_layer
    assert layer._is_compiled is True
    assert np.isclose(layer.density, compute_density(layer.weights))
    assert np.isclose(layer.spec_rad, compute_spec_rad(layer.weights))

    new_weights = np.ones_like(layer.weights)  # fully connected: density 1
    layer.set_weights(new_weights)
    assert np.array_equal(layer.weights, new_weights)
    assert np.isclose(layer.density, 1.0)
    assert np.isclose(layer.spec_rad, compute_spec_rad(new_weights))


@pytest.mark.parametrize(
    "weights, error",
    [
        (np.ones((50, 50)), ValueError),  # other size than the 100 nodes
        (np.ones((100, 50)), ValueError),  # not square
        (np.ones((200, 200)), ValueError),
        ([1.0, 2.0, 3.0], TypeError),  # not an array
    ],
)
def test_set_weights_rejects_other_shapes_and_types(weights, error):
    layer = _compiled_model().reservoir_layer
    before = layer.weights.copy()
    with pytest.raises(error):
        layer.set_weights(weights)
    assert np.array_equal(layer.weights, before)
