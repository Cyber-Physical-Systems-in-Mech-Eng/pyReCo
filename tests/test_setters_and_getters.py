import numpy as np
import pytest
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer

def test_setter_and_getters():
    # Generate sine and cosine signals
    omega = np.pi
    t = np.linspace(start=0, stop=3 * (2*np.pi/omega), num=300, endpoint=True)
    x = np.sin(omega * t)
    y = 2 * np.cos(omega * t)
    x_train = np.expand_dims(x, axis=(0,2))  # [n_batch, n_time, n_states]
    y_train = np.expand_dims(y, axis=(0,2))

    input_shape = (x_train.shape[1], x_train.shape[2])
    output_shape = (y_train.shape[1], y_train.shape[2])

    # Build RC model
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(nodes=50, activation='tanh', fraction_input=0.5))
    model.add(ReadoutLayer(output_shape))

    # Compile and fit
    model.compile(optimizer='ridge', metrics=['mean_squared_error'])
    model.fit(x_train, y_train)

    # ----- SET AND GET HP -----
    # 1. Set spec_rad and check actual spectral radius numerically
    new_spec_rad = 0.9
    model.set_hp(spec_rad=new_spec_rad)
    actual_spec_rad = np.max(np.abs(np.linalg.eigvals(model.reservoir_layer.weights)))
    assert np.isclose(model.get_hp()['spec_rad'], new_spec_rad, atol=1e-6), "Spec_rad not set correctly"
    assert np.isclose(actual_spec_rad, new_spec_rad, atol=1e-2), f"Actual spec_rad {actual_spec_rad} != set {new_spec_rad}"

    # 2. Set leakage_rate, check value and verify via a fake state update
    model.set_hp(leakage_rate=0.15)
    assert np.isclose(model.get_hp()['leakage_rate'], 0.15), "Leakage rate not set"

    # 3. Set activation, test numerically for both tanh and sigmoid
    model.set_hp(activation='tanh')
    test_vec = np.array([0.5, -1.0, 0.0])
    out1 = model.reservoir_layer.activation_fun(test_vec)
    assert np.allclose(out1, np.tanh(test_vec)), "tanh activation not correct"
    model.set_hp(activation='sigmoid')
    out2 = model.reservoir_layer.activation_fun(test_vec)
    assert np.allclose(out2, 1/(1+np.exp(-test_vec))), "sigmoid activation not correct"

    new_in_scal = 0.42
    model.set_hp(in_scal=new_in_scal)

    assert np.isclose(model.get_hp()['in_scal'], new_in_scal, atol=1e-6), "Input scaling value not set"

    # check that input weights were scaled accordingly
    max_abs_weight = np.max(np.abs(model.input_layer.weights))
    assert np.isclose(max_abs_weight, new_in_scal, atol=1e-2), (
        f"Max abs W_in {max_abs_weight} != expected in_scal {new_in_scal}"
    )