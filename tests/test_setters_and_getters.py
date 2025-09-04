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
    x_train = np.expand_dims(x, axis=(0, 2))  # [n_batch, n_time, n_states]
    y_train = np.expand_dims(y, axis=(0, 2))

    input_shape = (x_train.shape[1], x_train.shape[2])
    output_shape = (y_train.shape[1], y_train.shape[2])

    # Build RC model
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(nodes=50, activation='tanh', fraction_input=0.5))
    model.add(ReadoutLayer(output_shape=output_shape))

    # Set scaling
    input_scaling = 2.0
    output_scaling = 0.5
    model.set_input_scaling(input_scaling)
    model.set_output_scaling(output_scaling)

    # Compile and fit with default alpha
    model.compile(optimizer='ridge', metrics=['mean_squared_error'])
    model.set_normalization(normalize_inputs=True, normalize_outputs=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)

    # Check stored values and prediction validity
    assert np.isclose(model.input_scaling, input_scaling)
    assert np.isclose(model.output_scaling, output_scaling)
    assert np.isfinite(y_pred).all()

    # Check other HPs
    model.set_hp(spec_rad=0.9)
    actual_spec_rad = np.max(np.abs(np.linalg.eigvals(model.reservoir_layer.weights)))
    assert np.isclose(model.get_hp()['spec_rad'], 0.9, atol=1e-6)
    assert np.isclose(actual_spec_rad, 0.9, atol=1e-2)

    model.set_hp(leakage_rate=0.15)
    assert np.isclose(model.get_hp()['leakage_rate'], 0.15)

    test_vec = np.array([0.5, -1.0, 0.0])
    model.set_hp(activation='tanh')
    assert np.allclose(model.reservoir_layer.activation_fun(test_vec), np.tanh(test_vec))
    model.set_hp(activation='sigmoid')
    assert np.allclose(model.reservoir_layer.activation_fun(test_vec), 1 / (1 + np.exp(-test_vec)))

    # -------- Test alpha setting and numerical effect using set_hp --------
    model.set_hp(alpha=1e-4)
    assert np.isclose(model.get_hp("alpha")["alpha"], 1e-4)

    model.fit(x_train, y_train)
    pred_low_alpha = model.predict(x_train)

    model.set_hp(alpha=1e1)
    assert np.isclose(model.get_hp("alpha")["alpha"], 1e1)

    model.fit(x_train, y_train)
    pred_high_alpha = model.predict(x_train)

    # Sanity check: predictions should be numerically different due to different regularization
    assert not np.allclose(pred_low_alpha, pred_high_alpha, atol=1e-2), \
        "Changing alpha via set_hp should affect predictions after refitting"