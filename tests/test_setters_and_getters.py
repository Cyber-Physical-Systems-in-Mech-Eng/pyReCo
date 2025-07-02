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

    # Set known scaling factors
    input_scaling = 2.0
    output_scaling = 0.5
    model.set_input_scaling(input_scaling)
    model.set_output_scaling(output_scaling)

    # Compile and fit
    model.compile(optimizer='ridge', metrics=['mean_squared_error'])
    model.set_normalization(normalize_inputs=True, normalize_outputs=True)
    model.fit(x_train, y_train)

    # ----- Validate input/output scaling -----
    # Manually normalize and scale input
    x_mean = model.input_mean
    x_std = model.input_std
    x_manual = (x_train - x_mean) / x_std * input_scaling

    # Reservoir state projection (without internal dynamics) is not directly comparable
    # So we test consistency by predicting and redoing the inverse transform manually

    y_pred = model.predict(x_train)

    # Inverse transform of output:
    # predicted_raw = (normalized * output_std + output_mean)
    y_internal = y_pred * model.output_scaling  # undo model's own internal inverse scaling
    y_manual = (y_internal - model.output_mean) / model.output_std  # normalize manually

    # Check output stats
    mean_manual = y_manual.mean()
    std_manual = y_manual.std()

    assert np.isclose(model.input_scaling, input_scaling), "Input scaling not stored correctly"
    assert np.isclose(model.output_scaling, output_scaling), "Output scaling not stored correctly"
    assert np.isfinite(y_pred).all(), "Prediction contains non-finite values"

    # ----- Check HP setters/activation/parameters -----
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
