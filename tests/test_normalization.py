import numpy as np
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer

def test_input_output_normalization():
    # Generate synthetic input/output with known stats
    omega = np.pi
    t = np.linspace(0, 2 * np.pi, 300)
    x = np.stack([np.sin(omega * t), np.cos(omega * t)], axis=-1)  # [300, 2]
    y = 3 * np.sin(omega * t) + 1                                   # [300]

    x_train = np.expand_dims(x, axis=0)       # [1, 300, 2]
    y_train = np.expand_dims(y, axis=(0, 2))  # [1, 300, 1]

    # Build RC model
    model = RC()
    model.add(InputLayer(input_shape=(300, 2)))
    model.add(RandomReservoirLayer(nodes=100, activation='tanh'))
    model.add(ReadoutLayer(output_shape=(300, 1)))

    # Enable normalization and scaling
    model.set_normalization(True, True)
    model.set_input_scaling(1.0)
    model.set_output_scaling(1.0)

    # Compile and train
    model.compile(optimizer='ridge', metrics=['mean_squared_error'])
    model.fit(x_train, y_train)

    # --- Check normalization numerically ---

    # Manually normalize x and y using stored stats
    x_norm = (x_train - model.input_mean) / model.input_std
    y_norm = (y_train - model.output_mean) / model.output_std

    # Input checks
    assert x_norm.shape == x_train.shape
    assert np.allclose(x_norm.mean(axis=(0, 1)), 0, atol=0.1), "Input mean not ~0"
    assert np.allclose(x_norm.std(axis=(0, 1)), 1, atol=0.1), "Input std not ~1"

    # Output checks
    assert y_norm.shape == y_train.shape
    assert np.allclose(y_norm.mean(axis=(0, 1)), 0, atol=0.1), "Output mean not ~0"
    assert np.allclose(y_norm.std(axis=(0, 1)), 1, atol=0.1), "Output std not ~1"

    # --- Predict and check denormalization ---

    y_pred = model.predict(x_train)
    assert y_pred.shape == y_train.shape, "Prediction shape mismatch"
    assert np.all(np.isfinite(y_pred)), "Prediction contains NaNs or infs"

    # Compare std with relative error
    pred_std = y_pred.std()
    true_std = y_train.std()
    rel_error = np.abs(pred_std - true_std) / true_std
    assert rel_error < 0.3, f"Relative std error too high: {rel_error:.3f}"

    # Also compare mean with relative error
    pred_mean = y_pred.mean()
    true_mean = y_train.mean()
    rel_mean_error = np.abs(pred_mean - true_mean) / np.abs(true_mean)
    assert rel_mean_error < 0.1, f"Relative mean error too high: {rel_mean_error:.3f}"
