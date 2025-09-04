import numpy as np
import matplotlib.pyplot as plt
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer

def test_input_normalization_and_model_performance():
    # --- Generate synthetic data ---
    omega = np.pi
    t = np.linspace(0, 2 * np.pi, 300)
    x = np.stack([np.sin(omega * t), np.cos(omega * t)], axis=-1)  # [300, 2]
    y = 3 * np.sin(omega * t) + 1                                   # [300]

    x_train = np.expand_dims(x, axis=0)        # [1, 300, 2]
    y_train = np.expand_dims(y, axis=(0, 2))   # [1, 300, 1]

    # --- Set up RC model ---
    model = RC()
    model.add(InputLayer(input_shape=(300, 2)))
    model.add(RandomReservoirLayer(nodes=200))
    model.add(ReadoutLayer(output_shape=(300, 1)))

    # ✅ Toggle normalization here:
    use_input_norm = True
    use_output_norm = True
    model.set_normalization(normalize_inputs=use_input_norm, normalize_outputs=use_output_norm)
    model.set_input_scaling(1.0)

    model.compile(optimizer='ridge')
    model.fit(x_train, y_train)

    # --- Input normalization checks (only if enabled) ---
    if model.normalize_inputs:
        x_norm = (x_train - model.input_mean) / model.input_std
        x_denorm = x_norm * model.input_std + model.input_mean

        mean = x_norm.mean(axis=(0, 1))
        std = x_norm.std(axis=(0, 1))
        assert np.allclose(mean, 0, atol=0.01), f"Mean not ~0: {mean}"
        assert np.allclose(std, 1, atol=0.1), f"Std not ~1: {std}"
        assert np.allclose(x_train, x_denorm, atol=1e-6), "Denormalized input doesn't match original"
        print("✅ Input normalization and de-normalization successful")
        print("  Normalized mean:", mean)
        print("  Normalized std :", std)

    # --- Predict and evaluate ---
    y_pred = model.predict(x_train)
    rmse = np.sqrt(np.mean((y_pred - y_train)**2))
    print(f"✅ Model prediction RMSE: {rmse:.4f}")

    # --- Plot model output ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_train[0, :, 0], label='Ground Truth', linewidth=2)
    plt.plot(t, y_pred[0, :, 0], label='Model Prediction', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.title("RC Model Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_input_normalization_and_model_performance()
