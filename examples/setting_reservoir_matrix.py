"""
Assigning a custom reservoir matrix to a reservoir computing model.
"""
import numpy as np
from matplotlib import pyplot as plt
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence

"""
Classic RC built on random networks. 

Use case: predict a sine signal into the future (n_time) samples from the past are mapped to (n_time) samples in the 
future
"""

# some testing data: predict a sine signal from a sequence of 20 function values
X_train, X_test, y_train, y_test = sequence_to_sequence(
    name="sine_to_cosine",
    n_states=1,
    n_batch=20,
    n_time=50,
)

# set the dimensions (need to be [n_batch, n_time, n_states])
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

model = RC()
model.add(InputLayer(input_shape=input_shape))
model.add(
    RandomReservoirLayer(
        nodes=200, density=0.1, activation="tanh", leakage_rate=0.1, fraction_input=0.5
    ),
)
model.add(ReadoutLayer(output_shape, fraction_out=0.99))

# Compile the model
model.compile(discard_transients=10)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
loss_rc = model.evaluate(X_test, y_test, metrics=["mae"])
print(f"Test model loss: {loss_rc}")

# Make predictions for new data
y_pred = model.predict(X_test)
print(f"shape of predictions on test set: {y_pred.shape}")


"""
Now overwrite the reservoir matrix with a custom one.
"""

weights_old = model.reservoir_layer.weights
spec_rad_old = model.reservoir_layer.spec_rad
density_old = model.reservoir_layer.density

# set the new reservoir weights
weights_new = np.random.randn(200, 200) * 0.1
model.reservoir_layer.set_weights(weights_new)

# we can now also set the spectral radius
model.reservoir_layer.set_spec_rad(0.456789)
spec_rad_new = model.reservoir_layer.spec_rad
density_new = model.reservoir_layer.density

# Train the new model
model.fit(X_train, y_train)

# Evaluate the new model
loss_rc_new = model.evaluate(X_test, y_test, metrics=["mae"])
print(f"Test model loss: {loss_rc}")

# Make predictions for new data
y_pred_new = model.predict(X_test)
print(f"shape of predictions on test set: {y_pred.shape}")

"""
Visual comparison of the predictions with the old and new reservoir weights.
"""

print(f"old reservoir density: {density_old:.3f}, spectral radius: {spec_rad_old:.5f}")
print(f"new reservoir density: {density_new:.3f}, spectral radius: {spec_rad_new:.5f}")

# as we have discarded 10 transients, we can only compare the predictions after that
y_test = y_test[:, 10:, :]
X_test = X_test[:, 10:, :]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test.flatten(), y_pred.flatten(), label="old weights", color='red')
plt.scatter(y_test.flatten(), y_pred_new.flatten(), label="new weights", color="green")
plt.xlabel("ground truth")
plt.ylabel("predictions")
plt.title("Predictions vs. Ground Truth")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X_test[0, :, 0], color="gray", alpha=0.3)
plt.plot(X_test[0, :, 0], marker=".", color="black", label="input sequence")
plt.plot(y_test[0, :, 0], marker="o", color="blue", label="true target"
)
plt.plot(y_pred[0, :, 0], marker="x", color="red", 
         label=f"predicted (old weights, density={density_old:.3f}, spectral radius={spec_rad_old:.3f})")
plt.plot(y_pred_new[0, :, 0], marker="v", color="green", 
         label=f"predicted (new weights, density={density_new:.3f}, spectral radius={spec_rad_new:.3f})")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.legend()
plt.title("sequence to scalar prediction")
plt.show()

# figure comparing the network weights
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(weights_old, cmap="viridis", aspect="auto")
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Reservoir Weights (Old)")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(weights_new, cmap="viridis", aspect="auto")
plt.title("Reservoir Weights (New)")
plt.gca().set_aspect("equal", adjustable="box")
plt.colorbar()
plt.tight_layout()
plt.show()



