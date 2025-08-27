"""
Some testing use case for the CustomModel API of pyreco
"""

from matplotlib import pyplot as plt
from pyreco.custom_models import AutoRC as RC
from pyreco.layers import InputLayer, ReadoutLayer, FeedbackLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import sequence_to_sequence
from pyreco.optimizers import RidgeSK

"""
Classic RC built on random networks. 
Use case: predict a sine signal into the future (n_time) samples from the past are mapped to (n_time) samples in the 
future
"""

# some testing data: predict a sine signal from a sequence of 20 function values
A_train, A_test, B_train, B_test = sequence_to_sequence(
    name="sine_pred",
    n_states=1,
    n_batch=50,
    n_time=201
)

###generate y_train and y_test as previous time step of the input series
X_train = A_train#[:, 1:, :]
X_test = A_test#[:, 1:, :]
### remove last time step from X_train and X_test
y_train = A_train#[:, :-1, :]
y_test = A_test#[:, :-1, :]

print(f"Generated training data shapes: {X_train.shape}, {y_train.shape}")
print(f"Generated test data shapes: {X_test.shape}, {y_test.shape}")

# ##plot x and y
# plt.figure()
# for i in range(X_train.shape[0]):
#     plt.subplot(X_train.shape[0], 1, i + 1)
#     plt.plot(X_train[i, :, 0], color="k", alpha=0.3)
#     plt.plot(y_train[i, :, 0], color="b", alpha=0.3)
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
# plt.tight_layout()
# plt.show()

# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# set the dimensions (need to be [n_batch, n_time, n_states])
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(FeedbackLayer(feedback_shape=output_shape))
model_rc.add(
    RandomReservoirLayer(
        nodes=100, density=0.1, spec_rad=1.0, activation="tanh", leakage_rate=0.2, fraction_input=0.75),
)
model_rc.add(ReadoutLayer(output_shape, fraction_out=0.90))

# Compile the model
optim = RidgeSK(alpha=1.0)
model_rc.AutoRC_compile(
    optimizer=optim,
    metrics=["mean_squared_error"],
    discard_transients=10
)

# Train the model
model_rc.fit(X_train, y_train)

# print(model_rc.input_layer.weights)
# Make predictions for new data
y_pred = model_rc.predict(X_test)

# autoRC predictions
# X_test = X_train[:-2, :, :]  # use all but last time step for prediction
# y_test = y_train[:-2, :, :]  # use all but last time step for prediction
res_states, y_pred, input_weights, feedback_weights, readout_weights = model_rc.AutoRC_predict(X_test, fb_scale=1.0, T_run=3000)

print(f"shape of input weights: {input_weights.shape}")
print(f"shape of feedback weights: {feedback_weights.shape}")
print(f"shape of readout weights: {readout_weights.shape}")

print(f"input weights:\n{input_weights[:5,:]}\n\n"
      f"feedback weights:\n{feedback_weights[:5,:]}\n\n"
      f"readout weights:\n{readout_weights[:5,:]}")

print(f"shape of predictions on test set: {y_pred.shape}")

# Evaluate the model
loss_rc = model_rc.evaluate(X_test, y_test, metrics=["mae"])
print(f"Test model loss: {loss_rc}")

# plot predictions vs. ground truth
# r2_scatter(y_true=y_test, y_pred=y_pred)


# plot predictions
plt.figure()
### loop over batch samples in X_test
plt.title("Autonomously running RC predictions")
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[2]):
        plt.subplot(X_test.shape[0], X_test.shape[2], i * X_test.shape[2] + j + 1)
        plt.plot(X_test[i, :, j], color="gray", alpha=1, label="input sequence")
        plt.plot(y_test[i, :, j], color="blue", lw=0.75, label="true target")
        plt.plot(y_pred[i, :, j], color="red", lw=0.75, label="predicted")
        plt.xlabel("time")
        plt.ylabel(f"amplitude of state {j}")
    plt.legend()
plt.show()




"""
k-fold cross-validation of the model: estimate bias and variance of the model
"""
# from pyreco.cross_validation import cross_val

# val, mean, std_dev = cross_val(model_rc, X_train, y_train, n_splits=5, metric=["mse"])
# print(f"Cross-Validation MSE: {val}")
# print(f"Mean MSE: {mean:.3f}")
# print(f"Standard Deviation of MSE: {std_dev:.3f}")
