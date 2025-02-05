"""
Minimal testing example for predicting a sine function (discrete time setting)

x(t) -> x(t+dt)

"""

import numpy as np
from matplotlib import pyplot as plt

"""
generate some training and testing data from the sine function
"""

omega = np.pi
t = np.linspace(start=0, stop=5 * omega, num=51, endpoint=True)
signal = np.sin(omega * t)

# convert into inputs (function at current time point) and outputs (function at next time point)
x = signal[:-1]
y = signal[1:]

# train-test split 80%
ratio = 0.8
n = len(x)
shuffle_idx = np.random.choice(n, size=n, replace=False)
train_idx, test_idx = shuffle_idx[: int(n * ratio)], shuffle_idx[int(n * ratio) :]

# split data and create data shape [n_batch, n_timesteps, n_features]
X_train = np.expand_dims(x[train_idx], axis=(1, -1))
X_test = np.expand_dims(x[test_idx], axis=(1, -1))
y_train = np.expand_dims(y[train_idx], axis=(1, -1))
y_test = np.expand_dims(y[test_idx], axis=(1, -1))
print(f"shape of training data: {X_train.shape}, shape of test data: {y_test.shape}")

"""
Modeling using feed-forward neural nets in TensorFlow (sequential API)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model_ann = Sequential()  # Instantiate Sequential model
model_ann.add(Input(shape=(1,)))  # Add input layer
model_ann.add(Dense(units=100, activation="sigmoid"))  # Add hidden Dense layer
model_ann.add(Dense(units=1, activation="linear"))  # Add output layer

model_ann.compile(optimizer="adam", loss="mean_squared_error")  # Compile the model
hist = model_ann.fit(
    X_train, y_train, epochs=500, batch_size=64
)  # Train the model for 1000 epochs
loss_ann = model_ann.evaluate(X_test, y_test)  # Evaluate the model
print(f"\n\n Dense Neural Network: test set loss: {loss_ann}")
y_pred_ann = model_ann.predict(X_test)  # Make predictions for new data


"""
Modeling using reservoir computing built on the pyreco library (ours)
"""
from pyreco.models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

model_rc = RC()
model_rc.add(InputLayer(input_shape=(1, 1)))
model_rc.add(
    RandomReservoirLayer(
        nodes=100,
        density=0.1,
        activation="sigmoid",
        leakage_rate=0.2,
        fraction_input=0.5,
    )
)
model_rc.add(ReadoutLayer(output_shape=(1, 1), fraction_out=0.6))

# Compile the model
model_rc.compile(optimizer="ridge", metrics=["mean_squared_error"])

model_rc.fit(
    X_train, y_train
)  # , epochs=500)              # Train the model for 1000 epochs
y_pred_rc = model_rc.predict(X_test)  # Make predictions for new data

loss_rc = model_rc.evaluate(X_test, y_test)  # Evaluate the model
print(f"\n\n Reservoir Computer: test set loss: {loss_rc[0]}")
