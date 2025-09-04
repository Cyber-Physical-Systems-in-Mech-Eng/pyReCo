"""
Some testing use case for the CustomModel API of pyreco
"""

import numpy as np
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.utils_data import sequence_to_scalar
from pyreco.optimizers import RidgeSK

"""
Classic RC built on random networks. 

Use case: predict a sine signal into the future (n_time) samples from the past are mapped to (n_time) samples in the 
future
"""

# some testing data: predict a sine signal from a sequence of 20 function values
X_train, X_test, y_train, y_test = sequence_to_scalar(
    name="sine_prediction",
    n_states=1,
    n_batch=200,
    n_time_in=20,
)

# set the dimensions (need to be [n_batch, n_time, n_states])
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(
    RandomReservoirLayer(
        nodes=200, density=0.1, activation="tanh", leakage_rate=0.1, fraction_input=0.5
    ),
)
model_rc.add(ReadoutLayer(output_shape, fraction_out=0.99))

# Compile the model
optim = RidgeSK(alpha=0.5)
model_rc.compile(
    optimizer=optim,
    metrics=["mean_squared_error"],
)

# Train the model
model_rc.fit(X_train, y_train)

# Make predictions for new data
y_pred = model_rc.predict(X_test)
print(f"shape of predictions on test set: {y_pred.shape}")

# Evaluate the model
loss_rc = model_rc.evaluate(X_test, y_test, metrics=["mae"])
print(f"\nTest model loss (old input weights): {loss_rc[0]:.6f}")


# ""
# Now re-set the read-in weights
# ""

# check _connect_input_to_reservoir(self) in custom_models.py
# to see how we set the input weights.
new_input_weights = np.random.randn(200, 1)


#
# model_rc.input_layer.weights = new_input_weights  # works as well
model_rc._set_readin_weights(new_input_weights)

# Train the model
model_rc.fit(X_train, y_train)

# Evaluate the model
loss_rc = model_rc.evaluate(X_test, y_test, metrics=["mae"])
print(f"\nTest model loss (new input weights): {loss_rc[0]:.6f}")
