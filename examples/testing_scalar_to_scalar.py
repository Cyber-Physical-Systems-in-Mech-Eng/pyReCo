"""
Some testing use case learning a scalar-to-scalar mapping task
"""

from matplotlib import pyplot as plt
from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import vector_to_vector

"""
Classic RC built on random networks. 
Use case: predict sin(t+dt) from sin(t) (scalar-to-scalar mapping task)
"""
# some testing data: predict a sine signal.
X_train, X_test, y_train, y_test = vector_to_vector(
    name="sine_prediction", n_states=1, n_batch=200
)

# set the dimensions (need to be [n_batch, n_time, n_states])
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])
print(f"input data shape: {input_shape}")
print(f"output data shape: {output_shape}")

# build simple RC
model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(
    RandomReservoirLayer(
        nodes=100,
        density=0.5,
        leakage_rate=0.1,
        activation="sigmoid",
        fraction_input=0.5,
    )
)
model_rc.add(ReadoutLayer(output_shape, fraction_out=0.99))

# Compile the model
model_rc.compile()

# Train the model
model_rc.fit(X_train, y_train)

# Make predictions for new data
y_pred = model_rc.predict(X_test)
print(f"shape of predictions on test set: {y_pred.shape}")

# Evaluate the model
loss_rc = model_rc.evaluate(X_test, y_test, metrics=["mae"])
print(f"Test model loss: {loss_rc[0]:.4f}")

# plot predictions vs. ground truth
r2_scatter(y_true=y_test, y_pred=y_pred)
