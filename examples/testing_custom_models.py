"""
Some testing use case for the CustomModel API of pyreco
"""

from matplotlib import pyplot as plt
import sys
import os
import platform

# make pyReCo available locally
if platform.system() == 'Windows':  # WINDOWS
    curr_loc = os.getcwd()
    pyreco_path = os.path.join('C:\\',*curr_loc.split('\\')[1:-1], 'src')
    sys.path.append(pyreco_path)
elif platform.system() == 'Darwin':  # MAC
    curr_loc = os.getcwd()
    pyreco_path = curr_loc + '/src'
    sys.path.append(pyreco_path)

from pyreco.custom_models import RC as RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import scalar_to_scalar, vector_to_vector, sequence_to_sequence, x_to_x

"""
Classic RC built on random networks. 

Use case: predict a sine signal into the future (n_time) samples from the past are mapped to (n_time) samples in the 
future
"""

# some testing data: predict a sine signal.
X_train, X_test, y_train, y_test = sequence_to_sequence(name='sine_pred', n_batch=20, n_states=1, n_time=200)

# set the dimensions (need to be [n_batch, n_time, n_states])
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(RandomReservoirLayer(nodes=100, density=0.1, activation='sigmoid', leakage_rate=0.1 , fraction_input=0.5))
model_rc.add(ReadoutLayer(output_shape, fraction_out=0.99))

# Compile the model
discard_transients = 100
model_rc.compile(optimizer='ridge', metrics=['mean_squared_error'], discard_transients=discard_transients)

# Train the model
model_rc.fit(X_train, y_train)

# Make predictions for new data
y_pred = model_rc.predict(X_test)
print(f'shape of predictions on test set: {y_pred.shape}')

# Evaluate the model
loss_rc = model_rc.evaluate(X_test, y_test, metrics=['mae'])
print(f'Test model loss: {loss_rc}')

# plot predictions vs. ground truth
r2_scatter(y_true=y_test[:,discard_transients:, :], y_pred=y_pred)

plt.figure()
plt.plot(y_test[0,discard_transients:,0], label='ground truth', marker='.')
plt.plot(y_pred[0,:,0], label='prediction', marker='.')
plt.title('test set')
plt.legend()
plt.xlabel('time')
plt.ylabel('x1')
plt.show()


"""
k-fold cross-validation of the model: estimate bias and variance of the model
"""
from pyreco.cross_validation import cross_val

val, mean, std_dev = cross_val(model_rc, X_train, y_train, n_splits=5, metric = ['mse'])
print(f"Cross-Validation MSE: {val}")
print(f"Mean MSE: {mean:.3f}")
print(f"Standard Deviation of MSE: {std_dev:.3f}")

