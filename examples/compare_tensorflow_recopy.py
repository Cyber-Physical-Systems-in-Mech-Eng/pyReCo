"""
Comparison of a vanilla feedforward NN (implemented in TensorFlow) against a vanilla Reservoir Computer (implemented
in pyReCo). Try to make them roughly the same sizeand compare performance.

Benchmark case: sequence translation. map a sine to a cosine of the same frequency (i.e. learn a phase shift).

Requires no memory, hence no recurrent network is required per se

Merten Stender, TU Berlin, merten.stender@tu-berlin.de
09.10.2024
"""

import numpy as np
from matplotlib import pyplot as plt
import time

# make pyreco available locally
import sys
import os
import platform

if platform.system() == 'Windows':  # WINDOWS
    curr_loc = os.getcwd()
    pyreco_path = os.path.join('C:\\',*curr_loc.split('\\')[1:-1], 'src')
    sys.path.append(pyreco_path)
elif platform.system() == 'Darwin':  # MAC
    curr_loc = os.getcwd()
    pyreco_path = curr_loc + '/src'
    sys.path.append(pyreco_path)


"""
generate some training data: map a sine to a cosine (learn a phase shift). 
Randomly sample the frequencies to generate a very(!) small data set of 10 samples (8 for training, 2 for testing).
"""

from pyreco.utils_data import sequence_to_sequence

# generates a sine signal of varying frequency, and obtain the cosine of the same frequency as output. 100 time steps
X_train, X_test, y_train, y_test = sequence_to_sequence('sine_to_cosine', n_batch=10, n_states=1, n_time=100)

print(f'shape of training inputs: \t{X_train.shape}')
print(f'shape of training outputs: \t{y_train.shape}')


# set the dimensions for the models. First dim is the number of time steps (=100), second dim is the number of
# states/sensors
# (=1)
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])


"""
Modeling using feed-forward neural nets in TensorFlow (sequential API). 

Use a single hidden layer (100 nodes, tanh activation)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

model_ann = Sequential()                                            # Instantiate Sequential model
model_ann.add(Input(shape=input_shape))                             # Add input layer
model_ann.add(Dense(units=300, activation='tanh'))                  # Add hidden Dense layer
model_ann.add(Dense(units=1, activation='linear'))                  # Add output layer
model_ann.compile(optimizer='adam', loss='mean_squared_error')      # Compile the model

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1)  # prevent overfitting

# Train the model for 500 epochs
print('\n FEEDFORWARD NEURAL NET')
t_start_ann = time.time()
hist_ann = model_ann.fit(X_train, y_train, validation_data=(X_test, y_test),
                         callbacks=[es], epochs=1000)
t_ann = time.time() - t_start_ann


"""
Build a vanilla reservoir computer for the same task. Use 100 reservoir nodes with tanh activation
"""

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer

model_rc = RC()
model_rc.add(InputLayer(input_shape=input_shape))
model_rc.add(RandomReservoirLayer(nodes=30, activation='tanh', fraction_input=1.0))
model_rc.add(ReadoutLayer(output_shape=output_shape, fraction_out=1.0))
model_rc.compile(optimizer='ridge', metrics=['mean_squared_error'])                     # compile the model and train

print('\n TRAINING RESERVOIR COMPUTER')
t_start_rc = time.time()
model_rc.fit(X_train, y_train)
t_rc = time.time() - t_start_rc


"""
Now compare the predictions of both models 
"""
from pyreco.metrics import mse

# let both models make predictions
y_pred_ann = model_ann.predict(X_test)
y_pred_rc = model_rc.predict(X_test)

print(f'\nshape of predicted arrays:\t ANN: {y_pred_ann.shape}, \tRC: {y_pred_ann.shape}')

# evaluate some metrics (for simplicity on the train set)
score_ann = mse(y_true=y_test, y_pred=y_pred_ann)
score_rc = mse(y_true=y_test, y_pred=y_pred_rc)

# compare validation scores, compute times and number of trainable parameters
print(f'validation L2 scores: \t\t ANN: {score_ann:.4f} \t RC: {score_rc:.4f}')
print(f'training time [s]: \t\t\t ANN: {t_ann:.4f} \t RC: {t_rc:.4f}')
print(f'number trainable weights: \t ANN: {model_ann.count_params()} \t\t RC: {model_rc.trainable_weights}')

"""
Some plots
"""

from cpsmehelper import export_figure

# display input and output for first training sample
fig = plt.figure()
plt.plot(X_train[0,:,0], label='input', color='blue')
plt.plot(y_train[0,:,0], label='output', color='cyan')
plt.xlabel('time')
plt.legend()
plt.tight_layout()
export_figure(fig, 'compare_tf_rc_task.png', width=11, height=3)
plt.show()

# plot the predictions against ground truth
plot_idx = 0  # which valiation sample to plot
fig = plt.figure(figsize=(10,4), dpi=100)
plt.plot(y_test[plot_idx,:,0], label='ground truth', marker='.', color='#1D3557')
plt.plot(y_pred_ann[plot_idx,:,0], label=f'ANN (MSE={mse(y_true=y_test[plot_idx], y_pred=y_pred_ann[plot_idx]):.3f})',
         marker='.', color='#E63946')
plt.plot(y_pred_rc[plot_idx,:,0], label=f'RC (MSE={mse(y_true=y_test[plot_idx], y_pred=y_pred_rc[plot_idx]):.3f})',
         marker='.', color='#00b695')
plt.legend(loc='upper right')
plt.xlabel('time')
plt.tight_layout()
plt.title('Comparing ANN against RC for sine-to-cosine task')
# plt.savefig('model_predictions.png')
export_figure(fig, 'compare_tf_rc.png', width=11, height=9)
plt.show()

# plot the ANN network training
fig = plt.figure()   # plot the training history of the feedforward network
plt.plot(hist_ann.history['loss'], label='train')
plt.plot(hist_ann.history['val_loss'], label='test')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()


# plot an R^2 - like graphic
from pyreco.plotting import r2_scatter

r2_scatter(y_true=y_test, y_pred=y_pred_ann, title='Feedforward ANN')
r2_scatter(y_true=y_test, y_pred=y_pred_rc, title='Feedforward RC')

