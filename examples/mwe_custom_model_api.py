import numpy as np
from matplotlib import pyplot as plt


import sys
import os
import platform

# make pyreco available locally
if platform.system() == 'Windows':  # WINDOWS
    curr_loc = os.getcwd()
    pyreco_path = os.path.join('C:\\',*curr_loc.split('\\')[1:-1], 'src')
    sys.path.append(pyreco_path)
elif platform.system() == 'Darwin':  # MAC
    curr_loc = os.getcwd()
    pyreco_path = curr_loc + '/src'
    sys.path.append(pyreco_path)

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer

"""+
generate some training data: map a sine to a cosine (learn a phase shift) with signal amplification (learn to increase 
the amplitude in the targets)
"""

# generate 3 cycles of a sine (input) and of a cosine (output)
omega = np.pi
t = np.linspace(start=0, stop=3 * (2*np.pi/omega), num=300, endpoint=True)
x = np.sin(omega * t)
y = 2 * np.cos(omega * t)

x_train = np.expand_dims(x, axis=(0,2))  # obtain shape of [n_batch, n_time, n_states]
y_train = np.expand_dims(y, axis=(0,2))

# set the dimensions
input_shape = (x_train.shape[1], x_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

# build a custom RC model by adding layers with properties
model = RC()
model.add(InputLayer(input_shape=input_shape))
model.add(RandomReservoirLayer(nodes=200, activation='tanh', fraction_input=0.5))
model.add(ReadoutLayer(output_shape))

# compile the model
model.compile(optimizer='ridge', metrics=['mean_squared_error'])

# train the model
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_train)
print(f'shape of predicted array: {y_pred.shape}')

# evaluate some metrics (for simplicity on the train set)
metric_value = model.evaluate(X=x_train, y=y_train, metrics=['mse','mae'])
print(f'scores:{metric_value}')

plt.figure(figsize=(10,4), dpi=100)
plt.plot(y_train[0,:,0], label='ground truth', marker='.', color='#1D3557')
plt.plot(y_pred[0,:,0], label='prediction', marker='.', color='#E63946')
plt.legend()
plt.xlabel('time')
plt.tight_layout()
plt.savefig('model_predictions.png')
plt.show()

# plot an R^2 - like graphic
from pyreco.plotting import r2_scatter
r2_scatter(y_train, y_pred)

