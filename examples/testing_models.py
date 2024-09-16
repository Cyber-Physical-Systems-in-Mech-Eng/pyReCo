"""
Small test cases for the Model API

"""

import numpy as np
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

"""+
generate some training data: map a sine to a cosine (learn a phase shift) with signal amplification (learn to increase 
the amplitude in the targets)
"""

from pyreco.utils_data import sequence_to_sequence

# from pyreco.utils_data import scalar_to_scalar, vector_to_vector, sequence_to_sequence, x_to_x
X_train, X_test, y_train, y_test = sequence_to_sequence(name='sine_pred', n_batch=20, n_states=2, n_time=150)
print(f'shape of training data X: {X_train.shape}, shape of test data X: {X_test.shape}')
print(f'shape of training data y: {y_train.shape}, shape of test data y: {y_test.shape}')

from pyreco.models import ReservoirComputer as RC
from pyreco.plotting import r2_scatter

model = RC(num_nodes=500, activation='tanh')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'shape of predicted array: {y_pred.shape}')

r2_scatter(y_test, y_pred)

plt.figure(figsize=(10,4), dpi=100)
plt.plot(y_train[0,:,0], label='ground truth', marker='.', color='#1D3557')
plt.plot(y_pred[0,:,0], label='prediction', marker='.', color='#E63946')
plt.legend()
plt.xlabel('time')
plt.tight_layout()
plt.savefig('predictions_model.png')
plt.show()

metric_value = model.evaluate(X=X_test, y=y_test, metrics=['mse','mae'])
print(f'scores:{metric_value}')


from pyreco.cross_validation import cross_val

mses, mean_mses, std_dev_mses = cross_val(model, X_train, y_train, n_splits=5, metric='mse')
print(f"Cross-Validation MSE: {mses}")
print(f"Mean MSE: {mean_mses:.3f}")
print(f"Standard Deviation of MSE: {std_dev_mses:.3f}")
