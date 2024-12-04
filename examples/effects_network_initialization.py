"""
Study the effect of reservoir initialization, i.e. R(t=0) on the predictive performance of the RC. We will fit the
very same reservoir network, but everytime initialized differently. The variation of the predictive performance will
display the effect of initialization.

By Merten Stender, TU Berlin
merten.stender@tu-berlin.de
September 12, 2024
"""

import numpy as np
import sys
import os
import platform
from matplotlib import pyplot as plt

# make pyreco available locally
if platform.system() == 'Windows':  # WINDOWS
    curr_loc = os.getcwd()
    pyreco_path = os.path.join('C:\\',*curr_loc.split('\\')[1:-1], 'src')
    sys.path.append(pyreco_path)
elif platform.system() == 'Darwin':  # MAC
    curr_loc = os.getcwd()
    pyreco_path = curr_loc + '/src'
    sys.path.append(pyreco_path)

from pyreco.models import ReservoirComputer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import sequence_to_sequence


# some testing data: predict a sine signal one step into the future
X_train, X_test, y_train, y_test = sequence_to_sequence(name='sine_pred', n_batch=20, n_states=4, n_time=100)
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])

"""
Variation of the initial reservoir state sampling method
"""

# fit a reservoir computer using a large number of initial conditions; vary the sampling method for init conditions
sampling = ['ones', 'zeros', 'random', 'random_normal']

num_nodes = 100
n_init = 10
histories = []
for method in sampling:
    model = ReservoirComputer(num_nodes=num_nodes, activation='tanh', init_res_sampling=method)
    histories.append(model.fit(X=X_train, y=y_train, n_init=n_init))

# plot the variation in the training performance based on reservoir initialization
plt.figure()
for idx, method in enumerate(sampling):
    plt.hist(histories[idx]['train_scores'],bins=20, label=method, alpha=0.3)
plt.xlabel('training score')
plt.ylabel('counts')
plt.title(f'variation of initial reservoir states, {num_nodes} nodes, {n_init} init. conditions')
plt.legend()
plt.show()


"""
Dependency on reservoir size: the larger the reservoir, the less important the initial conditions?
"""
histories_rnd, histories_rand = [], []
mean_rnd, mean_rand = [], []
std_rnd, std_rand = [], []
num_nodes_grid = np.logspace(start=1, stop=3, num=20)
n_init = 20

for num_node in num_nodes_grid:

    # random normal
    model = ReservoirComputer(num_nodes=num_nodes, activation='tanh',init_res_sampling='random_normal')
    histories_rnd.append(model.fit(X=X_train, y=y_train, n_init=n_init))
    mean_rnd.append(np.mean(histories_rnd[-1]['train_scores']))
    std_rnd.append(np.std(histories_rnd[-1]['train_scores']))

    # random
    model = ReservoirComputer(num_nodes=num_nodes, activation='tanh',init_res_sampling='random')
    histories_rand.append(model.fit(X=X_train, y=y_train, n_init=n_init))
    mean_rand.append(np.mean(histories_rand[-1]['train_scores']))
    std_rand.append(np.std(histories_rand[-1]['train_scores']))

plt.figure()
plt.plot(num_nodes_grid, mean_rnd, label='random normal', marker='.')
plt.plot(num_nodes_grid, mean_rand, label='random', marker='.')
plt.xlabel('num nodes')
plt.xscale('log')
plt.ylabel('training error')
plt.show()