"""
Some plotting capabilities
"""

import numpy as np
from matplotlib import pyplot as plt

from .metrics import r2

def r2_scatter(y_true: np.ndarray, y_pred: np.ndarray, state_idx: int|tuple= None):
    # plots predictions against ground truth values as scatter plot
    # lets the user choose the output state to plot (if there are multiple states). If not provided, all states will
    # be plotted.
    # collapses the data along the time dimension to show time-dependent targets

    # expects arguments to be of 3D shape: [n_batch, n_timesteps, n_states]

    if y_true.ndim != y_pred.ndim:
        raise(ValueError('Inconsistent shapes! y_true and y_pred need to have the same shape!'))

    if (state_idx is not None) and (np.max(state_idx) >= y_true.ndim):
        raise(ValueError(f'Please select a valid state index, maximum being {y_true.ndim} for the given data'))

    # select the states to show in case supplied by the user
    if state_idx is not None:
        y_true = y_true[:, :, state_idx]
        y_pred = y_pred[:, :, state_idx]

    # now flatten into a vector (along states and along time if necessary)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    min_val = np.min(y_true)  # np.min([np.min(y_true), np.min(y_pred)])
    max_val = np.max(y_true)  # np.max([np.max(y_true), np.max(y_pred)])

    fig = plt.figure()
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle='solid',
             color='gray',
             label='perfect model')
    plt.plot(y_true, y_pred,
             linestyle='none',
             marker='.',
             markersize=5,
             color='black',
             label='model predictions')
    plt.xlabel('ground truth')
    plt.ylabel('predictions')
    plt.legend()
    plt.title(rf'$R^2=${r2(y_true, y_pred):.2f}')
    plt.tight_layout()
    plt.show()


