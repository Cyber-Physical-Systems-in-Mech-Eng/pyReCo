"""
Study the effect of reservoir network structure, i.e. the randomly generated reservoir adjacency matrix.

We will simply fit a large number of RCs, each with a randomly generated reservoir, on a benchmark case and observe
the variation in the scores.

By Merten Stender, TU Berlin
merten.stender@tu-berlin.de
September 23, 2024
"""

import numpy as np
import sys
import os
import platform
from matplotlib import pyplot as plt

# make pyreco available locally
if platform.system() == 'Windows':  # WINDOWS
    curr_loc = os.getcwd()
    pyreco_path = os.path.join('C:\\', *curr_loc.split('\\')[1:-1], 'src')
    sys.path.append(pyreco_path)
elif platform.system() == 'Darwin':  # MAC
    curr_loc = os.getcwd()
    pyreco_path = curr_loc + '/src'
    sys.path.append(pyreco_path)

from pyreco.models import ReservoirComputer
from pyreco.plotting import r2_scatter
from pyreco.utils_data import sequence_to_sequence

# some testing data: translate a sine signal to a cosine signal (i.e. sequence translation)
x_train, x_test, y_train, y_test = sequence_to_sequence(name='sine_to_cosine', n_batch=10, n_states=1, n_time=100)
input_shape = (x_train.shape[1], x_train.shape[2])
output_shape = (y_train.shape[1], y_train.shape[2])


def run_trials(num_trials,
               x_train, y_train, x_test, y_test,
               num_nodes: int, activation: str = 'tanh'):
    """
    Train and evaluate multiple Reservoir Computing (RC) models on the same dataset.

    This function fits `num_trials` separate RC models with the given number of nodes 
    and activation function. Each model is trained on the provided training data and 
    evaluated on the test data using the Mean Absolute Error (MAE). The function 
    returns the test scores for all trials, as well as the best and worst models 
    based on their evaluation performance.

    Parameters
    ----------
    num_trials : int
        The number of RC models to train and evaluate.
    x_train : array-like
        Input training data.
    y_train : array-like
        Target training data.
    x_test : array-like
        Input test data.
    y_test : array-like
        Target test data.
    num_nodes : int
        Number of reservoir nodes in each model.
    activation : str, optional
        Activation function used in the reservoir layer (default is 'tanh').

    Returns
    -------
    test_scores : np.ndarray
        A NumPy array containing the MAE scores for all trials.
    best_model : ReservoirComputer
        The trained model with the lowest MAE score on the test set.
    worst_model : ReservoirComputer
        The trained model with the highest MAE score on the test set.

    Notes
    -----
    - The function prints progress updates for each model, including its score 
      compared to previous best and worst models.
    - Each model is created, trained, evaluated, and deleted within the loop 
      to prevent excessive memory usage.
    - The `ReservoirComputer` class should be predefined and support `fit()` and 
      `evaluate()` methods.

    Examples
    --------
    >>> test_scores, best_model, worst_model = run_trials(
    ...     num_trials=10,
    ...     x_train=X_train, y_train=Y_train,
    ...     x_test=X_test, y_test=Y_test,
    ...     num_nodes=100, activation='relu'
    ... )
    """
    # fits num_trials RCs to the same data,
    # returns the list of scores, as well as best and worst model

    test_scores = []
    best_score, worst_score = 10 ** 6, 0
    best_model, worst_model = None, None
    for i in range(num_trials):
        print(f'network {i} / {num_trials}: {num_nodes} nodes, {activation} activation')

        # generate new model
        _model = ReservoirComputer(num_nodes=num_nodes, activation=activation)

        # fit to data
        _model.fit(X=x_train, y=y_train)

        # evaluate score
        _score = _model.evaluate(x_test, y_test, metrics='mae')[0]
        test_scores.append(_score)

        # store worst and best model
        if _score > worst_score:
            print(f'current score {_score:.4f} worse than before ({worst_score:.4f})')
            worst_model = _model
            worst_score = _score

        if _score < best_score:
            print(f'current score {_score:.4f} better than before ({best_score:.4f})')
            best_model = _model
            best_score = _score

        del _model

    test_scores = np.array(test_scores)  # turn into np array for easier handling

    return np.array(test_scores), best_model, worst_model


if __name__ == '__main__':

    # run the analysis for different numbers of reservoir nodes. Hypothesis: the larger the number of reservoir
    # nodes, the less will be the effect of network structure (redundancy of information)

    num_trials = 10000  # for each RC specification in terms of hyperparameters
    num_nodes = [10, 50, 100, 250, 500]  # number of nodes to try out
    activation = 'tanh'  # common activation function
    model_scores, best_models, worst_models = [], [], []

    for _nodes in num_nodes:
        # get scores for all trials for the given number of nodes
        _scores, _best_model, _worst_model = run_trials(num_trials=num_trials,
                                                        x_train=x_train, y_train=y_train,
                                                        x_test=x_test, y_test=y_test,
                                                        num_nodes=_nodes, activation=activation)
        model_scores.append(_scores)
        best_models.append(_best_model)
        worst_models.append(_worst_model)

    # plot the variation in the training performance based on reservoir initialization
    plt.figure()
    for idx, _nodes in enumerate(num_nodes):
        plt.hist(model_scores[idx], bins=20, label=fr'$N=${_nodes}', alpha=0.3)
    plt.xlabel('test score (MSE, sine to cosine)')
    plt.xlim([0, None])
    plt.ylabel('counts')
    plt.title(f'variation of reservoir network, {num_trials} different networks')
    plt.legend()
    plt.savefig('effects_network_structure_histogram.png')
    # plt.show()

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.boxplot(model_scores, showmeans=True, meanline=True)  # Create the boxplot
    ax.set_xticklabels(num_nodes) # Set the x-axis labels
    ax.set_xlabel(r'reservoir nodes $N$')
    ax.set_ylabel(r'test error ($L_1$)')
    ax.set_title(f'Performance variation by random networks ({num_trials} trials)')
    plt.savefig('effects_network_structure_boxplot.png')
    # plt.show()

    # plot predictions of best and worst model for first test sample
    plt.figure()
    plt.plot(y_test[0, :, 0], label='ground truth', color='black')
    for idx, _nodes in enumerate(num_nodes):
        y_pred_best = best_models[idx].predict(x_test)
        y_pred_worst = worst_models[idx].predict(x_test)
        plt.plot(y_pred_best[0, :, 0], label=f'best model (N={_nodes})', linestyle='dashed')
        # plt.plot(y_pred_worst[0, :, 0], label=f'worst model (N={_nodes})', linestyle='dashed')
    plt.xlabel('time')
    plt.ylabel('target / prediction')
    plt.title(rf'variation induced by network structure')
    plt.legend()
    plt.savefig('effects_network_structure_predictions.png')
    # plt.show()
