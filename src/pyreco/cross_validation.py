# Comment for cross val check
import numpy as np
from typing import List, Tuple
import warnings


def cross_val(model, X: np.ndarray, y: np.ndarray, n_splits: int, metric: str = 'mse') -> Tuple[
    List[float], float, float]:
    '''
    Performs k-fold cross-validation on a given model.

    Parameters:
    model : object
        The RC model to be validated.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    n_splits : int
        Number of folds.
    metrics : str
        Metric to evaluate.

    Returns:
    tuple
        A tuple containing:
        - list of metric's value for each fold
        - mean of the metric values
        - standard deviation of the metric values
    '''

    # issue warning if more than one metric is specified
    if type(metric) is list:
        if len(metric) > 1:
            metric = metric[0]
            warnings.warn('Only a single metric should be specified. Using the first metric in the list.')

    # get indices for splitting the data
    indices = np.arange(X.shape[0])
    # shuffle the indices
    np.random.shuffle(indices)

    # split the indices into n_splits parts (floor division)
    fold_sizes = np.full(n_splits, X.shape[0] // n_splits)
    # distribute remaining data (caused by floor division) as far as possible
    fold_sizes[:X.shape[0] % n_splits] += 1

    # get shuffled indices for folds
    current = 0
    fold_indices = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        fold_indices.append(indices[start:stop])
        current = stop

    # perform cross-validation
    metric_folds_values = []
    mean_metric_value = []
    std_dev_metric_value = []

    for i in range(n_splits):
        # select the test indices for the current fold
        test_indices = fold_indices[i]

        # select the train indices (test_indices are removed from indices)
        train_indices = np.setdiff1d(indices, test_indices)

        # split the data into training and testing sets
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # train the model
        model.fit(X_train, y_train)

        # calculate metric value for fold
        metric_value = model.evaluate(X_test, y_test, metric)

        # append metric value of fold
        metric_folds_values.append(metric_value)

    # get mean accuracy and standard deviation of metric values of all folds
    mean_metric_value = float(np.mean(metric_folds_values))
    std_dev_metric_value = float(np.std(metric_folds_values))

    # output the results
    return metric_folds_values, mean_metric_value, std_dev_metric_value
