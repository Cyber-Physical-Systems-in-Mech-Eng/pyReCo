import numpy as np
from sklearn.metrics import r2_score


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between the true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        The true target values.
    y_pred : np.ndarray
        The predicted values.

    Returns
    -------
    float
        The Mean Squared Error between `y_true` and `y_pred`.

    Notes
    -----
    The MSE is calculated as the average of the squared differences between the true and predicted values.
    """
    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # square the difference
    squared_err = np.square(err)

    # calculate the mean of the squared differences
    return np.mean(squared_err)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE) between the true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        The true target values.
    y_pred : np.ndarray
        The predicted values.

    Returns
    -------
    float
        The Mean Absolute Error between `y_true` and `y_pred`.

    Notes
    -----
    The MAE is calculated as the average of the absolute differences between the true and predicted values.
    """
    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # calculate the absolute difference between true and predicted values
    absolute_err = np.abs(err)

    # calculate the mean of the absolute differences
    return np.mean(absolute_err)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the R-squared (R²) score for scalar regression problems.

    Parameters
    ----------
    y_true : np.ndarray
        The true target values.
    y_pred : np.ndarray
        The predicted values.

    Returns
    -------
    float
        The R-squared score between `y_true` and `y_pred`.

    Notes
    -----
    The function flattens both `y_true` and `y_pred` along the time and state dimensions
    before calculating the R² score.
    The R² score is a statistical measure that represents the proportion of the variance
    in the dependent variable that is predictable from the independent variables.

    Uses `sklearn.metrics.r2_score` to compute the R² value.
    """
    # computes R2 score for scalar regression problems
    # we will flatten along time and state dimensions
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return r2_score(y_true=y_true, y_pred=y_pred)


def available_metrics() -> list:
    """
    Returns a list of available metric names for regression evaluation.

    Returns
    -------
    list
        A list of strings representing the names of available metrics, including:
        'mse', 'mean_squared_error', 'mae', 'mean_absolute_error', 'r2', 'r2_score'.

    Notes
    -----
    This function provides a predefined set of metric names that can be used
    for evaluating regression models, such as Mean Squared Error (MSE), Mean Absolute Error (MAE),
    and R-squared (R²) score.
    """
    return ['mse', 'mean_squared_error', 'mae', 'mean_absolute_error', 'r2', 'r2_score']


def assign_metric(metric: str):
    """
    Assigns the appropriate metric function based on the metric name.

    Parameters
    ----------
    metric : str
        The name of the metric function to assign. Available options are:
        'mse', 'mean_squared_error', 'mae', 'mean_absolute_error', 'r2', 'r2_score'.

    Returns
    -------
    function
        The corresponding metric function (e.g., `mse`, `mae`, `r2`) based on the input metric name.

    Raises
    ------
    ValueError
        If the provided metric name is not recognized or implemented.

    Notes
    -----
    This function checks if the given metric name exists in the list of available metrics
    and returns the appropriate function for calculating the metric. If the metric is not found,
    a ValueError is raised with a message pointing to the relevant implementation.
    """
    if metric not in available_metrics():
        raise (ValueError(f'metric {metric} is not implemented. Check for implementation in metrics.py and in '
                          f'assign_metric() function'))

    if metric == 'mse' or metric == 'mean_squared_error':
        return mse
    elif metric == 'mae' or metric == 'mean_absolute_error':
        return mae
    elif metric == 'r2' or metric == 'r2_score':
        return r2
