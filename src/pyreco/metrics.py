import numpy as np
from sklearn.metrics import r2_score


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # square the difference
    squared_err = np.square(err)

    # calculate the mean of the squared differences
    return np.mean(squared_err)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # calculate the absolute difference between true and predicted values
    absolute_err = np.abs(err)

    # calculate the mean of the absolute differences
    return np.mean(absolute_err)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # computes R2 score for scalar regression problems
    # we will flatten along time and state dimensions
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return r2_score(y_true=y_true, y_pred=y_pred)


def available_metrics() -> list:
    return ['mse', 'mean_squared_error', 'mae', 'mean_absolute_error', 'r2', 'r2_score']


def assign_metric(metric: str):
    if metric not in available_metrics():
        raise (ValueError(f'metric {metric} is not implemented. Check for implementation in metrics.py and in '
                          f'assign_metric() function'))

    if metric == 'mse' or metric == 'mean_squared_error':
        return mse
    elif metric == 'mae' or metric == 'mean_absolute_error':
        return mae
    elif metric == 'r2' or metric == 'r2_score':
        return r2
