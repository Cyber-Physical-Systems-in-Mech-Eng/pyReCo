"""
Metrics for scoring predictions.

All metrics take the true and the predicted values with identical shapes, typically
[n_batch, n_timesteps, n_states]. The last axis holds the channels (states); everything
before it is treated as samples. Metrics that depend on the scale of a channel (r2, nrmse)
are computed per channel and then averaged, so that every channel counts equally.

Every metric is either a loss (lower is better: mse, mae, rmse, nrmse) or a score
(higher is better: r2); see `is_loss`.
"""

import warnings

import numpy as np
from sklearn.metrics import r2_score


def _channels(y_true: np.ndarray, y_pred: np.ndarray):
    """Validate shapes and flatten to [n_samples, n_channels] (last axis = channels)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and "
            f"{y_pred.shape}."
        )
    if y_true.ndim == 0:
        raise ValueError("y_true and y_pred must be arrays, not scalars.")
    n_channels = 1 if y_true.ndim == 1 else y_true.shape[-1]
    return y_true.reshape(-1, n_channels), y_pred.reshape(-1, n_channels)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error over all entries."""
    y_true, y_pred = _channels(y_true, y_pred)

    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # square the difference
    squared_err = np.square(err)

    # calculate the mean of the squared differences
    return np.mean(squared_err)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error over all entries."""
    y_true, y_pred = _channels(y_true, y_pred)

    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # calculate the absolute difference between true and predicted values
    absolute_err = np.abs(err)

    # calculate the mean of the absolute differences
    return np.mean(absolute_err)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error over all entries, in the units of the data."""
    return float(np.sqrt(mse(y_true, y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized root mean squared error, averaged over channels:

        NRMSE = sqrt( mean_c  MSE_c / Var_c )

    MSE_c and Var_c are the mean squared error and the variance of the true values of
    channel c over all samples of the data that is evaluated. A perfect prediction gives
    0, and predicting the mean of every channel gives 1. For r2 as the per-channel
    average, NRMSE**2 = 1 - r2.

    Channels with zero variance cannot be normalized: they are left out with a warning.
    If all channels are constant, a ValueError is raised.
    """
    y_true, y_pred = _channels(y_true, y_pred)
    var = y_true.var(axis=0)
    channel_mse = np.mean(np.square(y_true - y_pred), axis=0)

    constant = var == 0
    if constant.all():
        raise ValueError("nrmse is undefined: the true values of all channels are constant.")
    if constant.any():
        warnings.warn(
            f"nrmse: channel(s) {np.flatnonzero(constant).tolist()} are constant and are "
            "left out.",
            RuntimeWarning,
        )

    return float(np.sqrt(np.mean(channel_mse[~constant] / var[~constant])))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination, computed per channel and averaged with equal weights.
    (Pooling the channels would let a channel with a large mean or variance hide a
    failing one.)
    """
    y_true, y_pred = _channels(y_true, y_pred)
    return r2_score(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")


# canonical name -> (function, lower is better)
_METRICS = {
    "mse": (mse, True),
    "mae": (mae, True),
    "rmse": (rmse, True),
    "nrmse": (nrmse, True),
    "r2": (r2, False),
}

# alternative names
_ALIASES = {
    "mean_squared_error": "mse",
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
    "normalized_root_mean_squared_error": "nrmse",
    "r2_score": "r2",
}


def available_metrics() -> list:
    return [
        "mse",
        "mean_squared_error",
        "mae",
        "mean_absolute_error",
        "rmse",
        "root_mean_squared_error",
        "nrmse",
        "normalized_root_mean_squared_error",
        "r2",
        "r2_score",
    ]


def _canonical(metric: str) -> str:
    if not isinstance(metric, str) or metric not in available_metrics():
        raise ValueError(
            f"metric {metric} is not implemented. Available metrics: "
            f"{available_metrics()}. Check for implementation in metrics.py"
        )
    return _ALIASES.get(metric, metric)


def assign_metric(metric: str):
    """Return the function that computes the metric with the given name."""
    return _METRICS[_canonical(metric)][0]


def is_loss(metric: str) -> bool:
    """True if lower values are better (mse, mae, rmse, nrmse), False for scores (r2)."""
    return _METRICS[_canonical(metric)][1]
