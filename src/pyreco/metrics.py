r"""
Metrics for scoring predictions.

All metrics take the true values :math:`y` and the predicted values :math:`\hat y` with identical
shapes, typically ``(n_batch, n_timesteps, n_states)``. The last axis holds the channels
:math:`c = 1, \dots, C`; everything before it is treated as :math:`N` samples. Metrics that depend
on the scale of a channel (``r2``, ``nrmse``) are computed per channel and then averaged, so that
every channel counts equally.

Every metric is either a loss (lower is better: ``mse``, ``mae``, ``rmse``, ``nrmse``) or a score
(higher is better: ``r2``); see :func:`is_loss`.
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
    r"""
    Mean squared error over all samples and channels.

    .. math::

        \mathrm{MSE} = \frac{1}{N C} \sum_{i=1}^{N} \sum_{c=1}^{C} \left(y_{ic} - \hat y_{ic}\right)^2

    Parameters
    ----------
    y_true, y_pred : ndarray
        True and predicted values of identical shape; the last axis holds the channels.

    Returns
    -------
    float
        The mean squared error, in the squared units of the data.
    """
    y_true, y_pred = _channels(y_true, y_pred)

    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # square the difference
    squared_err = np.square(err)

    # calculate the mean of the squared differences
    return np.mean(squared_err)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""
    Mean absolute error over all samples and channels.

    .. math::

        \mathrm{MAE} = \frac{1}{N C} \sum_{i=1}^{N} \sum_{c=1}^{C} \left| y_{ic} - \hat y_{ic} \right|

    Parameters
    ----------
    y_true, y_pred : ndarray
        True and predicted values of identical shape; the last axis holds the channels.

    Returns
    -------
    float
        The mean absolute error, in the units of the data.
    """
    y_true, y_pred = _channels(y_true, y_pred)

    # calculate the difference between true and predicted values
    err = y_true - y_pred

    # calculate the absolute difference between true and predicted values
    absolute_err = np.abs(err)

    # calculate the mean of the absolute differences
    return np.mean(absolute_err)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""
    Root mean squared error over all samples and channels, in the units of the data.

    .. math::

        \mathrm{RMSE} = \sqrt{\mathrm{MSE}}

    Parameters
    ----------
    y_true, y_pred : ndarray
        True and predicted values of identical shape; the last axis holds the channels.

    Returns
    -------
    float
        The root mean squared error.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""
    Normalized root mean squared error, averaged over the channels.

    .. math::

        \mathrm{NRMSE} = \sqrt{\frac{1}{C} \sum_{c=1}^{C} \frac{\mathrm{MSE}_c}{\mathrm{Var}_c}},
        \qquad
        \mathrm{MSE}_c = \frac{1}{N} \sum_{i=1}^{N} \left(y_{ic} - \hat y_{ic}\right)^2,
        \quad
        \mathrm{Var}_c = \frac{1}{N} \sum_{i=1}^{N} \left(y_{ic} - \bar y_c\right)^2 .

    A perfect prediction gives 0, and predicting the mean :math:`\bar y_c` of every channel gives 1.
    With the per-channel :math:`R^2_c = 1 - \mathrm{MSE}_c / \mathrm{Var}_c` of :func:`r2`,
    :math:`\mathrm{NRMSE}^2 = 1 - \frac{1}{C} \sum_c R^2_c`.

    Parameters
    ----------
    y_true, y_pred : ndarray
        True and predicted values of identical shape; the last axis holds the channels.

    Returns
    -------
    float
        The NRMSE (dimensionless).

    Warns
    -----
    RuntimeWarning
        Channels with :math:`\mathrm{Var}_c = 0` cannot be normalized and are left out.

    Raises
    ------
    ValueError
        If all channels are constant.
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
    r"""
    Coefficient of determination, computed per channel and averaged with equal weights.

    .. math::

        R^2 = \frac{1}{C} \sum_{c=1}^{C} \left( 1 - \frac{\sum_i (y_{ic} - \hat y_{ic})^2}{\sum_i (y_{ic} - \bar y_c)^2} \right)

    Pooling the channels would let a channel with a large mean or variance hide a failing one.
    The score is 1 for a perfect prediction and 0 for predicting the mean.

    Parameters
    ----------
    y_true, y_pred : ndarray
        True and predicted values of identical shape; the last axis holds the channels.

    Returns
    -------
    float
        The coefficient of determination (higher is better).
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
    """Names (and aliases) of all implemented metrics."""
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
