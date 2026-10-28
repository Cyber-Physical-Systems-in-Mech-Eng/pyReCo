"""
Helpers that cut the first time steps (the warm-up transient) off arrays.

Note: the models handle the warm-up themselves (``compile(discard_transients=d)``); these functions are
not used by them.
"""


def RemoveTransients_Res(ResStates, Transients):
    """Drop the first ``Transients`` rows of a reservoir state matrix of shape ``(n_time, N)``."""
    return ResStates[Transients:,:]

def RemoveTransient_Inps(X, Transients):
    """Drop the first ``Transients`` time steps of inputs of shape ``(n_batch, n_time, n_states)``."""
    return X[:,Transients:,:]

def RemoveTransient_Outs(Y, Transients):
    """Drop the first ``Transients`` time steps of targets of shape ``(n_batch, n_time, n_states)``."""
    return Y[:,Transients:,:]


def TransientRemover(What: str, ResStates, X, Y, Transients: int):
    """
    Remove the first ``Transients`` time steps from the selected arrays.

    Parameters
    ----------
    What : {"RX", "RXY"}
        Which arrays are returned: reservoir states and inputs, or states, inputs and targets.
    ResStates : ndarray
        Reservoir states of shape ``(n_time, N)``.
    X, Y : ndarray
        Inputs and targets of shape ``(n_batch, n_time, n_states)``.
    Transients : int
        Number of leading time steps to drop.

    Returns
    -------
    tuple of ndarray
        The shortened arrays selected by ``What``.
    """

    if What=='RX':
        return RemoveTransients_Res(ResStates, Transients), RemoveTransient_Inps(X, Transients)
    if What == 'RXY':
        return RemoveTransients_Res(ResStates, Transients), RemoveTransient_Inps(X, Transients), RemoveTransient_Outs(Y, Transients)
