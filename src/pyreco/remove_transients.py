
def RemoveTransients_Res(ResStates, Transients):
    """
    Removes transient states from the reservoir states by slicing the input array.

    Parameters
    ----------
    ResStates
        A 2D array representing the reservoir states. The shape should be 
        (n_timesteps, n_nodes), where n_timesteps is the number of time steps 
        and n_nodes is the number of nodes in the reservoir.
    Transients
        The number of initial transient steps to remove from the `ResStates` array.

    Returns
    -------
    2D array
        A new array with the transient states removed.
    """
    return ResStates[Transients:,:]

def RemoveTransient_Inps(X, Transients):
    """
    Removes transient inputs from the input data by slicing the array.

    Parameters
    ----------
    X
        A 3D array representing the input data. The shape should be 
        (n_batch, n_timesteps, n_features), where n_batch is the number of 
        input samples, n_timesteps is the number of time steps, and 
        n_features is the number of features.
    Transients
        The number of initial transient steps to remove from the `X` array.

    Returns
    -------
    3D array
        A new array with the transient inputs removed.
    """
    return X[:,Transients:,:]

def RemoveTransient_Outs(Y, Transients):
    """
    Removes transient outputs from the output data by slicing the array.

    Parameters
    ----------
    Y
        A 3D array representing the output data. The shape should be
        (n_batch, n_timesteps, n_output_features), where n_batch is the number of
        output samples, n_timesteps is the number of time steps, and
        n_output_features is the number of output features.
    Transients
        The number of initial transient steps to remove from the `Y` array.

    Returns
    -------
    3D array
        A new array with the transient outputs removed.
    """
    return Y[:,Transients:,:]


def TransientRemover(What: str, ResStates, X, Y, Transients: int):
    """
    Removes transient data from the given arrays based on the specified input.

    Parameters
    ----------
    What
        A string that determines which data to remove. Can be 'RX' or 'RXY'.
    ResStates
        The reservoir states data, usually a 3D array.
    X
        The input data array, usually a 3D array.
    Y
        The output data array, usually a 3D array.
    Transients
        The number of initial transient steps to remove.

    Returns
    -------
    tuple
        A tuple of arrays, where the contents depend on the value of `What`:
        - If 'RX', returns the reservoir states and input data with transients removed.
        - If 'RXY', returns the reservoir states, input data, and output data with transients removed.
    """
    if What=='RX':
        return RemoveTransients_Res(ResStates, Transients), RemoveTransient_Inps(X, Transients)
    if What == 'RXY':
        return RemoveTransients_Res(ResStates, Transients), RemoveTransient_Inps(X, Transients), RemoveTransient_Outs(Y, Transients)
