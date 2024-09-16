
def RemoveTransients_Res(ResStates, Transients):
    return ResStates[Transients:,:]

def RemoveTransient_Inps(X, Transients):
    return X[:,Transients:,:]

def RemoveTransient_Outs(Y, Transients):
    return Y[:,Transients:,:]


def TransientRemover(What: str, ResStates, X, Y, Transients: int):

    if What=='RX':
        return RemoveTransients_Res(ResStates, Transients), RemoveTransient_Inps(X, Transients)
    if What == 'RXY':
        return RemoveTransients_Res(ResStates, Transients), RemoveTransient_Inps(X, Transients), RemoveTransient_Outs(Y, Transients)
