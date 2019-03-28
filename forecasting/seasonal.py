import numpy as np


def seascomp(period, harmComponents):
    p = len(harmComponents)
    n = 2*p
    F = np.zeros([n, 1])
    F[0:n:2] = 1
    G = np.zeros([n, n])

    for j in range(p):
        c = np.cos(2*np.pi*harmComponents[j]/period)
        s = np.sin(2*np.pi*harmComponents[j]/period)
        idx = 2*j
        G[idx:(idx+2), idx:(idx+2)] = np.array([[c, s],[-s, c]])

    return [F, G]


def createFourierToSeasonalL(period, harmComponents, Fseas, Gseas):
    p = len(harmComponents)
    L = np.zeros([period, 2*p])
    L[0,:] = Fseas.reshape(-1)
    for i in range(1, period):
        L[i,:] = L[i-1,:] @ Gseas
        
    return L  
    

def fourierToSeasonal(mod):
    phi = mod.L @ mod.m[mod.iseas]
    var = mod.L @ mod.C[np.ix_(mod.iseas, mod.iseas)] @ mod.L.T
    return phi, var


def fourierToSeasonalFxnl(L, m, C, iseas):
    phi = L @ m[iseas]
    var = L @ C[np.ix_(iseas, iseas)] @ L.T
    return phi, var


