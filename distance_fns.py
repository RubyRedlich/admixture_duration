import numpy as np
import msprime
from helper import * 
from scipy.linalg import expm

# Functions for comparing the distances/resolvability between distributions of summary statistics from different models

# Compute likelihood of pairwise coalescence counts given a demographic model

# Compute PMF of pairwise coalescence times given a model (mrpast reimplementation)
## Ignore growth rates for now, TO DO: can add in later
def pair_coal_times_PMF(demography, nintervals=256, min_time=np.exp(3), max_time=np.exp(14)):
    # get the model from the demogrpahy object
    Q, E, epoch_bounds, state_index, pop_index, scale_factor = model_from_demography(demography)
    S = len(state_index) + 1

    # create the time discretization
    min_time = min_time if min_time > 0 else 1
    # TO DO: option for log scale or linear spaced time windows
    time_windows = np.exp(np.linspace(np.log(min_time), np.log(max_time), nintervals)) # not scaled
    time_windows = time_windows / scale_factor # scale by scale_factor = 2*Nancestral
    epoch_bounds = np.array(epoch_bounds) / scale_factor

    # set time_windows as epoch bounds based on which they are closest to
    is_epoch_bound = np.zeros(len(time_windows), dtype=np.bool_)
    for i in range(len(time_windows)-1):
        flag=False
        for e in epoch_bounds:
            if time_windows[i] <= e and e < time_windows[i+1]:
                flag=True
        is_epoch_bound[i] = flag
    epoch_bounds_discrete = time_windows[is_epoch_bound]
    # ensure epoch bounds always start at 0
    # TO DO: make this more robust later!
    if epoch_bounds_discrete[0] != 0: 
        epoch_bounds_discrete = np.insert(epoch_bounds_discrete,0,0)

    # helper function to normalize rows of a matrix
    def rowNorm(X):
        rowsums = X.sum(axis=1, keepdims=True) 
        return X / rowsums

    # initialize
    L = np.zeros((S-1, S-1))
    np.fill_diagonal(L, 1)
    e = 0
    coal_p = np.zeros(S-1)
    
    # loop through time steps
    CDF = np.zeros((S-1,len(time_windows)))
    for k, tk in enumerate(time_windows):
        P = expm(Q[e]*(tk - epoch_bounds_discrete[e]))
        Pl = P[:-1,:-1]
        Pc = P[:-1,-1]
        Pc = L @ Pc
        CDF[:,k] = (coal_p + Pc) - (coal_p * Pc)
        if is_epoch_bound[k]:
            if e < len(E)-1: # if not in the final epoch
                e += 1
                L = rowNorm(L @ Pl) @ E[e]
                coal_p = (coal_p + Pc) - (coal_p * Pc)
    # get PMF by taking differences between CDF intervals
    PMF = np.zeros((S-1,len(time_windows)-1))
    for i in range(CDF.shape[1]-1):
        PMF[:,i] = CDF[:,i+1]-CDF[:,i]
    
    return PMF

# Compute PMF of pairwise coalescence times given a model (my implementation)

# Compute likelihood ratio of models given pairwise coalescence counts

# Numerically calculate the theoretical pairwise coalescence time distribution given a demographic model

# Estimate the KL divergence (or TVD, etc.) between the theoretical pairwise coalescence time distributions