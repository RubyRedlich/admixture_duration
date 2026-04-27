import numpy as np
import msprime
from helper import * 
from scipy.linalg import expm

# Functions for comparing the distances/resolvability between distributions of summary statistics from different models

# Estimate Hellinger distance between two discretized pdfs
def pair_coal_times_H2(P,Q):
    return np.square(np.sqrt(P) - np.sqrt(Q)).sum() / 2

# Estimate the TVD between two discretized pdfs
def pair_coal_times_TVD(P, Q):
    # P and Q should be two pdfs with the SAME time interval discretization!
    return np.abs(P-Q).sum() / 2

# Compute likelihood of pairwise coalescence counts given a demographic model
def pair_coal_times_loglik(counts, pmf, time_scale="linear", total_counts=None):
    # returns \sum_k (log(pmf_k) * counts_k)

    # handle case where the counts and pmf are not the same length
    ncounts = len(counts) if counts.ndim == 1 else counts.shape[-1]
    if len(pmf) > ncounts:
        if time_scale=="linear":
            # TO DO: make this more robust/handle when counts length is not a multiple of pmf length
            step = len(pmf) // ncounts
            indices = np.arange(0, len(pmf), step)
            pmf = np.add.reduceat(pmf, indices)
            pmf = pmf[:ncounts]
        # TO DO: handle different sized time intervals on the log scale (NOT linear)
    # TO DO: fix this depending on dimensions of counts
    elif len(pmf) < len(counts): 
        if time_scale=="linear":
            step = len(counts) // len(pmf)
            indices = np.arange(0, len(counts), step)
            counts = np.add.reduceat(counts, indices)
        # TO DO: handle different sized time intervals on the log scale (NOT linear)

    # downsample counts to the desired total
    # SLOW/not used right now
    if total_counts is not None:
        sampled_counts = np.zeros_like(counts)
        weights = counts / counts.sum(axis=1, keepdims=True)
        if counts.ndim > 1:
            for rep in range(counts.shape[0]):
                for _ in range(total_counts):
                    i = np.random.choice(np.arange(counts.shape[1]), size=1, p=weights[rep])
                    sampled_counts[rep,i] += 1
        else:
            for _ in range(total_counts):
                i = np.random.choice(np.arange(len(counts)), size=1, p=weights)
                sampled_counts[i] += 1
        counts = sampled_counts

    # return the log likelihood
    if counts.ndim > 1:
        return (np.log(pmf) * counts).sum(axis=1)
    else:
        return (np.log(pmf) * counts).sum()

def LR(null_pmf, alt_pmf, null_counts, alt_counts, seed=42, alpha=0.05, total_counts=None):
    """
    The likelihood ratio compares L(obs|alt) / L(obs|null)

    To get a null distribution of LR: L(null_counts | alt_pmf) / L(null_counts | null_pmf)

    To get the distribution of LR: L(alt_counts | alt_pmf) / L(alt_counts | null_pmf)

    """
    ll0_null = pair_coal_times_loglik(null_counts, null_pmf, total_counts=total_counts)
    lla_null = pair_coal_times_loglik(null_counts, alt_pmf, total_counts=total_counts)
    lr_null = 2*(lla_null - ll0_null)
    threshold = np.quantile(lr_null, 1-alpha)

    ll0_alt = pair_coal_times_loglik(alt_counts, null_pmf, total_counts=total_counts)
    lla_alt = pair_coal_times_loglik(alt_counts, alt_pmf, total_counts=total_counts)
    lr_alt = 2*(lla_alt - ll0_alt)
    power = (lr_alt >= threshold).sum() / len(lr_alt)

    return lr_null, lr_alt, threshold, power

# Compute PMF of pairwise coalescence times given a model (mrpast reimplementation)
## Ignore growth rates for now, TO DO: can add in later
def pair_coal_times_PMF(demography, nintervals=256, min_time=np.exp(3), max_time=np.exp(14), time_scale="linear"):
    # get the model from the demogrpahy object
    Q, E, epoch_bounds, state_index, pop_index, scale_factor = model_from_demography(demography)
    S = len(state_index) + 1

    # check parameters
    if time_scale not in ["log", "linear"]:
        print("invalid time-scale, using linear")
        time_scale = "linear"

    # create the time discretization
    min_time = min_time if min_time > 0 else 1
    if time_scale == "log":
        time_windows = np.exp(np.linspace(np.log(min_time), np.log(max_time), nintervals)) # not scaled
    else:
        time_windows = np.linspace(min_time, max_time, nintervals)
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
    # INCLUDING DECREASING EPOCHS IN THE MODEL AND PROCEEDING WITH CALCULATION IN PLACE OF ASSERTION ERROR
    if epoch_bounds_discrete[0] != 0: 
        epoch_bounds_discrete = np.insert(epoch_bounds_discrete,0,0)
    assert(len(epoch_bounds_discrete) == len(epoch_bounds)), "The time discretization is not fine-grained enough to retain all epochs"

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