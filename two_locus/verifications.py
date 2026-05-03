import numpy as np 
import demes
import msprime
from markov_model import *
from helper import *

# Code to verify correctness of my two-locus markov model with structure

def expected_analytical_marginal(model='n_island'):
    if model == 'nisland':
        return

def build_two_island_demes_model(deme_sizes, mAB, mBA):
    b = demes.Builder()
    b.add_deme("A", epochs=[dict(start_size=deme_sizes[0])])
    b.add_deme("B", epochs=[dict(start_size=deme_sizes[1])])
    b.add_migration(source="A", dest="B", rate=mAB)
    b.add_migration(source="B", dest="A", rate=mBA)
    g = b.resolve()
    demography = msprime.Demography.from_demes(g)
    return demography

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
    # TO DO: AUTOMATICALLY DECREASE NUMBER OF EPOCHS IN THE MODEL AND PROCEED WITH CALCULATION IN PLACE OF ASSERTION ERROR
    if not epoch_bounds_discrete:
        epoch_bounds_discrete = np.array([0])
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

def compare_marginal_to_single_locus_pdf(
    num_demes, deme_sizes, migration_rates, recombination_rate,
    nintervals=500, min_time=1, max_time=80_000,
    ax=None, s0_two_locus = ((1,1),(1,1)), s0_single_locus = 1,
    ):
    if ax is None:
        fig, ax = plt.subplots()
    # create two-locus model 
    model = TwoLocusMarkovModel(num_demes, migration_rates, deme_sizes, recombination_rate)
    # get joint pdf
    model.eval_joint_pdf(s0_two_locus, nintervals=nintervals, min_time=min_time, max_time=max_time)
    # compute marginal
    mpdf = model.get_marginal_pdf()
    print(f'Marginal density sums to {mpdf.sum()}')
    # create demography 
    dem = build_two_island_demes_model(deme_sizes, migration_rates[0,1], migration_rates[1,0])
    # get sinle locus pdf
    pmf = pair_coal_times_PMF(dem, nintervals=nintervals, min_time=min_time, max_time=max_time)[s0_single_locus]
    # add to plot! 
    time_bins = model.joint_pdf[1]
    ax.plot(time_bins, mpdf, label='two-locus marginal pdf')
    ax.plot(time_bins[1:], pmf, label='single locus pdf')
    return ax

def compare_covariance(
    num_demes, deme_sizes, migration_rates, recombination_rates,
    nintervals=500, min_time=1, max_time=80_000,
    ax=None, s0_two_locus = ((1,1),(1,1))
):
    if ax is None:
        fig, ax = plt.subplots()
    
    covs = []
    rhos = np.array(recombination_rates)*4*deme_sizes[0]
    for r in recombination_rates:
        # create two-locus model 
        model = TwoLocusMarkovModel(num_demes, migration_rates, deme_sizes, r)
        # get joint pdf
        model.eval_joint_pdf(s0_two_locus, nintervals=nintervals, min_time=min_time, max_time=max_time)
        # get covariance
        time_bins = model.joint_pdf[1]
        dt = time_bins[1] - time_bins[0] # assume equally spaced time bins
        covs.append((np.diag(model.joint_pdf[0]) * dt).sum())
    
    # Get the expected curve
    if num_demes < 2:
        # covariance for single population (McVean 2002)
        rhos_exp = np.linspace(0,np.max(rhos),100)
        cov_exp = (18 + rhos_exp) / (18 + 13*rhos_exp + np.square(rhos_exp))
    else:
        pass

    ax.scatter(rhos, covs, label="two locus model P(T1=T2)")
    ax.plot(rhos_exp, cov_exp, label="Expected covariances")
    return ax, rhos, covs, rhos_exp, cov_exp
