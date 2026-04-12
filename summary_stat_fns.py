import numpy as np
import tskit

# Functions for computing summary statistics and their likelihoods from the simulated data/models

# Get the SFS from the simulated data

# Get the ALD from the simulated data

# Get the pairwise coalescence times from the simulated data
def pair_coalescence_counts(ts, window_size=10000, nintervals=256, min_time=np.exp(3), max_time=np.exp(14), time_scale="linear"):
    """
    Each tree sequence represents the genealogy of a non-recombining unit.
    To obtain a good estimate of the number of pairwise coalescence events occurring within a time interval, we want to sample multiple trees.
    Adjacent/nearby trees are correlated due to LD and we don't want to count the same coalescence event multiple times because it appears in adjacent trees.
    One solution is to sample trees some distance apart to make double counting unlikely.
    Alternatively, tskit normalizes by tree span. 
    Normalizing by tree span represents the PROBABILITY OF SAMPLING THAT TREE within that window. 
    It DOES NOT mean that different trees contribute differently to the coalescence counts depending on their lengths.
    Therefore, windows should be approx. the size of the distance one would use for independent sampling (~10-100kb)!
    Normalizing by tree span then represents the expected number of coalescence events in a given time interval if sampling one tree from that window!
    """
    # Check inputs
    # TO DO: make this an enum?
    if time_scale not in ["log", "linear"]:
        print("invalid time-scale, using linear")
        time_scale = "linear"

    # Get the time intervals (equally spaced on a log scale)
    min_time = min_time if min_time > 0 else 1
    if time_scale == "log":
        time_windows = np.exp(np.linspace(np.log(min_time), np.log(max_time), nintervals)) 
    else:
        time_windows = np.linspace(min_time, max_time, nintervals)

    # Get the sample sets
    name2id = {p.metadata["name"]:p.id for p in ts.populations()}
    nodes = [ts.node(n) for n in range(ts.num_nodes)]
    popnames = list(name2id.keys())
    sample_sets = [[n.id for n in nodes if n.time==0 and n.population==name2id[popname]] for popname in popnames]   

    # Remove empty sample sets
    popnames = [popnames[i] for i in range(len(sample_sets)) if len(sample_sets[i]) > 0]
    sample_sets = [sample_sets[i] for i in range(len(sample_sets)) if len(sample_sets[i]) > 0]

    # Get all state pairs
    all_states = [(i, j) for i in range(len(sample_sets)) for j in range(i, len(sample_sets))]
    # this is okay because sample_sets is in the same order as popnames by construction
    all_states_names = [(popnames[i], popnames[j]) for i in range(len(sample_sets)) for j in range(i, len(sample_sets))]

    # Get the bp windows 
    bp_windows = np.arange(0, ts.sequence_length+window_size, window_size, dtype=np.int32)
    bp_windows[-1] = int(min(bp_windows[-1], ts.sequence_length))

    # Get the pairwise coalescence counts
    counts = ts.pair_coalescence_counts(
        sample_sets=sample_sets,
        indexes=all_states,
        span_normalise=True,
        time_windows=time_windows,
        windows=bp_windows
    ) # (windows x indexes x time_intervals)

    # sum counts over windows
    counts = counts.sum(axis=0) # (indexes x time_intervals)

    # TO DO: include options to sample single trees spaced sample_bp_dist apart rather than normalizing by tree span
    
    return counts, all_states_names, time_windows[1:]