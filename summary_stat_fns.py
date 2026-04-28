import numpy as np
import tskit

# Functions for computing summary statistics and their likelihoods from the simulated data/models

# Get the SFS from the simulated data

# Get the ALD from the simulated data

# Helper function to sample trees spaced some distance apart in a tree sequence
def get_trees(ts, start_pos, inter_tree_dist, num_trees):
    # Returns a generator that is memory efficient because it doesn't require loading every tree at once in memory
    pos = start_pos
    nt = 0
    while pos < ts.sequence_length and nt < num_trees:
        tree = ts.at(pos)
        yield nt, tree
        pos += inter_tree_dist
        nt += 1

# More preceise control over sampling pairwise coalescence times
def sample_pair_coalescence_counts(ts, pops, inter_tree_dist, pairs_per_tree, n_sets, seed, nintervals=256, min_time=1, max_time=80_000):
    # Sampling will vary the number of trees sampled in order to compare power across different sample sizes
    max_trees = ts.sequence_length // inter_tree_dist
    # num_trees = np.arange(0, max_trees, max_trees // 10, dtype=np.int32)
    # num_trees[0] += 1
    num_trees = np.array([250, 1000, 1750, 2250], dtype=np.int32)
    
    # get rng for sampling
    rng = np.random.default_rng(seed=seed)

    # get the nodes in the pops to sample from
    namemap = {p.metadata['name']:p.id for p in ts.populations()}
    pop_membership = {p:[ts.node(n).id for n in range(ts.num_nodes) if ts.node(n).population==namemap[p] and ts.node(n).time==0] for p in pops}

    # sample
    all_times = []
    for nt in num_trees:
        times = np.zeros((n_sets, nt, pairs_per_tree))
        for s in range(n_sets):
            start_pos = rng.integers(inter_tree_dist)
            for t, tree in get_trees(ts, start_pos, inter_tree_dist, nt):
                # sample pairs from the tree
                for k in range(pairs_per_tree):
                    # sample pair
                    if len(pops) > 1:
                        assert len(pops)==2
                        n1 = rng.choice(pop_membership[pops[0]])
                        n2 = rng.choice(pop_membership[pops[1]])
                        tmrca = tree.tmrca(n1,n2)
                    else:
                        nodes = rng.choice(pop_membership[pops[0]], size=2, replace=False)
                        tmrca = tree.tmrca(nodes[0],nodes[1])
                    times[s, t, k] = tmrca
        times = times.reshape(n_sets, -1) # reshape to combine number of trees/number of pairs axes
        all_times.append(times)

    # perform binning - will make each array in all_times the same size
    min_time = min_time if min_time > 0 else 1
    time_bins = np.linspace(min_time, max_time, nintervals)
    counts = np.array([np.array([np.histogram(row, bins=time_bins)[0] for row in times]) for times in all_times]) # (10, n_sets, len(time_bins)-1)

    # if performance becomes an issue use a np.searchsorted approach instead
    """
    (from Claude)
    indices = np.searchsorted(time_bins, times, side='right') - 1  # shape: same as times
    n_bins = len(time_bins) - 1
    n_rows = times.shape[0]

    counts = np.zeros((n_rows, n_bins), dtype=int)
    valid = (indices >= 0) & (indices < n_bins)

    rows = np.where(valid, np.arange(n_rows)[:, None], 0)  # broadcast row indices
    np.add.at(counts, (rows[valid], indices[valid]), 1)
    """

    return counts

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