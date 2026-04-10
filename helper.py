# Build model from demography object
def order_pair(i,j):
    return (i,j) if i <=j else (j,i)

def model_from_demography(demography):
    # for each epoch: coalescence rates (population sizes), migration rates, instantaneous events (admixture pulses, pop splits) 
    
    # get info per epoch
    dd = demography.debug()
    nepochs = len(dd.epochs)

    # determine number of states (deme pairs (i,j))
    states = set()
    INACTIVE = msprime.demography.PopulationStateMachine.INACTIVE
    ACTIVE = msprime.demography.PopulationStateMachine.ACTIVE
    for epoch in dd.epochs:
        # get the active populations
        active_pops = [p for p in epoch.populations if p.state == ACTIVE]
        # form ordered pairs (to avoid duplicating pairs in a different order)
        pairs = [order_pair(active_pops[i].id, active_pops[j].id) for i in range(len(active_pops)) for j in range(i, len(active_pops))]
        # add to states set
        states.update(pairs)
    states = list(states)
    nstates = len(states)

    # collect info from the demography object
    coal_rates = np.zeros((nepochs, nstates))
    # fill coal_rates with inverse population sizes scaled by 2N_ancestral
    migration_rates = np.array([e.migration_matrix for e in dd.epochs])
    # scale migration rates by 2N_ancestral
    epoch_events = np.zeros((nepochs, nstates))
    # fill epoch events with transition probabilities due to population splits/admixture at epoch boundaries

    # make the Q matrix for each epoch
    Q = np.zeros((nepochs, nstates+1, nstates+1))
    for e in range(nepochs):
        for i in range(nstates):
            for j in range(nstates):
                if i != j: # fill diagonals later to make rows sum to zero
                    # from (i, j) to (k, l) backwards in time
                    k, l = states[i]
                    m, n = states[j]
                    # assume only on migration occurs per time step
                    if (int(m!=k) + int(n!=l) > 1) or (int(m!=l) + int(n!=k) > 1):
                        Q[e,i,j] = 0
                    else:
                        # break into cases