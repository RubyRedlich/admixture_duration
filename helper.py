import msprime
import numpy as np

# Build model from demography object
def order_pair(i,j):
    return (i,j) if i <=j else (j,i)

def model_from_demography(demography):
    # for each epoch: coalescence rates (population sizes), migration rates, instantaneous events (admixture pulses, pop splits) 
    
    # get info per epoch
    dd = demography.debug()
    nepochs = len(dd.epochs)

    # determine the states (active deme pairs (i,j) per epoch)
    INACTIVE = msprime.demography.PopulationStateMachine.INACTIVE
    ACTIVE = msprime.demography.PopulationStateMachine.ACTIVE
    # get active pops per epoch
    active_pops = [[p for p in epoch.populations if p.state == ACTIVE] for epoch in dd.epochs]
    # get states (active pop pairs per epoch)
    states = list(set(
        [order_pair(active_pops[e][i].id, active_pops[e][j].id) 
        for e in range(nepochs) for i in range(len(active_pops[e])) for j in range(i, len(active_pops[e]))]
    ))
    nstates = len(states)

    # get coalescence rates per epoch
    # there might be a much cleaner way to do this???
    Nancestral = dd.epochs[-1].populations[0].start_size # might not work for very general models
    assert Nancestral > 0
    state_index = {state: i for i, state in enumerate(states)}
    coal_rates = np.zeros((nepochs, nstates))
    for e, epoch in enumerate(dd.epochs):
        for p in active_pops[e]:
            assert p.start_size != 0
            if (p.id, p.id) in state_index:  
                coal_rates[e, state_index[(p.id, p.id)]] = 2*Nancestral / (2*p.start_size) # scale by Nancestral

    # get migration rates per epoch
    migration_rates = np.array([e.migration_matrix for e in dd.epochs])
    migration_rates = np.multiply(migration_rates, 2*Nancestral) # scale by Nancestral

    # get population splits and admixture events
    pop_index = {p.name:p.id for p in demography.populations}
    npops = len(pop_index)
    epoch_events = np.zeros((nepochs, nstates, nstates))
    # recorded events occur at the START of epoch e and should be applied at the end of epoch e-1
    POPSPLIT = msprime.demography.PopulationSplit
    ADMIXTURE = msprime.demography.MassMigration
    def move_lineage(anc, der, x):
        return anc if x in der else x
    def move_lineage_admx(source, dest, alpha, v):
        return [(dest, alpha), (v, 1 - alpha)] if v == source else [(v, 1.0)]
    for e, epoch in enumerate(dd.epochs):
        events = [event for event in epoch.events if isinstance(event,ADMIXTURE) or isinstance(event,POPSPLIT)]
        if len(events) > 0:
            for event in events:
                if isinstance(event,ADMIXTURE):
                    source = pop_index[event.source]
                    dest = pop_index[event.dest]
                    alpha = event.proportion
                    for x, y in states:
                        for new_x, px in move_lineage_admx(source, dest, alpha, x):
                            for new_y, py in move_lineage_admx(source, dest, alpha, y):
                                epoch_events[e][state_index[(x, y)], state_index[order_pair(new_x, new_y)]] += px * py
                elif isinstance(event,POPSPLIT):
                    anc = pop_index[event.ancestral]
                    der = {pop_index[d] for d in event.derived}
                    for x,y in states:
                        mx, my = move_lineage(anc, der, x), move_lineage(anc, der, y)
                        mx, my = order_pair(mx, my)
                        if mx==anc or my==anc:
                            epoch_events[e][state_index[(x,y)], state_index[(mx, my)]] = 1
        else:
            # if there are no events, should be the identity matrix
            np.fill_diagonal(epoch_events[e], 1)
    # check epoch_events rows sum to 1
    for e in range(nepochs):
        assert(epoch_events[e].sum(axis=1).sum() == nstates)

    # make the Q matrix for each epoch
    Q = np.zeros((nepochs, nstates+1, nstates+1))
    for e in range(nepochs):
        for i in range(nstates):
            for j in range(nstates):
                if i != j: # fill diagonals later to make rows sum to zero
                    # from (k, l) to (m, n) backwards in time
                    k, l = states[i]
                    m, n = states[j]
                    # assume only one migration occurs per time step
                    if (m != k) + (n != l) > 1 and (m != l) + (n != k) > 1:
                        Q[e,i,j] = 0
                    else:
                        # go through the cases
                        if k == m and l != n:       # l moves to n
                            Q[e, i, j] = (2**int(k==l))*migration_rates[e, l, n]
                        elif l == n and k != m:     # k moves to m
                            Q[e, i, j] = (2**int(k==l))*migration_rates[e, k, m]
                        elif k == n and l != m:     # l moves to m
                            Q[e, i, j] = (2**int(k==l))*migration_rates[e, l, m]
                        elif l == m and k != n:     # k moves to n
                            Q[e, i, j] = (2**int(k==l))*migration_rates[e, k, n]
                        else:
                            Q[e, i, j] = 0
        # fill in coalescence rates
        Q[e,nstates, nstates] = 0 # coal -> coal = 0
        Q[e,:nstates,nstates] = coal_rates[e]
        # set diagonals so rows sum to zero
        np.fill_diagonal(Q[e], -Q[e].sum(axis=1))

    return Q, epoch_events, states, state_index
