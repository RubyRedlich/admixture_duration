import numpy as np
from itertools import combinations_with_replacement, product
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Helper functions for building the model 

def _list2types(haps):
    types = np.zeros(8)
    for h in haps:
        types[h-1] += 1
    return types

def _flatten_tuple(nested):
    flat = tuple(int(item) for sub in nested for item in sub)
    return flat

def _check_pops(pops1, pops2, inds, constraints=None):
    # check constraints in pops1 (only used for coal, not recomb)
    # MAKE SURE PASSING ORDER REFLECTS THIS PROPERLY
    for i, pid in enumerate(inds):
        if pops1[pid] != pops2[i]:
            return False
    if constraints: # not None, not empty list
        # all vals of pops1 at the indicies in constraints[i] must match
        s = pops1[constraints[0]]
        for j in constraints:
            if pops1[j] != s:
                return False
    return True

def plot_pdf(x, y, pdf, ax=None, kind='surface', cmap='Blues', **kwargs):
    """
    kind: 'heatmap', 'surface', or 'wireframe'
    """
    if kind == 'heatmap':
        if ax is None:
            fig, ax = plt.subplots()
        im = ax.pcolormesh(x, y, pdf, cmap=cmap, **kwargs)
        plt.colorbar(im, ax=ax, label='PDF')

    else:
        X, Y = np.meshgrid(x, y)
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        if kind == 'surface':
            surf = ax.plot_surface(X, Y, pdf, cmap=cmap, **kwargs)
            plt.colorbar(surf, ax=ax, label='PDF')
        elif kind == 'wireframe':
            ax.plot_wireframe(X, Y, pdf, **kwargs)
        ax.set_zlabel('PDF')

    ax.set_xlabel('T1')
    ax.set_ylabel('T2')
    ax.set_title('PDF')
    return ax

class TwoLocusMarkovModel:

    def _assign_populations(self):
        self.states = {}
        state_idx = 1
        for key, val in self.two_locus_states.items():
            types = val[1]
            deme_string = ''.join([str(i+1) for i in range(self.num_demes)])
            # for each type, get the ways it can be distribution across num_demes demes
            stars_and_bars = [((i+1,)*int(n), list(combinations_with_replacement(deme_string, int(n)))) for i, n in enumerate(types) if n > 0]
            stars_and_bars = [[(h,p) for p in pop_states] for h, pop_states in stars_and_bars]
            # get all combinations across types
            combos = list(product(*stars_and_bars))
            # from combos, add each state to states dictionary
            for el in combos:
                haps, pops = zip(*el)
                self.states[state_idx] = (_flatten_tuple(haps), _flatten_tuple(pops))
                state_idx += 1

    def _migration_step(self, s1, s2):
        haps1, pops1 = s1
        haps2, pops2 = s2
        # migrations can only happen between the same two_locus_states
        # changing two_locus_state requires recombination or coalescence and only one event can happen per transition
        if haps1 != haps2:
            return 0
        else:
            # convert to numpy arrays for easy broadcasting operations
            haps1, pops1, haps2, pops2 = (np.array(el) for el in [haps1, pops1, haps2, pops2])
            migs = (pops1!=pops2)
            nmoves = migs.sum()
            if nmoves != 1: # zero or > 1 moves occur with rate 0 since only one migration can take place per step
                return 0
            else:
                moving_hap = haps1[migs][0]
                source = pops1[migs][0]
                dest = pops2[migs][0]
                m = (pops1[haps1==moving_hap] == source).sum() # multiplier if > 1 haplotype in the same population could move
                return self.migration_rates[source-1, dest-1] * m

    def _recomb_step(self, s1, s2):
        haps1, pops1 = s1
        haps2, pops2 = s2
        i = self.inverse_two_locus_states[haps1]
        j = self.inverse_two_locus_states[haps2]
        if (i,j) in self.recombination_transitions:
            check_inds = self.recombination_transitions[(i,j)]['pop_inds']
            valid = np.array([_check_pops(pops1, pops2, inds) for inds in check_inds])
            if np.any(valid):                 
                p = pops1[self.recombination_transitions[(i,j)]['event_pop'][np.where(valid)[0][0]]]
                return self.recombination_transitions[(i,j)]['rate'], p
            else:
                return 0, None
        else:
            return 0, None

    def _coal_step(self, s1, s2):
        haps1, pops1 = s1
        haps2, pops2 = s2
        i = self.inverse_two_locus_states[haps1]
        j = self.inverse_two_locus_states[haps2]
        if (i,j) in self.coalescence_transitions:
            check_inds = self.coalescence_transitions[(i,j)]['pop_inds']
            constraints = self.coalescence_transitions[(i,j)]['constraints']
            if constraints:
                assert len(check_inds) == len(constraints)
                valid = np.array([_check_pops(pops2, pops1, inds, cnsts) for inds, cnsts in zip(check_inds, constraints)])
            else:
                valid = np.array([_check_pops(pops2, pops1, inds) for inds in check_inds])
            if np.any(valid):                 
                p = pops2[self.coalescence_transitions[(i,j)]['event_pop'][np.where(valid)[0][0]]]
                return self.coalescence_transitions[(i,j)]['rate'], p
            else:
                return 0, None
        else:
            return 0, None

    def _build_Q_matrix(self):
        nstates = len(self.states)
        self.Q = np.zeros((nstates, nstates))
        for i in range(nstates):
            for j in range(nstates):
                if i!= j:
                    s1 = self.states[i+1]
                    s2 = self.states[j+1]

                    m = self._migration_step(s1, s2)
                    c, cp = self._coal_step(s1, s2) 
                    r, rp = self._recomb_step(s1, s2)

                    # CHECK ONLY ONE STEP OCCURS 
                    if m > 0:
                        assert c == 0 and r == 0
                        self.Q[i,j] = m * 2*self.deme_sizes[0] # scale rates by size of pop1
                    elif c > 0:
                        assert m == 0 and r == 0
                        self.Q[i,j] = c * (1/(2*self.deme_sizes[cp-1])) * 2*self.deme_sizes[0]
                    elif r > 0:
                        assert c == 0 and m == 0
                        rho = 4*self.deme_sizes[0]*self.recombination_rate
                        self.Q[i,j] = r * rho 
        np.fill_diagonal(self.Q, -self.Q.sum(axis=1)) # set diagonal so rows sum to 1

    def __init__(self, num_demes, migration_rates, deme_sizes, recombination_rate):
        # Build the model 
        # The following can be hard-coded because it is a property of any two-locus model and does not depend on the demographic model!
        # Store the types of haplotypes making up the two locus states
        self.num_demes = num_demes
        if migration_rates is None:
            migration_rates = np.zeros((num_demes, num_demes))
        else:
            if num_demes > 1:
                assert migration_rates.shape[0] == migration_rates.shape[1] and migration_rates.shape[0] == num_demes
        self.migration_rates = migration_rates
        assert len(deme_sizes) == num_demes
        self.deme_sizes = deme_sizes
        self.recombination_rate = recombination_rate
        self.haplotypes = {
            1: "OO",
            2: "O-",
            3: "-O",
            4: "X-",
            5: "-X"
        }
        # Define the 6 two locus states from the SMC' paper model
        self.two_locus_states = {
            1: [(1, 1), _list2types((1,1))],
            2: [(1, 2, 3), _list2types((1,2,3))],
            3: [(2, 2, 3, 3), _list2types((2, 2, 3, 3))],
            4: [(3, 3, 4), _list2types((3, 3, 4))],
            5: [(2, 2, 5), _list2types((2, 2, 5))],
            6: [(4, 5), _list2types((4, 5))],
        }
        # Create an inverse dictionary for fast lookup
        self.inverse_two_locus_states = {val[0]:key for key,val in self.two_locus_states.items()}
        # Store recombination and coalescence transitions for easily buidling the Q matrix
        # pop_inds are the INDICES in the FIRST tuple that should match the SECOND tuple
        # event_pop is the index in the FIRST population tuple in which the recombination event occurred
        self.recombination_transitions = {
            (1, 2):{
                'rate':1, 
                'pop_inds':[(0,1,1), (1,0,0)], 
                'event_pop':[1, 0, 1, 0]
                }, # (1,1) -> (1,2,3)
            (2, 3):{
                'rate':0.5, 
                'pop_inds':[(1,0,2,0), (1,0,0,2), (0,1,2,0), (0,1,0,2)], 
                'event_pop':[0,0,0,0]
                } # (1,2,3) -> (2,2,3,3)
        }
        # pop_inds contatins the INDICES in the SECOND tuple that must match the FIRST tuple
        # event_pop is the index in the SECOND tuple in which the coalescence event has occurred, it is the same length as pop_inds (the pop_inds arrangement that matches gives the event_pop)
        # constraints are INDICES in the SECOND tuple that must match as well (due to combining some states when we don't care about linkage anymore, certain transitions require additional constraints)
        # TO DO: maybe there is a more robust way to define these rules that doesn't rely on hard-coding in case I have made mistakes!
        self.coalescence_transitions = {
            (1,6):{
                'rate':1, 
                'pop_inds':[(0,0)], 
                'event_pop':[0], 
                'constraints':[(0,1)]
                }, # (1,1) -> (4,5)
            (2,1):{
                'rate':1, 
                'pop_inds':[(0,1,1), (1,0,0)], 
                'event_pop':[1,0], 
                'constraints':[]
                }, # (1,2,3) -> (1,1)
            (3,2):{
                'rate':4, 
                'pop_inds':[(1,0,2,0), (1,0,0,2), (0,1,2,0), (0,1,0,2)], 
                'event_pop':[0,0,0,0], 
                'constraints':[]
                }, # (2,2,3,3) -> (1,2,3)
            (2,4):{
                'rate':1, 
                'pop_inds':[(2,2,1), (2,2,0)], 
                'event_pop':[2,2], 
                'constraints':[(0,2), (1,2)]
                }, # (1,2,3) -> (3,3,4)
            (2,5):{
                'rate':1, 
                'pop_inds':[(2,1,2), (2,0,2)], 
                'event_pop':[2,2], 
                'constraints':[(0,2), (1,2)]
                }, # (1,2,3) -> (2,2,5)
            (3,4):{
                'rate':1, 
                'pop_inds':[(2,2,0,1), (2,2,1,0)], 
                'event_pop':[2,2], 
                'constraints':[]
                }, # (2,2,3,3) -> (3,3,4)
            (3,5):{
                'rate':1, 
                'pop_inds':[(0,1,2,2),(1,0,2,2)], 
                'event_pop':[2,2], 
                'constraints':[]
                }, # (2,2,3,3) -> (2,2,5)
            (4,6):{
                'rate':1, 
                'pop_inds':[(1,1,0)], 
                'event_pop':[1], 
                'constraints':[]
                }, # (3,3,4) -> (4,5)
            (5,6):{
                'rate':1, 
                'pop_inds':[(0,0,1)], 
                'event_pop':[0], 
                'constraints':[]
                } # (2,2,5) -> (4,5)
        }
        # Create the self.states dictionary which extends the two-locus markov model to include all deme combinations the haplotypes can occupy in each state
        self._assign_populations() 
        self.inverse_states = {val:key for key,val in self.states.items()}
        # Create the transition matrix
        self._build_Q_matrix()
        # Set pdf to None
        self.joint_pdf = None, None

    def eval_joint_pdf(self, initial_state, min_time=1, max_time=80_000, nintervals=2_000, show_plot=False, kind='heatmap', cmap='Blues'):
        # need to scale time in the same was as in Q!!! (by 2*deme_sizes[0])
        time_bins = np.linspace(min_time, max_time, nintervals) / (2*self.deme_sizes[0]) 
        P = expm(self.Q * time_bins[:,None,None]) 
        pdf = np.zeros((nintervals, nintervals))

        # get the initial state
        assert initial_state in self.inverse_states
        s0 = self.inverse_states[initial_state] - 1

        # Define collections of states - subtract one because states in the model (keys) are 1 indexed but python is zero indexed
        R0 = np.array([key for key,(hap, pop) in self.states.items() if self.inverse_two_locus_states[hap] in {1}]) - 1
        CB = np.array([key for key,(hap, pop) in self.states.items() if self.inverse_two_locus_states[hap] in {6}]) - 1
        Rplus = np.array([key for key,(hap, pop) in self.states.items() if self.inverse_two_locus_states[hap] in {2,3}]) - 1
        CL = np.array([key for key,(hap, pop) in self.states.items() if self.inverse_two_locus_states[hap] in {4}]) - 1
        CR = np.array([key for key,(hap, pop) in self.states.items() if self.inverse_two_locus_states[hap] in {5}]) - 1

        # Compute the discretized PDF
        for i in range(nintervals):
            for j in range(nintervals):
                if i > j:
                    pdf[i,j] = (P[j, s0, Rplus] @ self.Q[Rplus][:,CR] @ P[i-j,CR][:,CR] @ self.Q[CR][:,CB]).sum()
                elif i < j:
                    pdf[i,j] = (P[i, s0, Rplus] @ self.Q[Rplus][:,CL] @ P[j-i,CL][:,CL] @ self.Q[CL][:,CB]).sum()
                else:
                    pdf[i,j] = (P[i, s0, R0] @ self.Q[R0][:,CB]).sum()

        # Plot if desired
        if show_plot: 
            ax = plot_pdf(time_bins, time_bins, pdf, kind=kind, cmap=cmap)
            plt.show()

        self.joint_pdf = (pdf, time_bins)

    def get_marginal_pdf(self):
        # The joint pdf is a mixture of a 1D pdf on the diagonal and a 2D joint pdf elsewhere
        # Assume uniformly spaced time bins!
        joint_pdf, time_bins = self.joint_pdf
        dt = time_bins[1] - time_bins[0]
        diag = np.diag(joint_pdf)
        off_diag = joint_pdf.sum(axis=1) - diag
        pdf = diag * dt + off_diag * dt**2
        return pdf

