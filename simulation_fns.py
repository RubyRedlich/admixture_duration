# Functions for simulating admixture models 
import demes
import demesdraw
import msprime
import os

# Use demes to define demographic models - allow for flexible model specification from a yaml file + interfaces easily with moments

# Shortcut function to build an admixture demes model
def build_2pop_admixture_demes(Na, N1, N2, Tsplit, Tadmix_start, Tadmix_end, m12, m21):
    # Create a demes builder
    b = demes.Builder(
        description="2 pop admixture",
        time_units="generations" #??
    )
    # Add populations
    b.add_deme("ancestral", 
               epochs=[dict(end_time=Tsplit, start_size=Na)]
              )
    # defining end time will replace this population with the admixed population (OLD APPROACH)
    pulse = (Tadmix_start - Tadmix_end <= 1)
    # pop1_dict = dict(start_size=N1) if m21 == 0 or not pulse else dict(start_size=N1, end_time=Tadmix_start)
    pop1_dict = dict(start_size=N1)
    b.add_deme("pop1", 
               ancestors=["ancestral"], 
               epochs=[pop1_dict]
              )
    # pop2_dict = dict(start_size=N2) if m12 == 0 or not pulse else dict(start_size=N2, end_time=Tadmix_start)
    pop2_dict = dict(start_size=N2)
    b.add_deme("pop2", 
               ancestors=["ancestral"], 
               epochs=[pop2_dict]
              )
    if pulse:
        # # defining a population with multiple ancestors will create an admixed population (recommended approach)
        # if m21 > 0:
        #     b.add_deme("pop1_admix", 
        #             ancestors=["pop1", "pop2"], 
        #             proportions=[1-m21, m21], 
        #             start_time=Tadmix_start, 
        #             epochs=[dict(start_size=N1)])
        # if m12 > 0:
        #     b.add_deme("pop2_admix", 
        #             ancestors=["pop2", "pop1"], 
        #             proportions=[1-m12, m12], 
        #             start_time=Tadmix_start, 
        #             epochs=[dict(start_size=N2)])
        # use a pulse event to not change the deme names and be more consistent with the continuous migration case
        if m21 > 0:
            b.add_pulse(
                sources=["pop2"],
                dest="pop1",
                proportions=[m21],
                time=Tadmix_start
            )
        if m12 > 0:
            b.add_pulse(
                sources=["pop1"],
                dest="pop2",
                proportions=[m12],
                time=Tadmix_start
            )
    # Simulate continous migration over [Tadmix_start, Tadmix_end] interval at rate pop1 -m12-> pop2 and pop1 <-m21- pop2
    else:
        if m12 > 0:
            # source and dest are forwards in time
            b.add_migration(rate=m12, source="pop1", dest="pop2", start_time=Tadmix_start, end_time=Tadmix_end)
        if m21 > 0:
            b.add_migration(rate=m21, source="pop2", dest="pop1", start_time=Tadmix_start, end_time=Tadmix_end)

    # Create the graph from b
    g = b.resolve()

    return g

# Function to simulate from the demographic model
def simulate_genomes(
    demes_model_yaml = None, 
    # if no yaml provided, must pass a function that builds the demes model and corresponding demographic parameters
    dem_params = {"Na":10000, "N1":10000, "N2":2500, "Tsplit":20000, "Tadmix_start":2000, "Tadmix_end":1999, "m12":0, "m21":0.05},
    build_demes_fn = build_2pop_admixture_demes,
    samples = {"pop1":20}, # can be an integer, dictionary, or list of msprime.SampleSet objs
    ancestry_seed = 42,
    mutation_seed = 42,
    recombination_rate = 1e-8,
    recombination_map = None, # recombination map file
    mutation_rate = 125e-10, 
    sequence_length = 2e6, # 2Mb
    record_migrations = True, 
    num_replicates = 1,
    save_ts = False,
    outdir = None # must be a valid directory if save_ts = True
):
    
    # build demes graph from yaml file or function
    if demes_model_yaml:
        pass # TO DO: implement build from yaml
    else:
        assert(isinstance(dem_params, dict))
        # TO DO: check dem_params keys match build_demes_fn function signature
        graph = build_demes_fn(**dem_params)

    # build msprime demography from demes
    demography = msprime.Demography.from_demes(graph)

    tree_sequences = []
    for rep in range(num_replicates):
        # simulate ancestry
        ts = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate,
            record_migrations=record_migrations, # makes it easier to find introgressed tracts
            random_seed=ancestry_seed+rep
        )
        # simulate mutations
        ts = msprime.sim_mutations(
            ts,
            rate=mutation_rate,
            random_seed=mutation_seed+rep
        )
        tree_sequences.append(ts)
        # save tree sequences
        if save_ts:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            ts.dump(os.path.join(outdir, f"ts_{rep+1}"))

    # return tree sequences
    if num_replicates > 1:
        return tree_sequences # return the list of tree sequence replicates
    else:
        return tree_sequences[0] # return the tree sequence object
