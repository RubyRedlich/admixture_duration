import os
import numpy as np
import pickle

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

from simulation_fns import *
from summary_stat_fns import *

# worker must be defined outside of main!
def worker(params):
    model, batch_id, n_reps, sequence_length, max_time, nintervals, seeds = params

    results = []
    for i in range(n_reps):
        ts = simulate_genomes(
            dem_params=model["dem_params"],
            sequence_length=sequence_length,
            ancestry_seed=seeds[i],
            mutation_seed=seeds[i]
        )
        counts, _, _ = pair_coalescence_counts(
            ts,
            min_time=0, max_time=max_time,
            nintervals=nintervals, window_size=50e3
        )
        results.append(counts)

    return {"model": model["name"], "batch": batch_id, "counts": results}

def main(args):
    # Define the models for simulation
    ADMIX_START = 2000
    ALPHA = 0.05
    admix_vals = [(0,ADMIX_START-1), (ALPHA,ADMIX_START-1), (ALPHA,ADMIX_START-50), (ALPHA,ADMIX_START-500), (ALPHA,ADMIX_START-1000), (ALPHA,0)]
    models = [{
        "name":f"model_{i+1}",
        "dem_params":{
            "Na":10000, 
            "N1":10000, 
            "N2":2500, 
            "Tsplit":20000, 
            "Tadmix_start":ADMIX_START, 
            "Tadmix_end":t, 
            "m12":0, 
            "m21":alpha/(ADMIX_START-t)
        }
        } for i, (alpha, t) in enumerate(admix_vals)]

    TOTAL_REPS = args.total_reps
    BATCH_SIZE = args.batch_size
    N_WORKERS = args.nworkers  

    # generate unique seeds for each task
    rng = np.random.default_rng(args.seed)  # top-level seed for reproducibility
    n_tasks = len(models) * (TOTAL_REPS // BATCH_SIZE)
    all_seeds = rng.integers(1, 2**31, size=n_tasks * BATCH_SIZE)

    # define the tasks
    tasks = [
        (model, batch_id, BATCH_SIZE, args.sequence_length, args.max_time, args.nintervals, all_seeds[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].tolist())
        for i, (model, batch_id) in enumerate(
            [(model, batch_id)
            for model in models
            for batch_id in range(TOTAL_REPS // BATCH_SIZE)]
        )
    ]
    
    print("Simulating in parallel")
    grouped = defaultdict(list)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # .map automatically blocks until all tasks are finished and returns in submission order
        # can iterate through it directly
        for result in executor.map(worker, tasks):
            grouped[result["model"]].extend(result["counts"])

    # grouped is a dictionary mapping each model to a list of TOTAL_REPS counts arrays

    print("Summarizing")
    # for each model, calculate summary statitistics (means and 95% CIs)
    summary = {}
    for model_name, counts_list in grouped.items():
        arr = np.stack(counts_list)  # shape: (TOTAL_REPS, n_states, n_intervals)
        mean  = arr.mean(axis=0)                    # (n_states, n_intervals)
        lower = np.percentile(arr, 2.5,  axis=0)   # (n_states, n_intervals)
        upper = np.percentile(arr, 97.5, axis=0)   # (n_states, n_intervals)
        
        summary[model_name] = {"mean": mean, "lower": lower, "upper": upper}

    print("Saving")
    # save the results
    with open(args.outfile, "wb") as f:
        pickle.dump(summary, f)

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=20_000_000) 
    parser.add_argument('--max_time', type=int, default=80_000) 
    parser.add_argument('--nintervals', type=int, default=500) 
    parser.add_argument('--total_reps', type=int, default=100) 
    parser.add_argument('--batch_size', type=int, default=25) 
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outfile', type=str, default="simulations_1.pkl") 
    args = parser.parse_args()

    main(args)