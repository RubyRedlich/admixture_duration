import os
import numpy as np
import pickle
import json
from glob import glob

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

from simulation_fns import *
from summary_stat_fns import *

# worker must be defined outside of main!
def worker(params):
    model, batch_id, n_reps, sequence_length, pairs_per_tree, inter_tree_dist, n_sets, max_time, nintervals, seeds = params

    results = []
    for i in range(n_reps):
        ts = simulate_genomes(
            dem_params=model["dem_params"],
            sequence_length=sequence_length,
            ancestry_seed=seeds[i],
            mutation_seed=seeds[i]
        )
        counts = sample_pair_coalescence_counts(
            ts, 
            ["pop1"], # sample pairs from pop1 
            inter_tree_dist, 
            pairs_per_tree, 
            n_sets, 
            seeds[i], 
            nintervals=nintervals, min_time=1, max_time=max_time) # (len(num_trees), n_sets, nintervals-1)
        
        results.append(counts)

    return {"model": model["name"], "batch": batch_id, "counts": results}

def main(args):
    # Read in the models for simulation
    files = glob(args.model_path)
    models = []
    for file in files:
        with open(file, "r") as f:
            models.append(json.load(f))

    TOTAL_REPS = args.total_reps
    BATCH_SIZE = args.batch_size
    N_WORKERS = args.nworkers  

    # generate unique seeds for each task
    rng = np.random.default_rng(args.seed)  # top-level seed for reproducibility
    n_tasks = len(models) * (TOTAL_REPS // BATCH_SIZE)
    all_seeds = rng.integers(1, 2**31, size=n_tasks * BATCH_SIZE)

    # define the tasks
    tasks = [
        (model, batch_id, BATCH_SIZE, 
        args.sequence_length, args.num_pairs_per_tree, args.inter_tree_dist, args.num_sample_sets, 
        args.max_time, args.nintervals, all_seeds[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].tolist())
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
    # if args.save_all:
    print("Saving all results")
    with open(args.out_prefix+'_grouped.pkl', "wb") as f:
        pickle.dump(grouped, f)

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="model_*.json") 
    parser.add_argument('--sequence_length', type=int, default=20_000_000) 
    parser.add_argument('--num_pairs_per_tree', type=int, default=5)
    parser.add_argument('--inter_tree_dist', type=int, default=100_000) 
    parser.add_argument('--num_sample_sets', type=int, default=20) 
    parser.add_argument('--max_time', type=int, default=80_000) 
    parser.add_argument('--nintervals', type=int, default=500) 
    parser.add_argument('--total_reps', type=int, default=100) 
    parser.add_argument('--batch_size', type=int, default=25) 
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--save_all', action="store_true")
    # parser.add_argument('--save_trees', action="store_true")
    parser.add_argument('--out_prefix', type=str, default="simulations_1") 
    args = parser.parse_args()

    main(args)
