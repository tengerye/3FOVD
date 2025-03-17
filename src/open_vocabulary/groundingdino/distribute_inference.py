#!/usr/bin/env python3

import os
import subprocess
import time
from collections import deque
import argparse

import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(description="Distribute DINO inference across multiple GPUs and nodes.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of GPUs available on this machine."
    )
    parser.add_argument(
        "--max_processes_per_gpu",
        type=int,
        default=10,
        help="Maximum number of parallel processes per GPU."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="captions.csv",
        help="Path to the CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./groundingdino",
        help="Directory where output files are stored."
    )
    parser.add_argument(
        "--node_size",
        type=int,
        default=1,
        help="Total number of nodes (machines)."
    )
    parser.add_argument(
        "--node_idx",
        type=int,
        default=0,
        help="Index of the current node (0-based)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    NUM_GPUS = args.num_gpus
    MAX_PROCESSES_PER_GPU = args.max_processes_per_gpu
    CSV_PATH = args.csv_path
    OUTPUT_DIR = args.output_dir

    NODE_SIZE = args.node_size
    NODE_IDX = args.node_idx

    print(f"Creating {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH, header=0)
    TOTAL_CAPTIONS = len(df)

    # Validate node parameters
    if NODE_IDX < 0 or NODE_IDX >= NODE_SIZE:
        raise ValueError(f"Invalid node_idx: {NODE_IDX}, must be in [0, {NODE_SIZE - 1}].")

    # We'll generate an output file for each row index like:
    #   results_caption_{row_idx}.json
    # That will be passed to dino.py via --output_file.
    COMMAND_TEMPLATE = (
        "CUDA_VISIBLE_DEVICES={gpu_idx} python demo/main.py "
        "{row_idx} "  # positional argument for the caption row
        "--csv_path {csv_path} "
        "--output_file {output_dir}/{row_idx}.json"
    )

    # Create a list of row indices for this node only
    # (caption_row_idx in [0..TOTAL_CAPTIONS-1], but only those that
    # satisfy row_idx % node_size == node_idx)
    eligible_rows = [
        i for i in range(TOTAL_CAPTIONS)
        if i % NODE_SIZE == NODE_IDX
    ]

    print(f"Candidate rows {TOTAL_CAPTIONS}: {eligible_rows}")

    # Convert to a queue
    jobs_queue = deque(eligible_rows)

    # Keep track of running processes for each GPU
    #   running_procs[gpu_idx] = [ (Popen_obj, row_idx), ... ]
    running_procs = [[] for _ in range(NUM_GPUS)]

    while jobs_queue or any(running_procs[g] for g in range(NUM_GPUS)):
        # 1. Clean up finished processes
        for gpu_idx in range(NUM_GPUS):
            still_running = []
            for (proc, row_idx) in running_procs[gpu_idx]:
                if proc.poll() is None:
                    # Process is still running
                    still_running.append((proc, row_idx))
                else:
                    # Process has finished
                    print(f"[Node {NODE_IDX} GPU {gpu_idx}] Completed caption row {row_idx}.")
            running_procs[gpu_idx] = still_running

        # 2. Assign new jobs if there's capacity
        for gpu_idx in range(NUM_GPUS):
            while len(running_procs[gpu_idx]) < MAX_PROCESSES_PER_GPU and jobs_queue:
                row_idx = jobs_queue[0]  # peek at the front of the queue
                out_file = f"{OUTPUT_DIR}/{row_idx}.json"
                out_path = os.path.abspath(out_file)

                # Check if this file already exists
                if os.path.exists(out_path):
                    # Skip if we already have results
                    print(f"[Node {NODE_IDX} GPU {gpu_idx}] Skipping row {row_idx}, '{out_file}' exists.")
                    jobs_queue.popleft()
                    if not jobs_queue:
                        break
                    continue

                # If file doesn't exist, let's run the job
                cmd = COMMAND_TEMPLATE.format(
                    gpu_idx=gpu_idx,
                    row_idx=row_idx,
                    csv_path=CSV_PATH,
                    output_dir=OUTPUT_DIR
                )
                print(f"[Node {NODE_IDX} GPU {gpu_idx}] Starting caption row {row_idx}.")
                proc = subprocess.Popen(cmd, shell=True)
                running_procs[gpu_idx].append((proc, row_idx))
                jobs_queue.popleft()

                if not jobs_queue:
                    break

        time.sleep(180)  # wait briefly before checking again

    print(f"[Node {NODE_IDX}] All jobs completed or skipped.")


if __name__ == "__main__":
    main()
