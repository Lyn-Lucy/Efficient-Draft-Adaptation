"""
Multi-GPU parallel feature extraction scheduler.

Splits the dataset into N shards and launches one ge_data.py worker per GPU.

Usage:
    python allocation.py \
        --datafile /path/to/dataset.json \
        --modelname /path/to/TargetLLM \
        --outdir /path/to/eagle_datas/my_dataset_mufp16 \
        --total 66720 \
        --gpus 0 1 2 3 4 5 6 7

Arguments
---------
--total     : Total number of samples in the dataset.
              (e.g., 66720 for DeepMath-68k-generated, 68000 for ShareGPT)
--gpus      : Space-separated list of GPU IDs to use.
              Default: all 8 GPUs (0~7). You can use a subset, e.g. --gpus 0 1 2 3
--data_type : "sharegpt" (default) or "custom". Passed through to ge_data.py.
--outdir    : Output root directory. .ckpt files will be stored under
              <outdir>/0/, <outdir>/1/, ... (one sub-folder per GPU worker).
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Parallel feature extraction scheduler")
parser.add_argument("--datafile", type=str, required=True,
                    help="Path to input JSON dataset")
parser.add_argument("--modelname", type=str, required=True,
                    help="Path to target LLM (HuggingFace format)")
parser.add_argument("--outdir", type=str, required=True,
                    help="Root output directory for extracted features")
parser.add_argument("--total", type=int, required=True,
                    help="Total number of samples in the dataset")
parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)),
                    help="GPU IDs to use (default: 0 1 2 3 4 5 6 7)")
parser.add_argument("--data_type", type=str, default="sharegpt",
                    choices=["sharegpt", "custom"])
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# Print config
print("=" * 60)
print("EDA Feature Extraction — Parallel Scheduler")
print("=" * 60)
print(f"Dataset  : {args.datafile}")
print(f"Model    : {args.modelname}")
print(f"Output   : {args.outdir}")
print(f"Total    : {args.total} samples")
print(f"GPUs     : {args.gpus}")
print("=" * 60)


def split_range(total, n):
    """Split [0, total) evenly into n shards, return list of (start, end) tuples."""
    base = total // n
    extra = total % n
    intervals = []
    prev = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        intervals.append((prev, prev + size))
        prev += size
    return intervals


def run_command(cmd):
    print(f"[RUN] {cmd}")
    os.system(cmd)


ge_data_script = os.path.join(os.path.dirname(__file__), "ge_data.py")
splits = split_range(args.total, len(args.gpus))
commands = []

for i, (start, end) in enumerate(splits):
    gpu = args.gpus[i]
    cmd = (
        f"python {ge_data_script} "
        f"--start {start} --end {end} "
        f"--index {i} "
        f"--gpu_index {gpu} "
        f"--outdir {args.outdir} "
        f"--datafile {args.datafile} "
        f"--modelname {args.modelname} "
        f"--data_type {args.data_type}"
    )
    commands.append(cmd)

print(f"\nLaunching {len(commands)} workers ...\n")
with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for cmd in commands:
        executor.submit(run_command, cmd)

print("\nAll workers finished.")
