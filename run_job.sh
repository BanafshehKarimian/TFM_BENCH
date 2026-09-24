#!/bin/bash

set -e

echo "=================================================="
echo "Slurm Job ID : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Node         : $SLURMD_NODENAME"
echo "Job index    : $JOB_INDEX"
echo "Jobs file    : $JOBS_FILE"
echo "=================================================="

module purge
module load python/3.11
module load StdEnv/2023  intel/2023.2.1 cuda/11.8
module load cuda

source $SCRATCH/py311_tfm/bin/activate

cd /home/banafsh7/projects/def-egranger/banafsh7/TFM_BENCH

echo
echo "GPU:"
nvidia-smi

echo
echo "Starting benchmark..."
echo

python run_pair.py \
    --manifest "$JOBS_FILE" \
    --index "$JOB_INDEX"