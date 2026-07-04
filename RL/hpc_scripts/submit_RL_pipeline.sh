#!/bin/bash
# =============================================================================
# Submit staged RL tuning pipeline with SLURM dependencies
# =============================================================================

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$RL_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
    module load DefaultModules >/dev/null 2>&1 || true
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch command not found."
    echo "Load the SLURM/default modules first or run this script on the HPC login node."
    exit 1
fi

if [ ! -f "hpc_scripts/run_RL_stage1_BO.sh" ]; then
    echo "ERROR: run_RL_stage1_BO.sh not found in $RL_DIR"
    exit 1
fi

if [ ! -f "hpc_scripts/run_RL_stage2_array.sh" ]; then
    echo "ERROR: run_RL_stage2_array.sh not found in $RL_DIR"
    exit 1
fi

echo "Submitting RL Stage 1..."
STAGE1_JOB_ID="$(sbatch --parsable hpc_scripts/run_RL_stage1_BO.sh)"
echo "Stage 1 job submitted: $STAGE1_JOB_ID"

echo "Submitting RL Stage 2 array with dependency afterok:$STAGE1_JOB_ID..."
STAGE2_JOB_ID="$(sbatch --parsable --dependency=afterok:"$STAGE1_JOB_ID" hpc_scripts/run_RL_stage2_array.sh)"
echo "Stage 2 array job submitted: $STAGE2_JOB_ID"

echo "=============================================="
echo "RL pipeline submitted."
echo "Stage 1: $STAGE1_JOB_ID"
echo "Stage 2: $STAGE2_JOB_ID waits for successful Stage 1 completion."
echo "Monitor with:"
echo "squeue -j $STAGE1_JOB_ID,$STAGE2_JOB_ID"
echo "=============================================="
