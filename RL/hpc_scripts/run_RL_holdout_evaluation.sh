#!/bin/bash
# =============================================================================
# SLURM Job Configuration: independent RL holdout policy evaluation
# =============================================================================

#SBATCH --job-name=rl_holdout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=earth-1
#SBATCH --time=02:00:00
#SBATCH --constraint=rhel8
#SBATCH --array=0-13
#SBATCH --chdir=/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization/RL

# Output Configuration
#SBATCH --output=logs/rl_holdout_%A_%a.out
#SBATCH --error=logs/rl_holdout_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=freyfab2@students.zhaw.ch

set -e
set -u

# =============================================================================
# Editable Experiment Configuration
# =============================================================================

N_REPLICATIONS=100
EVAL_SEED_BASE=900000
SEED_STEP=1000
ACTION_SEED_BASE=1900000
OUTPUT_DIR="results/rl_experiment_results/rl_holdout_results/holdout_100_seeds"
OVERWRITE=false

# =============================================================================
# Paths and Environment
# =============================================================================

PROJECT_ROOT="/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization"
RL_DIR="$PROJECT_ROOT/RL"
LOG_DIR="$RL_DIR/logs"
VENV_PATH="/cfs/earth/scratch/freyfab2/ba_bo_env"
PYTHON_BIN="$VENV_PATH/bin/python"

mkdir -p "$LOG_DIR"

module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found after loading miniconda3/4.12.0."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: conda environment not found at $VENV_PATH"
    exit 1
fi

conda activate "$VENV_PATH"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: python executable not found at $PYTHON_BIN"
    exit 1
fi

if [ ! -d "$RL_DIR" ]; then
    echo "ERROR: RL directory not found at $RL_DIR"
    exit 1
fi

cd "$RL_DIR"
export PYTHONPATH="$RL_DIR/RL_simulation:$RL_DIR/experiments:$RL_DIR/statistical_analysis/holdout:${PYTHONPATH:-}"

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "=============================================="
echo "RL Holdout Evaluation"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: $ARRAY_TASK_ID"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "Python bin: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Output directory: $OUTPUT_DIR"
echo "N_REPLICATIONS: $N_REPLICATIONS"
echo "EVAL_SEED_BASE: $EVAL_SEED_BASE"
echo "SEED_STEP: $SEED_STEP"
echo "ACTION_SEED_BASE: $ACTION_SEED_BASE"
echo "=============================================="

"$PYTHON_BIN" - <<'PY'
import numpy
import pandas
import salabim

import rl_holdout_evaluation

print("Import check passed: numpy, pandas, salabim, rl_holdout_evaluation")
PY

COMMAND=(
    "$PYTHON_BIN"
    "statistical_analysis/holdout/rl_holdout_evaluation.py"
    "--policy-index" "$ARRAY_TASK_ID"
    "--n-replications" "$N_REPLICATIONS"
    "--eval-seed-base" "$EVAL_SEED_BASE"
    "--seed-step" "$SEED_STEP"
    "--action-seed-base" "$ACTION_SEED_BASE"
    "--output-dir" "$OUTPUT_DIR"
)

if [ "$OVERWRITE" = true ]; then
    COMMAND+=("--overwrite")
fi

"${COMMAND[@]}"

echo "=============================================="
echo "RL holdout policy task completed: $(date)"
echo "Array task: $ARRAY_TASK_ID"
echo "Combine after all array tasks finish with:"
echo "python statistical_analysis/holdout/rl_holdout_evaluation.py --combine-only --output-dir $OUTPUT_DIR --overwrite"
echo "=============================================="
