#!/bin/bash
# =============================================================================
# SLURM Job Configuration: staged RL tuning Stage 0 Bayesian Optimization
# =============================================================================

#SBATCH --job-name=rl_stage0_bo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=earth-3
#SBATCH --time=04:00:00
#SBATCH --constraint=rhel8
#SBATCH --chdir=/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization/RL

# Output Configuration
#SBATCH --output=logs/rl_stage0_bo_%j.out
#SBATCH --error=logs/rl_stage0_bo_%j.err
# Uncomment and adapt if desired.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=freyfab2@students.zhaw.ch

set -e
set -u

# =============================================================================
# Editable Experiment Configuration
# =============================================================================

N_TRIALS=100
TRAINING_EPISODES=100
EVAL_REPLICATIONS=10

BASE_SEED=12345
SEED_STEP=1000
BO_RANDOM_SEED=24680

# Leave empty to use the defaults from RL_experiment.py.
RUN_DURATION=""
RATE_MULTIPLIER=""

# If true, skip when Stage-0 final outputs already exist.
SKIP_COMPLETED=true

# =============================================================================
# Paths and Environment
# =============================================================================

PROJECT_ROOT="/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization"
RL_DIR="$PROJECT_ROOT/RL"
RESULTS_DIR="$RL_DIR/results/rl_experiment_results/rl_tuning_results/rl_tuning_hpc"
LOG_DIR="$RL_DIR/logs"
VENV_PATH="/cfs/earth/scratch/freyfab2/ba_bo_env"
PYTHON_BIN="$VENV_PATH/bin/python"

STAGE0_DIR="$RESULTS_DIR/stage0"

mkdir -p "$LOG_DIR" "$STAGE0_DIR"

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
    echo "Create the environment first or update VENV_PATH in this script."
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
export PYTHONPATH="$RL_DIR/RL_simulation:$RL_DIR/experiments:$RL_DIR/hyperparameter_tuning/RL_agent:$RL_DIR/hyperparameter_tuning/RL_agent/stage_1:$RL_DIR/hyperparameter_tuning/RL_agent/stage_2:${PYTHONPATH:-}"

if [ "$SKIP_COMPLETED" = true ] \
    && [ -f "$STAGE0_DIR/stage1_best_parameters.json" ] \
    && [ -f "$STAGE0_DIR/stage1_trials.csv" ]; then
    echo "Skipping completed RL Stage-0 BO run at $STAGE0_DIR."
    exit 0
fi

echo "=============================================="
echo "RL Stage-0 Bayesian Optimization"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "Python bin: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Output: $STAGE0_DIR"
echo "N_TRIALS: $N_TRIALS"
echo "TRAINING_EPISODES: $TRAINING_EPISODES"
echo "EVAL_REPLICATIONS: $EVAL_REPLICATIONS"
echo "=============================================="

"$PYTHON_BIN" - <<'PY'
import ax
import numpy
import pandas
import salabim

import rl_stage1_bo
import rl_tuning_common

print("Import check passed: ax, numpy, pandas, salabim, rl_stage1_bo, rl_tuning_common")
PY

COMMAND=(
    "$PYTHON_BIN"
    "hyperparameter_tuning/RL_agent/stage_1/rl_stage1_bo.py"
    "--output-dir" "$STAGE0_DIR"
    "--n-trials" "$N_TRIALS"
    "--training-episodes" "$TRAINING_EPISODES"
    "--eval-replications" "$EVAL_REPLICATIONS"
    "--base-seed" "$BASE_SEED"
    "--seed-step" "$SEED_STEP"
    "--bo-random-seed" "$BO_RANDOM_SEED"
)

if [ -n "$RUN_DURATION" ]; then
    COMMAND+=("--run-duration" "$RUN_DURATION")
fi

if [ -n "$RATE_MULTIPLIER" ]; then
    COMMAND+=("--rate-multiplier" "$RATE_MULTIPLIER")
fi

"${COMMAND[@]}"

echo "=============================================="
echo "RL Stage-0 BO completed: $(date)"
echo "Output directory: $STAGE0_DIR"
echo "Run Stage 2 with: sbatch hpc_scripts/run_RL_stage2_array.sh"
echo "=============================================="
