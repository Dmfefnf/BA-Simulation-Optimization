#!/bin/bash
# =============================================================================
# SLURM Job Configuration: RL final-run experiment
# =============================================================================

#SBATCH --job-name=rl_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=earth-3
#SBATCH --time=06:00:00
#SBATCH --constraint=rhel8
#SBATCH --exclusive
#SBATCH --chdir=/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization/RL

# Output Configuration
#SBATCH --output=logs/rl_final_%j.out
#SBATCH --error=logs/rl_final_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=freyfab2@students.zhaw.ch

set -e
set -u

# =============================================================================
# Editable Experiment Configuration
# =============================================================================

# Keep empty to use RL_experiment.py final-run defaults.
# Defaults in Python:
# - final training episodes: 10000
# - baseline replications: 30
# - random-agent replications: 30
# - evaluation replications: 30
TRAINING_EPISODES=""
BASELINE_REPLICATIONS=""
RANDOM_REPLICATIONS=""
EVAL_REPLICATIONS=""

BASE_SEED=12345
SEED_STEP=1000

# Leave empty to use the defaults from RL_experiment.py.
RUN_DURATION=""
RATE_MULTIPLIER=""

# Optional switches. Set to true to skip these comparison runs.
SKIP_BASELINES=false
SKIP_RANDOM=false

# If true, skip when the final summary and q-table outputs already exist.
SKIP_COMPLETED=true

# =============================================================================
# Paths and Environment
# =============================================================================

PROJECT_ROOT="/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization"
RL_DIR="$PROJECT_ROOT/RL"
RESULTS_DIR="$RL_DIR/rl_results_final_10000_hpc"
LOG_DIR="$RL_DIR/logs"
VENV_PATH="/cfs/earth/scratch/freyfab2/ba_bo_env"
PYTHON_BIN="$VENV_PATH/bin/python"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

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

if [ "$SKIP_COMPLETED" = true ] \
    && [ -f "$RESULTS_DIR/rl_summary.json" ] \
    && [ -f "$RESULTS_DIR/q_table_final.pkl" ]; then
    echo "Skipping completed RL final run at $RESULTS_DIR."
    exit 0
fi

echo "=============================================="
echo "RL Final-Run Experiment"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "Python bin: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Output: $RESULTS_DIR"
echo "TRAINING_EPISODES: ${TRAINING_EPISODES:-RL_experiment.py final default}"
echo "BASELINE_REPLICATIONS: ${BASELINE_REPLICATIONS:-RL_experiment.py default}"
echo "RANDOM_REPLICATIONS: ${RANDOM_REPLICATIONS:-RL_experiment.py default}"
echo "EVAL_REPLICATIONS: ${EVAL_REPLICATIONS:-RL_experiment.py default}"
echo "SKIP_BASELINES: $SKIP_BASELINES"
echo "SKIP_RANDOM: $SKIP_RANDOM"
echo "=============================================="

"$PYTHON_BIN" - <<'PY'
import numpy
import pandas
import salabim

import RL_experiment
import lab_analysis_simulation_RL

print("Import check passed: numpy, pandas, salabim, RL_experiment, lab_analysis_simulation_RL")
PY

COMMAND=(
    "$PYTHON_BIN"
    "RL_experiment.py"
    "--final-run"
    "--output-dir" "$RESULTS_DIR"
    "--base-seed" "$BASE_SEED"
    "--seed-step" "$SEED_STEP"
)

if [ -n "$TRAINING_EPISODES" ]; then
    COMMAND+=("--training-episodes" "$TRAINING_EPISODES")
fi

if [ -n "$BASELINE_REPLICATIONS" ]; then
    COMMAND+=("--baseline-replications" "$BASELINE_REPLICATIONS")
fi

if [ -n "$RANDOM_REPLICATIONS" ]; then
    COMMAND+=("--random-replications" "$RANDOM_REPLICATIONS")
fi

if [ -n "$EVAL_REPLICATIONS" ]; then
    COMMAND+=("--eval-replications" "$EVAL_REPLICATIONS")
fi

if [ "$SKIP_BASELINES" = true ]; then
    COMMAND+=("--skip-baselines")
fi

if [ "$SKIP_RANDOM" = true ]; then
    COMMAND+=("--skip-random")
fi

if [ -n "$RUN_DURATION" ]; then
    COMMAND+=("--run-duration" "$RUN_DURATION")
fi

if [ -n "$RATE_MULTIPLIER" ]; then
    COMMAND+=("--rate-multiplier" "$RATE_MULTIPLIER")
fi

"${COMMAND[@]}"

echo "=============================================="
echo "RL final run completed: $(date)"
echo "Output directory: $RESULTS_DIR"
echo "=============================================="
