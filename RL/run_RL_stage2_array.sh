#!/bin/bash
# =============================================================================
# SLURM Job Configuration: staged RL tuning Stage 2 candidate array
# =============================================================================

#SBATCH --job-name=rl_stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=earth-3
#SBATCH --time=02:00:00
#SBATCH --constraint=rhel8
#SBATCH --array=0-3
#SBATCH --chdir=/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization/RL

# Output Configuration
#SBATCH --output=logs/rl_stage2_%A_%a.out
#SBATCH --error=logs/rl_stage2_%A_%a.err
# Uncomment and adapt if desired.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=freyfab2@students.zhaw.ch

set -e
set -u

# =============================================================================
# Editable Experiment Configuration
# =============================================================================

# For N_TOP=5, also change the SBATCH array line above to: #SBATCH --array=0-4
N_TOP=4
TRAINING_EPISODES=1000
EVAL_REPLICATIONS=10

BASE_SEED=12345
SEED_STEP=1000

# Leave empty to use the defaults from RL_experiment.py.
RUN_DURATION=""
RATE_MULTIPLIER=""

# If true, skip a candidate when candidate_summary.json, evaluation.csv and
# q_table.pkl already exist in that candidate directory.
SKIP_COMPLETED=true

# =============================================================================
# Paths and Environment
# =============================================================================

PROJECT_ROOT="/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization"
RL_DIR="$PROJECT_ROOT/RL"
RESULTS_DIR="$RL_DIR/rl_tuning_hpc"
LOG_DIR="$RL_DIR/logs"
VENV_PATH="/cfs/earth/scratch/freyfab2/ba_bo_env"
PYTHON_BIN="$VENV_PATH/bin/python"

STAGE1_DIR="$RESULTS_DIR/stage1"
STAGE2_DIR="$RESULTS_DIR/stage2"

mkdir -p "$LOG_DIR" "$STAGE2_DIR"

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

if [ ! -f "$STAGE1_DIR/stage1_trials.csv" ]; then
    echo "ERROR: Stage-1 trials file not found at $STAGE1_DIR/stage1_trials.csv"
    echo "Run Stage 1 first: sbatch run_RL_stage1_BO.sh"
    exit 1
fi

cd "$RL_DIR"

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "$ARRAY_TASK_ID" -ge "$N_TOP" ]; then
    echo "Skipping array task $ARRAY_TASK_ID because N_TOP=$N_TOP."
    exit 0
fi

CANDIDATE_RANK=$((ARRAY_TASK_ID + 1))

echo "=============================================="
echo "RL Stage-2 Candidate Array Task"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: $ARRAY_TASK_ID"
echo "Candidate Rank: $CANDIDATE_RANK"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "Python bin: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Stage-1 input: $STAGE1_DIR"
echo "Stage-2 output: $STAGE2_DIR"
echo "N_TOP: $N_TOP"
echo "TRAINING_EPISODES: $TRAINING_EPISODES"
echo "EVAL_REPLICATIONS: $EVAL_REPLICATIONS"
echo "=============================================="

"$PYTHON_BIN" - <<'PY'
import ax
import numpy
import pandas
import salabim

import rl_tuning_common
import run_stage2_candidate_from_stage1_rank

print("Import check passed: ax, numpy, pandas, salabim, rl_tuning_common, run_stage2_candidate_from_stage1_rank")
PY

COMMAND=(
    "$PYTHON_BIN"
    "run_stage2_candidate_from_stage1_rank.py"
    "--stage1-dir" "$STAGE1_DIR"
    "--output-dir" "$STAGE2_DIR"
    "--candidate-rank" "$CANDIDATE_RANK"
    "--n-top" "$N_TOP"
    "--training-episodes" "$TRAINING_EPISODES"
    "--eval-replications" "$EVAL_REPLICATIONS"
    "--base-seed" "$BASE_SEED"
    "--seed-step" "$SEED_STEP"
)

if [ "$SKIP_COMPLETED" = true ]; then
    COMMAND+=("--skip-completed")
fi

if [ -n "$RUN_DURATION" ]; then
    COMMAND+=("--run-duration" "$RUN_DURATION")
fi

if [ -n "$RATE_MULTIPLIER" ]; then
    COMMAND+=("--rate-multiplier" "$RATE_MULTIPLIER")
fi

"${COMMAND[@]}"

echo "=============================================="
echo "RL Stage-2 candidate completed: $(date)"
echo "Candidate rank: $CANDIDATE_RANK"
echo "Combine after all array tasks finish with:"
echo "python combine_rl_stage2_results.py --stage2-dir $STAGE2_DIR --expected-n-top $N_TOP"
echo "=============================================="
