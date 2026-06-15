#!/bin/bash
# =============================================================================
# SLURM Job Configuration: Bayesian Optimization multi-run job array
# =============================================================================

#SBATCH --job-name=bo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=earth-1
#SBATCH --time=00:30:00
#SBATCH --constraint=rhel8
#SBATCH --array=0-19
#SBATCH --chdir=/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization/BO

# Output Configuration
#SBATCH --output=logs/bo_%A_%a.out
#SBATCH --error=logs/bo_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=freyfab2@students.zhaw.ch

set -e
set -u

# =============================================================================
# Editable Experiment Configuration
# =============================================================================

N_RUNS=20
N_TRIALS=30
N_REPLICATIONS=30

BASE_SIMULATION_SEED=12345
RUN_SEED_STEP=100000
BASE_BO_RANDOM_SEED=24680
OPTIMIZER_SEED_STEP=1000

# Leave empty to use the defaults from BO.py.
RUN_DURATION=""
RATE_MULTIPLIER=""
OBJECTIVE_TIME_MODE="mean"

# Objective weights. Defaults match objective_utils.DEFAULT_OBJECTIVE_WEIGHTS.
OBJECTIVE_WEIGHT_COMPLETED=1.0
OBJECTIVE_WEIGHT_LATE_TOTAL=3.0
OBJECTIVE_WEIGHT_STATION_CAPACITY=0.8
OBJECTIVE_WEIGHT_WORKER_CAPACITY=0.35
OBJECTIVE_WEIGHT_TIME_IN_SYSTEM_MEAN=0.4
OBJECTIVE_WEIGHT_TIME_IN_SYSTEM_STD=0.2

# =============================================================================
# Paths and Environment
# =============================================================================

PROJECT_ROOT="/cfs/earth/scratch/freyfab2/BA/BA-Simulation-Optimization"
BO_DIR="$PROJECT_ROOT/BO"
RESULTS_DIR="$BO_DIR/multi_run_results"
LOG_DIR="$BO_DIR/logs"
VENV_PATH="/cfs/earth/scratch/freyfab2/ba_bo_env"

mkdir -p "$LOG_DIR" "$RESULTS_DIR/bo"

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

if [ ! -d "$BO_DIR" ]; then
    echo "ERROR: BO directory not found at $BO_DIR"
    exit 1
fi

cd "$BO_DIR"

RUN_INDEX="${SLURM_ARRAY_TASK_ID}"

if [ "$RUN_INDEX" -ge "$N_RUNS" ]; then
    echo "Skipping array task $RUN_INDEX because N_RUNS=$N_RUNS."
    exit 0
fi

RUN_LABEL=$(printf "run_%02d" "$RUN_INDEX")
OUTPUT_DIR="$RESULTS_DIR/bo/$RUN_LABEL"

BASE_SEED=$((BASE_SIMULATION_SEED + RUN_INDEX * RUN_SEED_STEP))
BO_RANDOM_SEED=$((BASE_BO_RANDOM_SEED + RUN_INDEX * OPTIMIZER_SEED_STEP))

if [ -f "$OUTPUT_DIR/bo_best_parameters.json" ] \
    && [ -f "$OUTPUT_DIR/bo_trials.csv" ] \
    && [ -f "$OUTPUT_DIR/bo_replications.csv" ]; then
    echo "Skipping completed BO run $RUN_INDEX at $OUTPUT_DIR."
    exit 0
fi

export RUN_INDEX
export N_TRIALS
export N_REPLICATIONS
export BASE_SEED
export BO_RANDOM_SEED
export OUTPUT_DIR
export RUN_DURATION
export RATE_MULTIPLIER
export OBJECTIVE_TIME_MODE
export OBJECTIVE_WEIGHT_COMPLETED
export OBJECTIVE_WEIGHT_LATE_TOTAL
export OBJECTIVE_WEIGHT_STATION_CAPACITY
export OBJECTIVE_WEIGHT_WORKER_CAPACITY
export OBJECTIVE_WEIGHT_TIME_IN_SYSTEM_MEAN
export OBJECTIVE_WEIGHT_TIME_IN_SYSTEM_STD

echo "=============================================="
echo "Bayesian Optimization Run"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: $RUN_INDEX"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start Time: $(date)"
echo "Python: $(which python)"
python --version
echo "Output: $OUTPUT_DIR"
echo "=============================================="

python - <<'PY'
import json
import os

import BO as bo


def optional_float(name: str) -> float | None:
    value = os.environ[name].strip()
    return None if value == "" else float(value)


run_index = int(os.environ["RUN_INDEX"])
output_dir = os.environ["OUTPUT_DIR"]

bo.OBJECTIVE_WEIGHTS.update(
    {
        "completed": float(os.environ["OBJECTIVE_WEIGHT_COMPLETED"]),
        "late_total": float(os.environ["OBJECTIVE_WEIGHT_LATE_TOTAL"]),
        "station_capacity": float(os.environ["OBJECTIVE_WEIGHT_STATION_CAPACITY"]),
        "worker_capacity": float(os.environ["OBJECTIVE_WEIGHT_WORKER_CAPACITY"]),
        "time_in_system_mean": float(
            os.environ["OBJECTIVE_WEIGHT_TIME_IN_SYSTEM_MEAN"]
        ),
        "time_in_system_std": float(os.environ["OBJECTIVE_WEIGHT_TIME_IN_SYSTEM_STD"]),
    }
)

result = bo.run_experiment(
    n_trials=int(os.environ["N_TRIALS"]),
    n_replications=int(os.environ["N_REPLICATIONS"]),
    base_seed=int(os.environ["BASE_SEED"]),
    seed_step=bo.SEED_STEP,
    bo_random_seed=int(os.environ["BO_RANDOM_SEED"]),
    output_dir=output_dir,
    run_index=run_index,
    run_duration=optional_float("RUN_DURATION"),
    rate_multiplier=optional_float("RATE_MULTIPLIER"),
    objective_time_mode=os.environ["OBJECTIVE_TIME_MODE"],
    save_after_each_trial=False,
    return_records=False,
)

print(json.dumps(bo.json_safe(result["best_result"]), indent=2))
PY

echo "=============================================="
echo "BO run completed: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Combine later with: python combine_multi_run_results.py"
echo "=============================================="
