# BA-Simulation-Optimization

This repository contains the code, experiments, and result data for the bachelor thesis:

> Bayesian Optimization and Reinforcement Learning for Capacity Optimization and Dispatching in a Discrete-Event Simulation of a Testing Laboratory

The thesis investigates two complementary optimization roles in a discrete-event simulation of a testing laboratory:

- Bayesian Optimization (BO) for static optimization of station and worker capacities.
- Q-learning-based Reinforcement Learning (RL) for dynamic dispatching decisions at the Sorting station.
- Bayesian Optimization as a meta-optimization method for selected RL hyperparameters.

BO and RL are not treated as directly interchangeable algorithms for exactly the same task. BO optimizes static configurations or RL hyperparameters before a simulation run, whereas the RL agent repeatedly selects dispatching actions during the simulation.

## Research Questions and Hypotheses

The main research question is how Bayesian Optimization and Q-learning-based Reinforcement Learning can be applied and evaluated for different optimization tasks within a discrete-event simulation of a testing laboratory.

The main hypotheses of the thesis are:

- H1: Bayesian Optimization achieves lower final best objective values than Random Search under the same evaluation budget.
- H2: Changed objective-component weights lead BO to identify different optimal capacity configurations.
- H3: A Q-learning agent achieves better simulation performance through dynamic dispatching decisions than fixed dispatching rules and a random agent.
- H4: BO-based hyperparameter tuning improves the Q-learning agent compared with manually selected baseline hyperparameters.

Further content details are available in `Documents/BA.pdf` and in the LaTeX source of the thesis.

## Repository Structure

```text
.
|-- Base Simulation/
|-- BO/
|-- RL/
|-- Documents/
|-- archive/
|-- requirements.txt
`-- README.md
```

- `Base Simulation/`: original simulation model, input data, example results, and calibration.
- `BO/`: BO-specific simulation variant, BO and Random Search experiments, SLURM scripts, notebooks, results, and statistical analysis.
- `RL/`: RL-specific simulation variant, agents, standard experiments, BO-based RL tuning, holdout evaluation, SLURM scripts, notebooks, results, and statistical analysis.
- `Documents/`: stored thesis documents, especially the PDF version.
- `archive/`: old or no longer used files. This folder is kept only for traceability and is not part of the final reproduction workflow.

The former `HPC-Examples/` folder was removed because those files were only examples for the final SLURM scripts.

## Installation

The project requires Python 3.10 or newer. This is particularly important on the HPC system because the base Python versions may be 3.7 or older, while some packages in `requirements.txt` require at least Python 3.10.

Local setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important libraries include `salabim` for discrete-event simulation, `ax-platform`/`botorch` for Bayesian Optimization, and `numpy`, `pandas`, `scipy`, `matplotlib`, and `seaborn` for analysis and visualization.

## HPC Setup

Full reproduction of the RL experiments is intended to be done on the HPC system. BO multi-runs can in principle be run on local hardware, but HPC execution is also recommended because of runtime and reproducibility.

On the HPC system, the Miniconda modules must be loaded before creating or activating the environment:

```bash
module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
```

Example for creating a new Conda environment with Python 3.10:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -p /cfs/earth/scratch/<user>/ba_bo_env python=3.10 -y
conda activate /cfs/earth/scratch/<user>/ba_bo_env
python -m pip install --upgrade pip
python -m pip install -r /cfs/earth/scratch/<user>/BA/BA-Simulation-Optimization/requirements.txt
```

For personal execution, the following parts of the SLURM scripts usually need to be adapted:

- `#SBATCH --chdir=...`: set the working directory to your own repository path.
- `#SBATCH --output=...` and `#SBATCH --error=...`: set log paths to your own directories.
- `#SBATCH --mail-user=...`: set your own email address for job notifications or remove mail notifications.
- `PROJECT_ROOT`, `BO_DIR`, `RL_DIR`, `RESULTS_DIR`: adapt them to your own HPC directory structure.
- `VENV_PATH` and `PYTHON_BIN`: point them to your own Conda/venv directory.
- For job arrays: set `#SBATCH --array=...` consistently with `N_RUNS`, `N_TOP`, or the number of policies.
- If necessary: adapt partition, runtime, memory, and CPU count to your HPC environment.

## Base Simulation

`Base Simulation/lab_analysis_simulation_evalfailure.py` contains the original testing-laboratory simulation model. It represents the process with stations, resources, arrival rates, processing times, due dates, and evaluation failures.

Important files:

- `Base Simulation/base_library.py`: helper functions for the simulation.
- `Base Simulation/hourly_arrival_rates.csv`: time-dependent arrival rates.
- `Base Simulation/scenarios.xlsx` and `scenarios_transposed.xlsx`: scenario and parameter data.
- `Base Simulation/results.xlsx`: stored base results.
- `Base Simulation/rate_multiplier_calibration.py`: calibration of the `rate_multiplier` for the BO simulation variant. The script tests multipliers from 0.1 to 3.0 across several seeds and selects the value whose mean late fraction is closest to 0.5.

## Bayesian Optimization

The BO part optimizes static capacity parameters of the simulation. Each candidate configuration is evaluated with several replications. The objective function combines normalized components such as completed orders, late or incomplete orders, station capacities, worker capacity, and optionally mean time in system.

Important files:

- `BO/BO_simulation/lab_analysis_simulation_BO.py`: BO simulation variant with parameterized station and worker capacities.
- `BO/BO_simulation/base_library.py`: helper functions for the BO simulation.
- `BO/experiments/optimizers/objective_utils.py`: calculation of the weighted objective function.
- `BO/experiments/optimizers/BO.py`: main Bayesian Optimization runner using Ax.
- `BO/experiments/optimizers/random_search.py`: Random Search baseline with the same evaluation scheme.
- `BO/experiments/single_run/run_single_bo.py`: starts one independent BO run.
- `BO/experiments/single_run/run_single_random_search.py`: starts one independent Random Search run.
- `BO/experiments/multi_run/multi_run_bo.py`: starts 20 BO runs with 30 trials and 30 replications each.
- `BO/experiments/multi_run/multi_run_random_search.py`: starts 20 Random Search runs with the same budget.
- `BO/experiments/multi_run/combine_multi_run_results.py`: combines the individual run folders into shared CSV files and a `summary.csv`.
- `BO/statistical_analysis/statistical_analysis_BO.py`: statistical analysis for H1 and H2.

Typical local execution of a single BO run:

```bash
cd BO
python experiments/single_run/run_single_bo.py --run-index 0 --output-dir results/bo_results
```

Typical local execution of a single Random Search run:

```bash
cd BO
python experiments/single_run/run_single_random_search.py --run-index 0 --output-dir results/random_search_results
```

Multi-run execution:

```bash
cd BO
python experiments/multi_run/multi_run_bo.py
python experiments/multi_run/multi_run_random_search.py
python experiments/multi_run/combine_multi_run_results.py
python statistical_analysis/statistical_analysis_BO.py
```

HPC scripts:

- `BO/hpc_scripts/run_BO.sh`: BO multi-run as a SLURM array. The script is currently preconfigured for `multi_run_results_changed_weights` and uses changed objective weights.
- `BO/hpc_scripts/run_randomsearch.sh`: Random Search multi-run as a SLURM array with the same changed objective weights.

For the default weights, the BO HPC scripts need to be adapted, especially `RESULTS_DIR` and the `OBJECTIVE_WEIGHT_*` variables.

Important BO result folders:

- `BO/results/bo_results/`: single BO run.
- `BO/results/random_search_results/`: single Random Search run.
- `BO/results/multi_run_results/`: combined BO-vs.-Random-Search evaluation under default weights.
- `BO/results/multi_run_results_changed_weights/`: corresponding evaluation with changed objective weights.
- `BO/results/statistical_analysis_results/`: statistical tests and descriptive statistics for H1 and H2.

## Reinforcement Learning

The RL part uses a separate simulation variant. A tabular Q-learning agent decides at the Sorting station between four dispatching rules:

- FIFO: First In First Out.
- EDD: Earliest Due Date.
- Longest Waiting Time.
- Highest Lateness Risk.

The fixed dispatching rules and a random agent are used as baselines. The final RL experiments are computationally intensive and designed for HPC execution.

Important simulation and agent files:

- `RL/RL_simulation/lab_analysis_simulation_RL.py`: RL simulation variant with explicit queues, RL decision points, and reward calculation.
- `RL/RL_simulation/RL_agents.py`: `BaselineAgent`, `RandomAgent`, and `QLearningAgent`.
- `RL/RL_simulation/base_library_RL.py`: helper functions for the RL simulation.
- `RL/hyperparameter_tuning/rate_multiplier/rate_multiplier_calibration_RL.py`: calibration of the `rate_multiplier` for RL. The script tests multipliers from 0.1 to 3.0 across several seeds and selects the value whose mean late fraction is closest to 0.5.

Standard experiments:

- `RL/experiments/RL_experiment.py`: runs baselines, the random agent, and Q-learning training/evaluation. Defaults are 30 baseline replications, 30 random-agent replications, 1000 training episodes, and 30 evaluation replications. With `--final-run`, the 10,000-episode final run is used.
- `RL/hpc_scripts/run_RL_final_experiment.sh`: HPC script for the final 10,000-episode Q-learning run.
- `RL/hpc_scripts/run_RL_5day_experiment.sh`: HPC script for a variant with a 5-day simulation duration (`RUN_DURATION=7200` minutes).

Example for a small local test run:

```bash
cd RL
python experiments/RL_experiment.py --training-episodes 10 --baseline-replications 2 --random-replications 2 --eval-replications 2 --output-dir results/rl_experiment_results/test_run
```

This is intended only as a smoke test, not as a scientific reproduction.

## RL Hyperparameter Tuning with BO

The BO-based RL tuning is structured in two stages:

- Stage 1: screening many candidates with a limited budget.
- Stage 2: retraining and evaluation of the best Stage-1 candidates with a larger budget.
- Holdout Evaluation: independent evaluation of the final policies on new seeds.

Important files:

- `RL/hyperparameter_tuning/RL_agent/rl_tuning_common.py`: shared logic for training, evaluation, search spaces, and result format.
- `RL/hyperparameter_tuning/RL_agent/stage_1/rl_stage1_bo.py`: BO screening of RL hyperparameters.
- `RL/hyperparameter_tuning/RL_agent/stage_1/run_single_rl_bo_trial.py`: runs a single Stage-1 trial.
- `RL/hyperparameter_tuning/RL_agent/stage_2/run_stage2_candidate_from_stage1_rank.py`: starts a Stage-2 candidate based on its Stage-1 rank.
- `RL/hyperparameter_tuning/RL_agent/stage_2/run_single_rl_stage2_candidate.py`: trains and evaluates a single Stage-2 candidate.
- `RL/hyperparameter_tuning/RL_agent/stage_2/combine_rl_stage2_results.py`: combines completed Stage-2 array tasks and writes `stage2_candidates.csv` and `stage2_best_candidate.json`.
- `RL/hyperparameter_tuning/lateness_risk/BO_lateness_risk.py`: separate BO evaluation for lateness-risk parameters.

HPC scripts for RL tuning:

- `RL/hpc_scripts/run_RL_stage0_BO.sh`: short pre-run/screening run with 100 training episodes. The script uses the same Stage-1 logic but writes to a `stage0` output folder.
- `RL/hpc_scripts/run_RL_stage1_BO.sh`: Stage-1 BO with 100 trials, 1000 training episodes, and 10 evaluation replications.
- `RL/hpc_scripts/run_RL_stage2_array.sh`: Stage 2 as a SLURM array over the best Stage-1 candidates, with 10,000 training episodes and 30 evaluation replications.
- `RL/hpc_scripts/submit_RL_pipeline.sh`: starts Stage 1 and then Stage 2 with a SLURM dependency.

Typical HPC execution:

```bash
cd RL
sbatch hpc_scripts/run_RL_stage1_BO.sh
sbatch hpc_scripts/run_RL_stage2_array.sh
python hyperparameter_tuning/RL_agent/stage_2/combine_rl_stage2_results.py --stage2-dir results/rl_experiment_results/rl_tuning_results/rl_tuning_hpc/stage2 --expected-n-top 10
```

Alternatively:

```bash
cd RL
bash hpc_scripts/submit_RL_pipeline.sh
```

## Holdout Evaluation and Statistical Analysis

The holdout evaluation evaluates trained policies on independent seeds and does not retrain the agents. It is important because Stage 2 is used for model selection and therefore does not provide an unbiased final performance estimate.

Important files:

- `RL/statistical_analysis/holdout/rl_holdout_evaluation.py`: evaluates policies on holdout seeds, can run individual policies via `--policy-index` for SLURM arrays, and combines results with `--combine-only`.
- `RL/statistical_analysis/holdout/combine_holdout_results.py`: helper script for combining holdout results.
- `RL/statistical_analysis/stage_2/statistical_analysis_RL.py`: statistical tests and descriptive statistics for H3 and H4.
- `RL/hpc_scripts/run_RL_holdout_evaluation.sh`: SLURM array for the holdout evaluation. The array is currently set up for 14 policies (`0-13`).

Typical HPC execution:

```bash
cd RL
sbatch hpc_scripts/run_RL_holdout_evaluation.sh
python statistical_analysis/holdout/rl_holdout_evaluation.py --combine-only --output-dir results/rl_experiment_results/rl_holdout_results/holdout_100_seeds --overwrite
python statistical_analysis/stage_2/statistical_analysis_RL.py
```

Important RL result folders:

- `RL/results/rl_experiment_results/rl_standard_results/`: manual Q-learning and baseline experiments.
- `RL/results/rl_experiment_results/rl_tuning_results/`: BO-based RL hyperparameter tuning.
- `RL/results/rl_experiment_results/rl_holdout_results/`: independent holdout evaluation.
- `RL/results/statistical_analysis_results/`: statistical RL analyses.
- `RL/results/old_unused_deprecated_results/`: old, non-final results.

## Notebooks

The notebooks are mainly used for exploratory analysis and visualization:

- `BO/notebooks/`: BO run visualizations, multi-run analyses, and meeting overviews.
- `RL/notebooks/`: RL hyperparameter tuning, holdout evaluation, agent comparisons, visuals, and meeting overviews.

The reproducible core results should be generated through the Python and SLURM scripts. Notebooks are useful for inspection and plotting, but they are not the primary reproduction interface.

## Recommended Reproduction Path

1. Create an environment with Python 3.10 or newer and install `requirements.txt`.
2. For BO: run BO and Random Search multi-runs or use the existing results in `BO/results/`.
3. Combine BO results with `BO/experiments/multi_run/combine_multi_run_results.py`.
4. Generate BO statistics with `BO/statistical_analysis/statistical_analysis_BO.py`.
5. For RL: run standard/final experiments and RL tuning on the HPC system.
6. Combine Stage-2 candidates and run the holdout evaluation.
7. Generate RL statistics with `RL/statistical_analysis/stage_2/statistical_analysis_RL.py`.

Full reproduction of all RL experiments can take a long time depending on the HPC queue, memory, and seed configuration. Small local runs are only suitable as technical tests.

## Interpretation Notes

- The BO and RL simulation variants are related, but not identical. BO optimizes static capacities; RL optimizes dynamic dispatching decisions in a separate simulation variant.
- BO results should be interpreted relative to the chosen objective function and its weights.
- RL results depend strongly on the state space, action space, reward function, seeds, and holdout protocol.
- The final scientific conclusions should be read from the thesis and the final result folders, not from `archive/` or `old_unused_deprecated_results/`.
