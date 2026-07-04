import sys
from pathlib import Path

import numpy as np

RL_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_DIR = RL_ROOT / "RL_simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

from lab_analysis_simulation_RL import simulate_rl

rate_values = np.arange(0.1, 3.1, 0.1)
seeds = range(12345, 12355)

results = []

for rm in rate_values:
    fractions = []

    for seed in seeds:
        result = simulate_rl(
            random_seed=seed,
            animate=False,
            rate_multiplier=rm,
        )

        completed = result["n_orders_completed"]
        late = result["n_orders_late"]

        if completed > 0:
            fractions.append(late / completed)

    mean_fraction = np.mean(fractions)
    results.append((rm, mean_fraction))

best = min(results, key=lambda x: abs(x[1] - 0.5))

print("Best rate_multiplier:", best[0])
print("Mean late fraction:", best[1])

for rm, frac in results:
    print(rm, round(frac, 3))
