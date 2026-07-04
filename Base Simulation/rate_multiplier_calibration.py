import numpy as np
from lab_analysis_simulation_evalfailure import simulate

rate_values = np.arange(0.1, 3.1, 0.1)
seeds = range(12345, 12355)

results = []

for rm in rate_values:
    fractions = []

    for seed in seeds:
        result = simulate(
            random_seed=seed,
            animate=False,
            rate_multiplier=rm,
        )

        completed = result["n_orders_completed"]
        late = result["out_of_date_count"]

        if completed > 0:
            fractions.append(late / completed)

    mean_fraction = np.mean(fractions)
    results.append((rm, mean_fraction))

best = min(results, key=lambda x: abs(x[1] - 0.5))

print("Best rate_multiplier:", best[0])
print("Mean late fraction:", best[1])

for rm, frac in results:
    print(rm, round(frac, 3))