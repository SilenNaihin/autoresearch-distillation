"""Seed 7: Random restart hill climbing on k-subsets.

Start with a random k-subset, compute agreement score with y. Then try
swapping one bit at a time, accepting improvements. Restart from random
if stuck. Optimizes for low sample usage.
"""

import numpy as np


def solve(x, y, n_bits, k_sparse):
    n_use = min(x.shape[0], 30)
    x_sub = x[:n_use]
    y_sub = y[:n_use]

    rng = np.random.RandomState(7)

    def score_subset(subset):
        prod = np.ones(n_use)
        for idx in subset:
            prod = prod * x_sub[:, idx]
        return np.mean(prod == y_sub)  # fraction of exact matches

    best_global = None
    best_global_score = 0.0

    for restart in range(50):
        # Random initial subset
        current = sorted(rng.choice(n_bits, size=k_sparse, replace=False).tolist())
        current_score = score_subset(current)

        if current_score == 1.0:
            return current

        # Hill climb: swap one bit at a time
        improved = True
        while improved:
            improved = False
            for pos in range(k_sparse):
                for candidate in range(n_bits):
                    if candidate in current:
                        continue
                    trial = current.copy()
                    trial[pos] = candidate
                    trial = sorted(trial)
                    trial_score = score_subset(trial)
                    if trial_score > current_score:
                        current = trial
                        current_score = trial_score
                        improved = True
                        if current_score == 1.0:
                            return current
                        break  # restart inner loop after improvement
                if improved:
                    break

        if current_score > best_global_score:
            best_global_score = current_score
            best_global = current

    return best_global if best_global else list(range(k_sparse))
