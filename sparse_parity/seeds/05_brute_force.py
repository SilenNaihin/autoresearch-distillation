"""Seed 5: Brute-force subset scan — check all C(n,k) combinations.

For n=20, k=3, there are C(20,3)=1140 combinations. With small sample
count, this is feasible. Optimize for DMC by accessing data sequentially.
"""

import numpy as np


def solve(x, y, n_bits, k_sparse):
    # Use minimal samples for brute force
    n_use = min(x.shape[0], 30)
    x_sub = x[:n_use]
    y_sub = y[:n_use]

    best_subset = None
    best_score = -1.0

    # Generate all k-combinations manually (avoid itertools for DMC)
    # For k=3, triple nested loop
    if k_sparse == 3:
        for i in range(n_bits):
            for j in range(i + 1, n_bits):
                for k in range(j + 1, n_bits):
                    # Compute product of these three bits
                    prod = x_sub[:, i] * x_sub[:, j] * x_sub[:, k]
                    # Check agreement with y
                    score = abs(np.mean(prod * y_sub))
                    if score > best_score:
                        best_score = score
                        best_subset = [i, j, k]
    else:
        # General case: use itertools
        from itertools import combinations
        for combo in combinations(range(n_bits), k_sparse):
            prod = np.ones(n_use)
            for idx in combo:
                prod = prod * x_sub[:, idx]
            score = abs(np.mean(prod * y_sub))
            if score > best_score:
                best_score = score
                best_subset = list(combo)

    return sorted(best_subset) if best_subset else list(range(k_sparse))
