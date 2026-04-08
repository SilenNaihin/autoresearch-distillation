"""Seed 1: Brute-force triple scan with minimal samples.

Check all C(n,k) combinations directly. For n=20, k=3 this is 1140 triples.
Uses a small sample subset and sequential access for lower DMC.
"""

import numpy as np


def solve(x, y, n_bits, k_sparse):
    # Use minimal samples — 30 is enough for statistical confidence
    n_use = min(x.shape[0], 30)
    x_sub = x[:n_use]
    y_sub = y[:n_use]

    best_subset = None
    best_score = 0.0

    # Check all k-subsets
    if k_sparse == 3:
        for i in range(n_bits):
            col_i = x_sub[:, i]
            for j in range(i + 1, n_bits):
                col_ij = col_i * x_sub[:, j]
                for k in range(j + 1, n_bits):
                    prod = col_ij * x_sub[:, k]
                    score = abs(np.mean(prod * y_sub))
                    if score > best_score:
                        best_score = score
                        best_subset = [i, j, k]
                        if score > 0.9:
                            return sorted(best_subset)
    else:
        from itertools import combinations
        for combo in combinations(range(n_bits), k_sparse):
            prod = np.ones(n_use)
            for idx in combo:
                prod = prod * x_sub[:, idx]
            score = abs(np.mean(prod * y_sub))
            if score > best_score:
                best_score = score
                best_subset = list(combo)
                if score > 0.9:
                    return sorted(best_subset)

    return sorted(best_subset) if best_subset else list(range(k_sparse))
