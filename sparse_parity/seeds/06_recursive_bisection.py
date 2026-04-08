"""Seed 6: Build-up search — find secret bits one at a time.

Find the first secret bit by checking which single bit, when fixed to +1,
biases y toward the product of the remaining secret bits. Uses data
splitting to isolate one bit at a time.
"""

import numpy as np


def solve(x, y, n_bits, k_sparse):
    n_samples = x.shape[0]

    # Strategy: for each bit i, split data by x_i = +1 vs x_i = -1.
    # If i is secret, the conditional distributions of y differ.
    # Specifically, in the x_i=+1 subset, y = product of OTHER secret bits.
    # In the x_i=-1 subset, y = -product of OTHER secret bits.
    # So E[y | x_i=+1] = 0 and E[y | x_i=-1] = 0 (remaining bits are random).
    # But E[y * x_i] = 0 too (as we know).

    # Better: for a candidate triple (i,j,k), prod(x_{i,j,k}) should equal y.
    # We can verify by checking if ALL samples agree.

    # Use subset: check exact agreement, not just correlation
    n_use = min(n_samples, 50)
    x_sub = x[:n_use]
    y_sub = y[:n_use]

    # Build up one bit at a time using conditional independence
    # Step 1: Find a pair (i,j) where x_i * x_j is NOT independent of y
    # Check via: does x_i * x_j predict anything about y across subsets?

    # Actually for k=3, brute force over all triples but check exact match
    # (not just mean but exact agreement on all samples)
    if k_sparse == 3:
        for i in range(n_bits):
            col_i = x_sub[:, i]
            for j in range(i + 1, n_bits):
                col_ij = col_i * x_sub[:, j]
                for k in range(j + 1, n_bits):
                    prod = col_ij * x_sub[:, k]
                    # Check exact agreement (stronger than correlation)
                    if np.all(prod == y_sub):
                        return [i, j, k]

    # Fallback: correlation-based
    best_subset = None
    best_score = 0.0
    from itertools import combinations
    for combo in combinations(range(n_bits), k_sparse):
        prod = np.ones(n_use)
        for idx in combo:
            prod = prod * x_sub[:, idx]
        score = abs(np.mean(prod * y_sub))
        if score > best_score:
            best_score = score
            best_subset = list(combo)

    return sorted(best_subset)
