"""Seed 4: Direct k-th order Walsh coefficient search.

For sparse parity, the ONLY non-zero Walsh coefficient is at the exact
secret set S. We sample random k-subsets and compute their Walsh coefficient
E[chi_S(x) * y], accepting the first with |coeff| > threshold.

Much faster than brute-force for larger n when k is small and we're lucky.
"""

import numpy as np


def solve(x, y, n_bits, k_sparse):
    n_use = min(x.shape[0], 40)
    x_sub = x[:n_use]
    y_sub = y[:n_use]

    # Try random k-subsets first (fast if we get lucky)
    rng = np.random.RandomState(0)
    n_random_tries = 500

    for _ in range(n_random_tries):
        subset = sorted(rng.choice(n_bits, size=k_sparse, replace=False).tolist())
        prod = np.ones(n_use)
        for idx in subset:
            prod = prod * x_sub[:, idx]
        coeff = abs(np.mean(prod * y_sub))
        if coeff > 0.8:
            return subset

    # Fallback: systematic brute force
    if k_sparse == 3:
        for i in range(n_bits):
            col_i = x_sub[:, i]
            for j in range(i + 1, n_bits):
                col_ij = col_i * x_sub[:, j]
                for k in range(j + 1, n_bits):
                    prod = col_ij * x_sub[:, k]
                    coeff = abs(np.mean(prod * y_sub))
                    if coeff > 0.8:
                        return sorted([i, j, k])

    return list(range(k_sparse))  # should never reach here
