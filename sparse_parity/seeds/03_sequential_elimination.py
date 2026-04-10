"""Seed 3: Sequential elimination via GF(2) rank testing.

For each candidate bit, test if removing it from the GF(2) system still
allows a consistent solution. Non-secret bits can be removed without
affecting consistency; secret bits cannot.
"""

import numpy as np


def solve(x, y, n_bits, k_sparse):
    # Map {-1, +1} -> {0, 1}
    x_bin = ((1 - x) // 2).astype(np.int8)
    y_bin = ((1 - y) // 2).astype(np.int8)

    n_use = min(x_bin.shape[0], n_bits + 5)
    A = x_bin[:n_use]
    b = y_bin[:n_use]

    candidates = list(range(n_bits))

    while len(candidates) > k_sparse:
        # For each candidate, check if the system is still consistent without it
        worst_bit = None
        best_residual = float("inf")

        for c in candidates:
            others = [j for j in candidates if j != c]
            # Build sub-matrix with only 'others' columns
            A_sub = A[:, others]
            # Try to solve A_sub @ x = b over GF(2) using Gaussian elimination
            # If the system is consistent, bit c is not needed
            aug = np.zeros((n_use, len(others) + 1), dtype=np.int8)
            aug[:, :len(others)] = A_sub
            aug[:, len(others)] = b

            # Gaussian elimination
            row = 0
            for col in range(len(others)):
                pivot = -1
                for r in range(row, n_use):
                    if aug[r, col] == 1:
                        pivot = r
                        break
                if pivot < 0:
                    continue
                if pivot != row:
                    aug[[row, pivot]] = aug[[pivot, row]]
                for r in range(n_use):
                    if r != row and aug[r, col] == 1:
                        aug[r] = (aug[r] + aug[row]) % 2
                row += 1

            # Check consistency: any row with all-zero left side but non-zero right side?
            inconsistent = False
            for r in range(row, n_use):
                if aug[r, len(others)] == 1:
                    inconsistent = True
                    break

            if not inconsistent:
                # System is consistent without bit c — c is likely not secret
                # Use number of non-zero solution entries as tiebreaker
                n_ones = sum(aug[i, len(others)] for i in range(row))
                if n_ones < best_residual:
                    best_residual = n_ones
                    worst_bit = c

        if worst_bit is not None:
            candidates.remove(worst_bit)
        else:
            # All bits seem necessary — remove the one with lowest pair interaction
            candidates.pop()

    return sorted(candidates)
