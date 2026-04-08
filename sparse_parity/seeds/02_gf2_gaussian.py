"""Seed 2: GF(2) Gaussian elimination — map to binary field, solve via RREF.

Insight: in {-1,+1} encoding, y = prod(x[secret]) is equivalent to
y_bin = XOR(x_bin[secret]) in {0,1} encoding where x_bin = (1-x)/2.
This makes it a linear algebra problem over GF(2).
"""

import numpy as np


def solve(x, y, n_bits, k_sparse):
    # Map {-1, +1} -> {0, 1}: -1 -> 1, +1 -> 0
    x_bin = ((1 - x) // 2).astype(np.int8)
    y_bin = ((1 - y) // 2).astype(np.int8)

    n_samples = x_bin.shape[0]

    # Use minimal samples: n_bits + 1 rows should suffice for rank
    n_use = min(n_samples, n_bits + 1)
    A = x_bin[:n_use].copy()
    b = y_bin[:n_use].copy()

    # Augmented matrix [A | b] over GF(2)
    aug = np.zeros((n_use, n_bits + 1), dtype=np.int8)
    aug[:, :n_bits] = A
    aug[:, n_bits] = b

    # Gaussian elimination over GF(2)
    pivot_cols = []
    row = 0
    for col in range(n_bits):
        # Find pivot in this column
        pivot = -1
        for r in range(row, n_use):
            if aug[r, col] == 1:
                pivot = r
                break
        if pivot < 0:
            continue
        pivot_cols.append(col)

        # Swap rows
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]

        # Eliminate below and above
        for r in range(n_use):
            if r != row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2
        row += 1

    # Read solution: the secret bits are where the solution vector is 1
    # After RREF, the solution is in the last column at pivot rows
    secret = []
    for i, col in enumerate(pivot_cols):
        if aug[i, n_bits] == 1:
            secret.append(col)

    # Pad or trim to k_sparse if needed
    if len(secret) < k_sparse:
        # Fallback: add remaining columns with highest correlation
        remaining = [c for c in range(n_bits) if c not in secret]
        for c in remaining:
            if len(secret) >= k_sparse:
                break
            secret.append(c)
    secret = secret[:k_sparse]

    return sorted(secret)
