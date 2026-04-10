"""Evaluation harness for sparse parity challenge.

Runs solve() from solve.py against 3 random seeds, measures accuracy and DMC
via the sparse-parity-challenge TrackedArray / LRU stack tracker.

Output format (key-value, parsed by task_config):
    accuracy: 1.0
    dmc: 45000.0
    time_s: 0.003
    n_samples_used: 200

If accuracy < 95%, DMC is set to 999999999 (effectively infinite) so the
reward function naturally returns 0.
"""

import importlib.util
import sys
import time

import numpy as np

# Evaluation parameters (match the official challenge leaderboard)
N_BITS = 20
K_SPARSE = 3
N_TRAIN = 200
N_TEST = 200
SEEDS = [42, 123, 456]
ACCURACY_GATE = 0.95


def generate_data(seed: int):
    """Generate sparse parity dataset for a given seed."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(N_BITS, size=K_SPARSE, replace=False).tolist())

    x_train = rng.choice([-1.0, 1.0], size=(N_TRAIN, N_BITS))
    y_train = np.prod(x_train[:, secret], axis=1)

    x_test = rng.choice([-1.0, 1.0], size=(N_TEST, N_BITS))
    y_test = np.prod(x_test[:, secret], axis=1)

    return x_train, y_train, x_test, y_test, secret


def evaluate_once(solve_fn, seed: int, use_tracker: bool = True):
    """Run solve() on one seed. Returns (correct: bool, dmc: float, time_s: float)."""
    x_train, y_train, x_test, y_test, secret = generate_data(seed)

    dmc = 0.0
    if use_tracker:
        try:
            from sparse_parity.tracked_numpy import TrackedArray, tracking_context
            from sparse_parity.lru_tracker import LRUStackTracker

            tracker = LRUStackTracker()
            with tracking_context(tracker):
                x_t = TrackedArray(x_train, "x", tracker)
                y_t = TrackedArray(y_train, "y", tracker)

                t0 = time.perf_counter()
                result = solve_fn(x_t, y_t, N_BITS, K_SPARSE)
                elapsed = time.perf_counter() - t0

            dmc = tracker.summary()["dmd"]  # tracker calls it "dmd" internally
        except ImportError:
            # Fall back to plain numpy (no DMC tracking)
            print("WARNING: sparse_parity package not installed, DMC tracking disabled", file=sys.stderr)
            use_tracker = False

    if not use_tracker:
        t0 = time.perf_counter()
        result = solve_fn(x_train, y_train, N_BITS, K_SPARSE)
        elapsed = time.perf_counter() - t0

    result = sorted(result)
    correct = result == secret

    return correct, dmc, elapsed


def main():
    # Import solve from solve.py in the same directory
    spec = importlib.util.spec_from_file_location("solve_module", "solve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solve_fn = mod.solve

    total_correct = 0
    total_dmc = 0.0
    total_time = 0.0

    for seed in SEEDS:
        correct, dmc, elapsed = evaluate_once(solve_fn, seed)
        total_correct += int(correct)
        total_dmc += dmc
        total_time += elapsed

    accuracy = total_correct / len(SEEDS)
    avg_dmc = total_dmc / len(SEEDS)
    avg_time = total_time / len(SEEDS)

    # If accuracy is below gate, set DMC to effectively infinite
    if accuracy < ACCURACY_GATE:
        avg_dmc = 999999999.0

    # Output in key-value format for task_config parsing
    print(f"accuracy: {accuracy}")
    print(f"dmc: {avg_dmc:.1f}")
    print(f"time_s: {avg_time:.6f}")
    print(f"n_samples_used: {N_TRAIN}")

    # Exit with error if accuracy below gate (gives crash feedback)
    if accuracy < ACCURACY_GATE:
        print(f"\nFAILED: accuracy {accuracy:.0%} < {ACCURACY_GATE:.0%} gate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
