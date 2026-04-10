"""Pre-evaluate all seed solutions and populate the reuse buffer.

Run this once before training to give PUCT diverse starting points.

Usage:
    python seed_buffer.py [--buffer-path /data/reuse_buffer_sparse_parity.json]
                          [--isolated-dir /data/isolated_buffers/]
"""

import argparse
import importlib.util
import json
import sys
import os
from pathlib import Path

# Add parent dir to path for lib imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluate_once, N_BITS, K_SPARSE, N_TRAIN, SEEDS, ACCURACY_GATE


SEED_DIR = Path(__file__).parent / "seeds"

SEED_FILES = [
    ("01_correlation.py", "correlation"),
    ("02_gf2_gaussian.py", "gf2_gaussian"),
    ("03_sequential_elimination.py", "sequential_elimination"),
    ("04_walsh_hadamard.py", "walsh_hadamard"),
    ("05_brute_force.py", "brute_force"),
    ("06_recursive_bisection.py", "recursive_bisection"),
    ("07_coordinate_descent.py", "coordinate_descent"),
]


def load_solve_fn(path: Path):
    """Load solve() from a seed file."""
    spec = importlib.util.spec_from_file_location("seed_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve


def evaluate_seed(path: Path, use_tracker: bool = True):
    """Evaluate a seed solution across all seeds. Returns (accuracy, avg_dmc, avg_time)."""
    solve_fn = load_solve_fn(path)

    total_correct = 0
    total_dmc = 0.0
    total_time = 0.0

    for seed in SEEDS:
        correct, dmc, elapsed = evaluate_once(solve_fn, seed, use_tracker=use_tracker)
        total_correct += int(correct)
        total_dmc += dmc
        total_time += elapsed

    accuracy = total_correct / len(SEEDS)
    return accuracy, total_dmc / len(SEEDS), total_time / len(SEEDS)


def populate_shared_buffer(buffer_path: Path, results: list[dict]):
    """Populate a single shared buffer with all seeds."""
    # Use the worst passing seed as the baseline (state 0)
    passing = [r for r in results if r["accuracy"] >= ACCURACY_GATE]
    if not passing:
        print("WARNING: No seeds pass the accuracy gate!")
        passing = results

    passing.sort(key=lambda r: r["dmc"], reverse=True)  # worst first
    baseline = passing[0]

    data = {"next_id": 0, "total_visits": 0, "states": {}}

    # State 0: baseline (worst passing seed)
    data["states"]["0"] = {
        "id": 0,
        "content": baseline["code"],
        "metric_value": baseline["dmc"],
        "reward": 0.0,
        "parent_id": None,
        "n_visits": 0,
        "best_child_reward": 0.0,
        "family": baseline["family"],
    }
    data["next_id"] = 1

    # Add remaining seeds as children of state 0
    for r in results:
        if r is baseline:
            continue
        if r["accuracy"] < ACCURACY_GATE:
            continue
        reward = max(0.0, baseline["dmc"] - r["dmc"])
        sid = data["next_id"]
        data["states"][str(sid)] = {
            "id": sid,
            "content": r["code"],
            "metric_value": r["dmc"],
            "reward": reward,
            "parent_id": 0,
            "n_visits": 0,
            "best_child_reward": 0.0,
            "family": r["family"],
        }
        data["next_id"] = sid + 1

    # Update baseline's best_child_reward
    max_reward = max(s["reward"] for s in data["states"].values())
    data["states"]["0"]["best_child_reward"] = max_reward

    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(buffer_path, "w") as f:
        json.dump(data, f, ensure_ascii=True)

    print(f"Shared buffer: {len(data['states'])} states -> {buffer_path}")


def populate_isolated_buffers(isolated_dir: Path, results: list[dict]):
    """Create one buffer per seed family."""
    isolated_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if r["accuracy"] < ACCURACY_GATE:
            print(f"  SKIP {r['family']} (accuracy {r['accuracy']:.0%})")
            continue

        buf_path = isolated_dir / f"reuse_buffer_sparse_parity_{r['family']}.json"
        data = {
            "next_id": 1,
            "total_visits": 0,
            "states": {
                "0": {
                    "id": 0,
                    "content": r["code"],
                    "metric_value": r["dmc"],
                    "reward": 0.0,
                    "parent_id": None,
                    "n_visits": 0,
                    "best_child_reward": 0.0,
                    "family": r["family"],
                }
            },
        }
        with open(buf_path, "w") as f:
            json.dump(data, f, ensure_ascii=True)
        print(f"  {r['family']}: DMC={r['dmc']:.0f} -> {buf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-path", default="/data/reuse_buffer_sparse_parity.json")
    parser.add_argument("--isolated-dir", default="/data/isolated_buffers")
    parser.add_argument("--no-tracker", action="store_true", help="Skip DMC tracking (accuracy only)")
    args = parser.parse_args()

    print("Evaluating seed solutions...\n")

    results = []
    for filename, family in SEED_FILES:
        path = SEED_DIR / filename
        code = path.read_text()
        print(f"  {family}...", end=" ", flush=True)
        accuracy, dmc, time_s = evaluate_seed(path, use_tracker=not args.no_tracker)
        status = "PASS" if accuracy >= ACCURACY_GATE else "FAIL"
        print(f"{status} accuracy={accuracy:.0%} dmc={dmc:,.0f} time={time_s:.4f}s")
        results.append({
            "family": family,
            "accuracy": accuracy,
            "dmc": dmc,
            "time_s": time_s,
            "code": code,
        })

    print(f"\n--- Populating buffers ---\n")
    populate_shared_buffer(Path(args.buffer_path), results)
    print()
    populate_isolated_buffers(Path(args.isolated_dir), results)

    # Print summary table
    print(f"\n{'Family':<25} {'Accuracy':>8} {'DMC':>12} {'Time':>8}")
    print(f"{'─'*25} {'─'*8} {'─'*12} {'─'*8}")
    for r in sorted(results, key=lambda r: r["dmc"]):
        acc_str = f"{r['accuracy']:.0%}"
        print(f"{r['family']:<25} {acc_str:>8} {r['dmc']:>12,.0f} {r['time_s']:>8.4f}s")


if __name__ == "__main__":
    main()
