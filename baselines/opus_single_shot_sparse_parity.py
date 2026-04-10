"""Opus 4.6 single-shot baseline for sparse parity challenge.

Like the ICL baseline, but each turn is independent (no feedback history).
The buffer is pre-seeded with 7 known seed solutions so the model gets
diverse starting points via PUCT selection.

Usage:
    python baselines/opus_single_shot_sparse_parity.py --max-turns 50 --run-name opus-single-shot
    python baselines/opus_single_shot_sparse_parity.py --max-turns 50 --buffer-path /data/buffers/single_shot.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from difflib import unified_diff
from pathlib import Path

import litellm
import wandb

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_config import TaskConfig
from lib.reuse_buffer import ReuseBuffer
from lib.experiment_cache import ExperimentCache, cache_path_for

TASK_CONFIG = PROJECT_ROOT / "tasks" / "sparse_parity.yaml"
MODEL = "bedrock/us.anthropic.claude-opus-4-6-v1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results"
SEEDS_DIR = PROJECT_ROOT / "sparse_parity" / "seeds"

# Estimated DMC scores for seed solutions (lower = better).
# Measured values where available; conservative estimates otherwise.
SEED_ESTIMATED_DMC = {
    "01_correlation.py": 2_900_000,       # measured
    "02_gf2_gaussian.py": 738_000,        # measured
    "03_sequential_elimination.py": 1_500_000,
    "04_walsh_hadamard.py": 2_000_000,
    "05_brute_force.py": 3_000_000,
    "06_recursive_bisection.py": 2_500_000,
    "07_coordinate_descent.py": 1_800_000,
}


def load_task():
    """Load the sparse parity task config."""
    return TaskConfig.from_yaml(TASK_CONFIG)


def load_seed_solutions() -> list[tuple[str, str, float]]:
    """Load all seed .py files from the seeds directory.

    Returns list of (filename, content, estimated_dmc) tuples.
    """
    seeds = []
    for seed_file in sorted(SEEDS_DIR.glob("*.py")):
        if seed_file.name == "__init__.py":
            continue
        content = seed_file.read_text()
        estimated_dmc = SEED_ESTIMATED_DMC.get(seed_file.name, 2_000_000)
        seeds.append((seed_file.name, content, estimated_dmc))
    return seeds


def make_diff(baseline: str, modified: str) -> str:
    """Produce a unified diff between two code strings."""
    if baseline == modified:
        return "(no changes)"
    return "".join(unified_diff(
        baseline.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="a/solve.py", tofile="b/solve.py",
    ))


def evaluate_solve_py(solve_py_content: str, task: TaskConfig) -> dict:
    """Run evaluate.py in an isolated temp dir. Returns parsed metrics."""
    tmpdir = tempfile.mkdtemp(prefix="sparse_parity_eval_")
    try:
        src_dir = PROJECT_ROOT / task.workspace.source_dir
        for f in src_dir.iterdir():
            if f.name == "__pycache__":
                continue
            if f.is_file():
                shutil.copy2(f, tmpdir)
            elif f.is_dir():
                shutil.copytree(f, Path(tmpdir) / f.name, ignore=shutil.ignore_patterns("__pycache__"))

        (Path(tmpdir) / task.workspace.target_file).write_text(solve_py_content)

        env = os.environ.copy()
        result = subprocess.run(
            ["python3", "evaluate.py"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        metrics = task.parse_metrics(result.stdout) if result.stdout else {}
        return {
            "metrics": metrics,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"metrics": {}, "returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"metrics": {}, "returncode": -1, "stdout": "", "stderr": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_agent_turn(
    task: TaskConfig,
    solve_py: str,
    temperature: float = 1.0,
    model: str = MODEL,
) -> str:
    """One agent turn: send solve.py, get back modified solve.py.

    Single-shot: no feedback history is provided. Each turn is independent.
    """
    system_prompt = task.prompt.system
    instance_prompt = task.build_instance_prompt(solve_py)

    instance_prompt += (
        "\n\nOutput ONLY the complete solve.py file content (the full file, not a diff). "
        "Start with the imports and function definition. Do not include markdown fences or explanations outside the code."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
    )

    content = response.choices[0].message.content or ""

    # Extract Python code from response
    if "```python" in content:
        start = content.index("```python") + len("```python")
        end = content.index("```", start)
        return content[start:end].strip()
    elif "```" in content:
        start = content.index("```") + 3
        newline = content.index("\n", start)
        start = newline + 1
        end = content.index("```", start)
        return content[start:end].strip()
    elif "import numpy" in content or "def solve" in content:
        lines = content.split("\n")
        code_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ", '"""', "def ")):
                code_start = i
                break
        return "\n".join(lines[code_start:]).strip()

    return content.strip()


def seed_buffer(buffer: ReuseBuffer, task: TaskConfig) -> None:
    """Pre-seed the PUCT buffer with the 7 seed solutions.

    Uses buffer.seed() for the default solve.py (id=0), then buffer.add()
    for each seed solution with estimated DMC scores and computed rewards.
    """
    baseline = task.scoring.baseline  # 10M

    # Seed id=0: the default solve.py from the workspace
    default_solve = (PROJECT_ROOT / task.workspace.source_dir / task.workspace.target_file).read_text()
    buffer.seed(default_solve, baseline)

    # Load and add each seed solution
    seeds = load_seed_solutions()
    for filename, content, estimated_dmc in seeds:
        # reward = baseline - dmc (for minimize direction, improvement over baseline)
        reward = max(0.0, baseline - estimated_dmc)
        buffer.add(content, estimated_dmc, reward, parent_id=0)
        print(f"  Seeded: {filename}  DMC~{estimated_dmc:,.0f}  reward={reward:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Opus 4.6 single-shot baseline for sparse parity")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--run-name", type=str, default="opus-single-shot")
    parser.add_argument("--buffer-path", type=str, default="/data/reuse_buffer_sparse_parity_single_shot.json")
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    model_name = args.model

    task = load_task()
    buffer = ReuseBuffer(Path(args.buffer_path), direction=task.scoring.direction)
    cache = ExperimentCache(write_path=cache_path_for(task.name, "opus_single_shot"))

    # Pre-seed buffer with 7 seed solutions if empty
    if len(buffer) == 0:
        print("Seeding buffer with seed solutions...")
        seed_buffer(buffer, task)
        print(f"Buffer seeded: {len(buffer)} states\n")

    baseline_code = buffer._data["states"].get("0", {}).get("content", "")

    # Init wandb
    config = {
        "model": model_name,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "buffer_path": args.buffer_path,
        "method": "opus_single_shot",
        "n_seeds": 7,
        "feedback_history": False,
    }
    wandb.init(project="sparse-parity-sdpo", name=args.run_name, config=config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.run_name}-{int(time.time())}.jsonl"

    best_dmc = float("inf")

    for turn in range(args.max_turns):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Turn {turn}/{args.max_turns}  |  Best DMC: {best_dmc:,.0f}  |  Buffer: {len(buffer)} states")
        print(f"{'='*60}")

        # PUCT selection — model gets a diverse starting point each turn
        selections = buffer.select(1)
        if selections:
            parent_id, selected_code = selections[0]
        else:
            parent_id = 0
            selected_code = baseline_code

        print(f"  Selected parent state: {parent_id}")

        # Generate modified solve.py (single-shot: no history)
        try:
            modified = run_agent_turn(task, selected_code, args.temperature, model=model_name)
        except Exception as e:
            print(f"  Generation failed: {e}")
            wandb.log({"turn": turn, "status": "generation_error", "reward": 0.0})
            continue

        diff_text = make_diff(selected_code, modified)
        cache_diff = make_diff(baseline_code, modified)

        if selected_code.strip() == modified.strip():
            print("  No changes made")
            wandb.log({"turn": turn, "status": "no_change", "reward": 0.0})
            continue

        # Check cache
        cached = cache.get(cache_diff)
        if cached is not None:
            print(f"  Cached result: {cached}")
            wandb.log({"turn": turn, "status": "cached", "reward": 0.0})
            continue

        # Evaluate
        print("  Evaluating...", end=" ", flush=True)
        eval_result = evaluate_solve_py(modified, task)
        eval_time = time.time() - t0

        metrics = eval_result["metrics"]
        dmc = metrics.get("dmc")
        accuracy = metrics.get("accuracy")

        if eval_result["returncode"] != 0 or dmc is None:
            crash_info = (eval_result["stderr"] or eval_result["stdout"] or "unknown")[:500]
            print(f"CRASH: {crash_info[:100]}")
            cache.put(cache_diff, {"crashed": True, "tail": crash_info})
            wandb.log({"turn": turn, "status": "crash", "reward": 0.0, "eval_time": eval_time})
            continue

        reward, status, reward_feedback = task.compute_reward(dmc)

        print(f"DMC={dmc:,.0f}  accuracy={accuracy}  reward={reward:.0f}  status={status}")

        # Update buffer if improvement over baseline
        if reward > 0:
            buffer.add(modified, dmc, reward, parent_id)
            if dmc < best_dmc:
                best_dmc = dmc

        cache.put(cache_diff, {"metric_value": dmc, "dmc": dmc, "accuracy": accuracy},
                  metric_value=dmc, diff_text_raw=cache_diff)

        wandb.log({
            "turn": turn,
            "env/dmc": dmc,
            "env/accuracy": accuracy,
            "reward": reward,
            "best_dmc": best_dmc,
            "buffer_size": len(buffer),
            "eval_time": eval_time,
            "status": status,
            "parent_id": parent_id,
        })

        # Log to JSONL
        with output_path.open("a") as f:
            f.write(json.dumps({
                "turn": turn, "dmc": dmc, "accuracy": accuracy,
                "reward": reward, "status": status, "diff": diff_text,
                "parent_id": parent_id, "eval_time": eval_time,
            }) + "\n")

    print(f"\n{'='*60}")
    print(f"Opus single-shot baseline complete: {args.max_turns} turns")
    print(f"Best DMC: {best_dmc:,.0f}")
    print(f"Buffer: {len(buffer)} states")
    print(f"Output: {output_path}")

    wandb.log({"final/best_dmc": best_dmc, "final/buffer_size": len(buffer)})
    wandb.finish()


if __name__ == "__main__":
    main()
