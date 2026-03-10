"""
Multi-turn baseline: Qwen3-14B base model with mini-swe-agent, no weight updates.

Identical to the SDPO loop except no training — measures pure in-context learning.
The model uses bash tools to edit train.py (same interface as SDPO's agent_loop.py),
then we dispatch the modified file to the A100 for a 5-min experiment.

Architecture:
    box1 (h100_azure): vLLM server (Qwen3-14B) + this script
    a100-backup-1:     experiment execution

Setup (on box1):
    pip install vllm wandb mini-swe-agent
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-14B --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.9
    # Then run baseline:
    python loop_baseline.py --max-turns 30

Compare wandb curves: autoresearch-baseline vs autoresearch-sdpo
"""

import argparse
import json
import os
import shutil
import time
from difflib import unified_diff
from pathlib import Path

import wandb

from bash_tool import create_isolated_workdir, run_agent_episode
from environment import BASELINE_VAL_BPB, compute_reward, parse_metrics
from prompts import SYSTEM_PROMPT, build_instance_prompt
from runners import FLEET, SSHRunner

# Experiments run on A100 backup only — H100 fleet reserved for SDPO.
EXPERIMENT_FLEET = [s for s in FLEET if s.name == "a100-gpu0"]

MODEL = "Qwen/Qwen3-14B"
VLLM_BASE_URL = "http://localhost:8000/v1"
OUTPUT_DIR = "outputs/baseline"


def make_diff(baseline: str, modified: str) -> str:
    if baseline == modified:
        return "(no changes)"
    return "".join(unified_diff(
        baseline.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="a/train.py", tofile="b/train.py",
    ))


def dispatch_experiment(train_py: str) -> tuple[dict, int, str]:
    """Dispatch modified train.py to experiment GPU.

    Returns (metrics_dict, returncode, feedback_text).
    """
    for slot in EXPERIMENT_FLEET:
        runner = SSHRunner(slot, timeout=600)
        output = runner.run(train_py)
        if output.returncode == 255:
            print(f"  [baseline] {slot.name} SSH failed, trying next...")
            continue

        metrics = parse_metrics(output.stdout) if output.stdout else {}
        val_bpb = metrics.get("val_bpb")

        if output.returncode != 0:
            crash_tail = (output.stderr or output.stdout or "no output").strip()[-500:]
            return metrics, output.returncode, f"Experiment crashed (exit {output.returncode}):\n{crash_tail}"

        if val_bpb is None:
            tail = "\n".join((output.stdout or "").strip().splitlines()[-15:])
            return metrics, 0, f"Experiment ran but produced no val_bpb. Output tail:\n{tail}"

        return metrics, 0, ""

    return {}, -1, "All experiment GPUs unreachable."


def log_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-turn baseline (in-context learning)")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--step-limit", type=int, default=20,
                        help="Max bash tool calls per episode (same as SDPO)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--vllm-base-url", type=str, default=VLLM_BASE_URL)
    parser.add_argument("--run-name", type=str, default="qwen3-14b-baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / f"{args.run_name}-{int(time.time())}.jsonl"

    baseline = Path("autoresearch/train.py").read_text()

    print(f"Model:       {args.model}")
    print(f"vLLM:        {args.vllm_base_url}")
    print(f"Max turns:   {args.max_turns}")
    print(f"Step limit:  {args.step_limit}")
    print(f"Temperature: {args.temperature}")
    print(f"Fleet:       {[s.name for s in EXPERIMENT_FLEET]}")
    print(f"Output:      {output_path}")
    print()

    # Create mini-swe-agent model pointing at local vLLM
    from minisweagent.models.litellm_model import LitellmModel
    model = LitellmModel(
        model_name=f"openai/{args.model}",
        model_kwargs={
            "api_base": args.vllm_base_url,
            "api_key": "dummy",
            "temperature": args.temperature,
        },
        cost_tracking="ignore_errors",
    )

    # Init wandb
    config = {
        "model": args.model,
        "max_turns": args.max_turns,
        "step_limit": args.step_limit,
        "temperature": args.temperature,
        "seed": args.seed,
        "fleet": [s.name for s in EXPERIMENT_FLEET],
        "baseline_val_bpb": BASELINE_VAL_BPB,
        "method": "multi-turn mini-swe-agent (no weight updates)",
    }
    wandb.init(project="autoresearch-baseline", name=args.run_name, config=config)
    log_jsonl(output_path, {"config": config})

    # Multi-turn loop
    best_val_bpb = float("inf")
    feedback_history: list[str] = []

    for turn in range(args.max_turns):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Turn {turn}/{args.max_turns}")
        print(f"{'='*60}")

        # Build instance prompt with accumulated feedback
        instance_prompt = build_instance_prompt(baseline, [])
        if feedback_history:
            feedback_block = "\n\n".join(
                f"### Attempt {i+1}\n{fb}" for i, fb in enumerate(feedback_history)
            )
            instance_prompt += (
                f"\n\n## Results from previous attempts\n{feedback_block}"
                "\n\nLearn from these results. Don't repeat things that didn't work. "
                "Build on approaches that improved val_bpb."
            )

        # Run mini-swe-agent episode (model edits train.py via bash)
        workdir = create_isolated_workdir()
        try:
            modified, trajectory = run_agent_episode(
                workdir=workdir,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                instance_prompt=instance_prompt,
                step_limit=args.step_limit,
            )
        except Exception as e:
            print(f"  Agent episode failed: {e}")
            feedback_history.append(f"Agent error: {e}")
            wandb.log({"turn": turn, "status": "agent_error", "reward": 0.0})
            log_jsonl(output_path, {"turn": turn, "status": "agent_error", "error": str(e)})
            shutil.rmtree(workdir, ignore_errors=True)
            continue
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        diff_text = make_diff(baseline, modified)
        n_tool_calls = sum(1 for m in trajectory if m.get("role") == "assistant"
                          and "tool_calls" in str(m.get("content", "")))

        print(f"  Tool calls: {n_tool_calls}")
        print(f"  Diff size:  {len(diff_text)} chars")

        if baseline == modified:
            feedback = "No changes were made to train.py."
            print(f"  Status: no_changes")
            feedback_history.append(feedback)
            wandb.log({"turn": turn, "status": "no_changes", "reward": 0.0,
                        "tool_calls": n_tool_calls})
            log_jsonl(output_path, {"turn": turn, "status": "no_changes",
                                    "tool_calls": n_tool_calls})
            continue

        # Dispatch to experiment GPU
        print(f"  Dispatching experiment to {EXPERIMENT_FLEET[0].name}...")
        metrics, returncode, error_feedback = dispatch_experiment(modified)
        val_bpb = metrics.get("val_bpb")
        experiment_time = time.time() - t0

        if returncode != 0 or (returncode == 0 and val_bpb is None and error_feedback):
            feedback = f"Changes:\n{diff_text[:1000]}\n\n{error_feedback}"
            status = "crash" if returncode != 0 else "no_metric"
            print(f"  Status: {status}")
            feedback_history.append(feedback)
            wandb.log({"turn": turn, "status": status, "reward": 0.0,
                        "experiment_time": experiment_time, "tool_calls": n_tool_calls})
            log_jsonl(output_path, {"turn": turn, "status": status,
                                    "diff": diff_text[:2000], "feedback": error_feedback,
                                    "experiment_time": experiment_time})
            continue

        # Compute reward
        reward, status, reward_feedback = compute_reward(val_bpb, best_val_bpb)
        if status == "improvement":
            best_val_bpb = val_bpb

        memory_gb = metrics.get("peak_vram_mb", 0) / 1024
        feedback = f"Changes:\n{diff_text[:1000]}\n\n{reward_feedback} Memory: {memory_gb:.1f} GB."
        feedback_history.append(feedback)

        print(f"  val_bpb={val_bpb:.6f}  best={best_val_bpb:.6f}  "
              f"reward={reward:.4f}  status={status}  time={experiment_time:.0f}s")

        wandb.log({
            "turn": turn, "val_bpb": val_bpb, "best_val_bpb": best_val_bpb,
            "reward": reward, "memory_gb": memory_gb,
            "experiment_time": experiment_time, "status": status,
            "tool_calls": n_tool_calls,
            **{f"env/{k}": v for k, v in metrics.items()},
        })
        log_jsonl(output_path, {
            "turn": turn, "val_bpb": val_bpb, "best_val_bpb": best_val_bpb,
            "reward": reward, "status": status, "memory_gb": memory_gb,
            "experiment_time": experiment_time, "diff": diff_text,
            "metrics": metrics, "tool_calls": n_tool_calls,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"Baseline complete: {args.max_turns} turns")
    print(f"Best val_bpb: {best_val_bpb:.6f} (baseline: {BASELINE_VAL_BPB})")
    print(f"Output: {output_path}")

    wandb.log({"final/best_val_bpb": best_val_bpb,
               "final/improvement": max(0, BASELINE_VAL_BPB - best_val_bpb)})
    wandb.finish()


if __name__ == "__main__":
    main()
