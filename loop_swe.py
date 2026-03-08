"""
Multi-turn data collection loop using mini-swe-agent.

Each iteration:
  1. Build prompt from baseline train.py + experiment history
  2. Run mini-swe-agent editing session (model reads/edits train.py via bash)
  3. Dispatch modified train.py to GPU fleet
  4. Parse metrics, compute reward, log results

Replaces the single-turn diff-based approach in loop.py with a multi-turn
bash-tool approach that allows the model to verify its edits before submitting.

GPU split: GPU 0 = vLLM inference, GPU 1+ = train.py experiments (via fleet).
"""

import json
import os
import shutil
from itertools import count
from pathlib import Path

from bash_tool import create_isolated_workdir, run_agent_episode
from environment import compute_reward, parse_metrics
from loop import (
    AUTORESEARCH_DIR,
    HISTORY_TAIL,
    RESULTS_FILE,
    ROLLOUT_FILE,
    TRAIN_PY,
    ensure_vllm_server,
    init_results_file,
    read_history,
    start_vllm_server,
    stop_vllm_server,
)
from prompts import SYSTEM_PROMPT, build_instance_prompt
from runners import GPUPoolRunner


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_result(iteration: int, val_bpb: float, memory_gb: float, status: str, description: str):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{iteration}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def save_rollout(trajectory: list[dict], metrics: dict, reward: float, status: str, iteration: int):
    """Save the multi-turn trajectory as a rollout."""
    rollout = {
        "trajectory": trajectory,
        "metrics": metrics,
        "reward": reward,
        "status": status,
        "iteration": iteration,
    }
    os.makedirs(os.path.dirname(ROLLOUT_FILE), exist_ok=True)
    with open(ROLLOUT_FILE, "a") as f:
        f.write(json.dumps(rollout) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_results_file()
    start_vllm_server()

    pool = GPUPoolRunner()
    baseline = Path(AUTORESEARCH_DIR, TRAIN_PY).read_text()
    best_val_bpb = float("inf")

    print(f"[loop_swe] Starting. Results: {RESULTS_FILE}, Rollouts: {ROLLOUT_FILE}")

    try:
        for iteration in count(1):
            print(f"\n{'='*60}\n[loop_swe] Iteration {iteration}\n{'='*60}")

            # 1. Build prompts
            history = read_history()
            system = SYSTEM_PROMPT
            instance = build_instance_prompt(baseline, history)

            # 2. Create isolated workdir
            workdir = create_isolated_workdir(AUTORESEARCH_DIR)

            try:
                # 3. Run mini-swe-agent editing session
                print("[loop_swe] Starting agent editing session...")
                ensure_vllm_server()

                from minisweagent.models.litellm_model import LitellmModel
                model = LitellmModel(
                    model_name="openai/Qwen/Qwen3-32B",
                    model_kwargs={
                        "temperature": 0.7,
                        "max_tokens": 32768,
                        "drop_params": True,
                    },
                    cost_tracking="ignore_errors",
                )

                # Set env vars for local vLLM
                os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
                os.environ["OPENAI_API_KEY"] = "dummy"

                modified_train_py, trajectory = run_agent_episode(
                    workdir, model, system, instance, step_limit=20,
                )

                print(f"[loop_swe] Agent completed. Trajectory: {len(trajectory)} messages")

                # 4. Dispatch to GPU fleet
                print("[loop_swe] Dispatching to GPU fleet...")
                output = pool.run(modified_train_py)

                # 5. Handle crash
                if output.returncode != 0:
                    crash_tail = "\n".join(
                        (output.stderr or output.stdout or "no output").strip().splitlines()[-30:]
                    )
                    print(f"[loop_swe] Crash (exit {output.returncode})")
                    log_result(iteration, 0.0, 0.0, "crash", f"exit {output.returncode}")
                    save_rollout(trajectory, {}, -1.0, "crash", iteration)
                    continue

                # 6. Parse metrics
                metrics = parse_metrics(output.stdout)
                val_bpb = metrics.get("val_bpb")
                memory_gb = metrics.get("peak_vram_mb", 0) / 1024

                if val_bpb is None:
                    print("[loop_swe] Could not parse val_bpb")
                    log_result(iteration, 0.0, 0.0, "crash", "no val_bpb in output")
                    save_rollout(trajectory, metrics, -1.0, "crash", iteration)
                    continue

                # 7. Compute reward
                reward, status, feedback = compute_reward(val_bpb, best_val_bpb)
                if status == "improvement":
                    best_val_bpb = val_bpb
                feedback += f" Memory: {memory_gb:.1f} GB."

                print(f"[loop_swe] val_bpb={val_bpb:.6f} | best={best_val_bpb:.6f} | {status}")
                log_result(iteration, val_bpb, memory_gb, status, feedback[:60])
                save_rollout(trajectory, metrics, reward, status, iteration)

            except Exception as e:
                print(f"[loop_swe] Error in iteration {iteration}: {e}")
                log_result(iteration, 0.0, 0.0, "error", str(e)[:60])
                save_rollout([], {}, -1.0, "error", iteration)

            finally:
                shutil.rmtree(workdir, ignore_errors=True)

    except KeyboardInterrupt:
        print("\n[loop_swe] Interrupted.")
    finally:
        stop_vllm_server()
        print(f"[loop_swe] Done.")


if __name__ == "__main__":
    main()
