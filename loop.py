"""
Self-distillation loop: Qwen3 4B proposes patches to train.py, we run them, collect rollouts.

Each iteration is independent:
  1. Read baseline train.py + results history
  2. Model outputs a unified diff (git patch)
  3. Apply patch, run experiment on GPU 1, parse results
  4. Revert train.py to baseline
  5. Log results + save rollout for later SFT/DPO training

GPU split: GPU 0 = vLLM inference, GPU 1 = train.py experiments.
"""

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLLM_MODEL = "Qwen/Qwen3-4B"
VLLM_PORT = 8000
INFERENCE_GPU = "0"
EXPERIMENT_GPU = "1"
MAX_TOKENS = 32768
TEMPERATURE = 0.7
EXPERIMENT_TIMEOUT = 600
MAX_CONSECUTIVE_FAILURES = 3
VLLM_STARTUP_TIMEOUT = 300
RESULTS_FILE = "results.tsv"
ROLLOUT_FILE = "rollouts/rollouts.jsonl"
AUTORESEARCH_DIR = "autoresearch"
TRAIN_PY = "train.py"  # relative to AUTORESEARCH_DIR
HISTORY_TAIL = 20

# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

_vllm_process = None


def start_vllm_server():
    global _vllm_process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = INFERENCE_GPU

    vllm_bin = os.path.expanduser("~/vllm-env/bin/vllm")
    cmd = [vllm_bin] if os.path.exists(vllm_bin) else ["vllm"]
    cmd += [
        "serve", VLLM_MODEL,
        "--port", str(VLLM_PORT),
        "--max-model-len", "32768",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.9",
    ]

    print(f"[loop] Starting vLLM: {' '.join(cmd)}")
    _vllm_process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    url = f"http://localhost:{VLLM_PORT}/health"
    deadline = time.time() + VLLM_STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                print("[loop] vLLM healthy.")
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        if _vllm_process.poll() is not None:
            out = _vllm_process.stdout.read().decode(errors="replace")
            raise RuntimeError(f"vLLM died (code {_vllm_process.returncode}):\n{out[-2000:]}")
        time.sleep(3)
    raise TimeoutError(f"vLLM not healthy after {VLLM_STARTUP_TIMEOUT}s")


def stop_vllm_server():
    global _vllm_process
    if _vllm_process and _vllm_process.poll() is None:
        _vllm_process.terminate()
        try:
            _vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _vllm_process.kill()
    _vllm_process = None


def ensure_vllm_server():
    if _vllm_process is None or _vllm_process.poll() is not None:
        stop_vllm_server()
        start_vllm_server()


# ---------------------------------------------------------------------------
# vLLM API
# ---------------------------------------------------------------------------

def call_vllm(system: str, user: str) -> str:
    url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
    payload = json.dumps({
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:500]}") from e
    return json.loads(resp.read())["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous ML researcher optimizing a GPT pretraining script.

## Your task
Propose a modification to train.py that will lower val_bpb (bits per byte on validation data). The training budget is fixed at 5 minutes wall-clock. Lower val_bpb is better.

## Rules
- You may ONLY modify train.py. Do not touch prepare.py or any other file.
- You can only use packages already in pyproject.toml: torch, numpy, kernels, matplotlib, pandas, pyarrow, requests, rustbpe, tiktoken.
- Do not modify the evaluation harness. The evaluate_bpb function in prepare.py is the ground truth.
- VRAM is a soft constraint. Some increase is OK for meaningful val_bpb gains.
- Simpler is better. A small improvement that adds ugly complexity is not worth it.
- The script runs on a single GPU via: uv run train.py

## Output format
Output your reasoning, then a single unified diff (git patch) that can be applied with `git apply`. The diff must modify only train.py.

Format your diff inside a fenced code block:

```diff
--- a/train.py
+++ b/train.py
@@ ... @@
 context line
-old line
+new line
 context line
```

Output exactly one diff block. Do not output multiple separate diffs.\
"""


def build_user_message(train_py_content: str, history_lines: list[str]) -> str:
    parts = ["## Current train.py\n```python\n" + train_py_content + "\n```"]

    if history_lines:
        parts.append("## Experiment history (recent)\n```\n" + "\n".join(history_lines) + "\n```")
        parts.append(
            "Each row shows: iteration, val_bpb, memory_gb, status, description.\n"
            "Learn from past experiments. Don't repeat things that didn't work. "
            "Build on ideas that improved val_bpb."
        )
    else:
        parts.append("No experiments have been run yet. This is the baseline train.py.")

    parts.append(
        "Propose a single modification to train.py that you think will lower val_bpb. "
        "Output your reasoning followed by a unified diff."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Diff parsing and application
# ---------------------------------------------------------------------------

def extract_diff(response: str) -> str | None:
    m = re.search(r"```diff\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    m = re.search(r"(--- a/.*?\n\+\+\+ b/.*?\n@@.*?)(?:\n```|$)", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return None


def apply_patch(diff_text: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff_text)
        patch_path = f.name
    try:
        r = subprocess.run(["git", "apply", "--check", patch_path],
                           capture_output=True, text=True, cwd=AUTORESEARCH_DIR, timeout=10)
        if r.returncode == 0:
            r = subprocess.run(["git", "apply", patch_path],
                               capture_output=True, text=True, cwd=AUTORESEARCH_DIR, timeout=10)
            if r.returncode == 0:
                return True, ""
            return False, r.stderr.strip()
        r = subprocess.run(["git", "apply", "--3way", patch_path],
                           capture_output=True, text=True, cwd=AUTORESEARCH_DIR, timeout=10)
        if r.returncode == 0:
            return True, ""
        return False, r.stderr.strip()
    finally:
        os.unlink(patch_path)


def revert_train_py():
    subprocess.run(["git", "checkout", "--", TRAIN_PY], cwd=AUTORESEARCH_DIR, check=True, timeout=10)


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run_experiment() -> tuple[str, str, int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = EXPERIMENT_GPU
    try:
        r = subprocess.run(["uv", "run", "train.py"], capture_output=True, text=True,
                           env=env, cwd="autoresearch", timeout=EXPERIMENT_TIMEOUT)
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT: exceeded {EXPERIMENT_TIMEOUT}s", -1


def parse_results(output: str) -> dict:
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        for key in ("val_bpb", "peak_vram_mb", "training_seconds", "total_seconds",
                     "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth"):
            if line.startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return metrics


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results_file():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("iteration\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def read_history() -> list[str]:
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE) as f:
        lines = f.read().strip().splitlines()
    if len(lines) <= 1:
        return []
    return [lines[0]] + lines[1:][-HISTORY_TAIL:]


def log_result(iteration: int, val_bpb: float, memory_gb: float, status: str, description: str):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{iteration}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


# ---------------------------------------------------------------------------
# Rollout saving
# ---------------------------------------------------------------------------

def save_rollout(system: str, user: str, assistant: str, feedback: str, metadata: dict):
    rollout = {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": feedback},
        ],
        "metadata": metadata,
    }
    os.makedirs(os.path.dirname(ROLLOUT_FILE), exist_ok=True)
    with open(ROLLOUT_FILE, "a") as f:
        f.write(json.dumps(rollout) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_description(response: str, max_len: int = 80) -> str:
    for line in response.splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "---", "+++", "@@", "-", "+")):
            continue
        line = re.sub(r"^#+\s*", "", line)
        if len(line) > 10:
            return line[:max_len]
    return "no description"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_results_file()
    start_vllm_server()

    baseline = Path(AUTORESEARCH_DIR, TRAIN_PY).read_text()
    iteration = 0
    consecutive_failures = 0
    best_val_bpb = float("inf")

    print(f"[loop] Starting. Results: {RESULTS_FILE}, Rollouts: {ROLLOUT_FILE}")

    try:
        while True:
            iteration += 1
            print(f"\n{'='*60}\n[loop] Iteration {iteration}\n{'='*60}")

            revert_train_py()
            history = read_history()
            system = SYSTEM_PROMPT
            user_msg = build_user_message(baseline, history)

            # Query model
            print("[loop] Querying model...")
            ensure_vllm_server()
            try:
                response = call_vllm(system, user_msg)
            except Exception as e:
                print(f"[loop] vLLM call failed: {e}")
                consecutive_failures += 1
                save_rollout(system, user_msg, "", f"ERROR: {e}",
                             {"status": "error", "iteration": iteration})
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    stop_vllm_server()
                    consecutive_failures = 0
                continue

            print(f"[loop] Response: {len(response)} chars")

            # Extract diff
            diff_text = extract_diff(response)
            if diff_text is None:
                print("[loop] No diff found in response")
                log_result(iteration, 0.0, 0.0, "parse_error", "no diff in response")
                save_rollout(system, user_msg, response,
                             "ERROR: No unified diff found. Output a ```diff ... ``` block.",
                             {"status": "parse_error", "iteration": iteration})
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
                continue

            # Apply patch
            success, error = apply_patch(diff_text)
            if not success:
                print(f"[loop] Patch failed: {error}")
                revert_train_py()
                log_result(iteration, 0.0, 0.0, "patch_error", f"patch failed: {error[:60]}")
                save_rollout(system, user_msg, response, f"ERROR: Patch failed: {error}",
                             {"status": "patch_error", "iteration": iteration})
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
                continue

            description = extract_description(response)
            print(f"[loop] Running experiment: {description}")

            # Run experiment
            stdout, stderr, returncode = run_experiment()

            # Crashed
            if returncode != 0:
                print(f"[loop] Crash (exit {returncode})")
                revert_train_py()
                crash_tail = "\n".join((stderr or stdout or "no output").strip().splitlines()[-30:])
                log_result(iteration, 0.0, 0.0, "crash", description[:60])
                save_rollout(system, user_msg, response, f"CRASH (exit {returncode}):\n{crash_tail}",
                             {"status": "crash", "iteration": iteration})
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
                continue

            # Parse metrics
            metrics = parse_results(stdout)
            val_bpb = metrics.get("val_bpb")
            memory_gb = metrics.get("peak_vram_mb", 0) / 1024

            if val_bpb is None:
                print("[loop] Could not parse val_bpb")
                revert_train_py()
                tail = "\n".join(stdout.strip().splitlines()[-20:])
                log_result(iteration, 0.0, 0.0, "crash", f"no val_bpb: {description[:50]}")
                save_rollout(system, user_msg, response, f"ERROR: no val_bpb in output:\n{tail}",
                             {"status": "crash", "iteration": iteration})
                consecutive_failures += 1
                continue

            # Revert (always — each iteration is independent)
            revert_train_py()

            # Evaluate
            improved = val_bpb < best_val_bpb
            if improved:
                delta = val_bpb - best_val_bpb
                old_best = best_val_bpb
                best_val_bpb = val_bpb
                status = "improvement"
                feedback = (f"SUCCESS: val_bpb={val_bpb:.6f} (new best). "
                            f"Previous best: {old_best:.6f}, delta: {delta:.6f}. "
                            f"Memory: {memory_gb:.1f} GB.")
            else:
                status = "no_improvement"
                feedback = (f"NO IMPROVEMENT: val_bpb={val_bpb:.6f}, best={best_val_bpb:.6f}. "
                            f"Memory: {memory_gb:.1f} GB. Try something different.")

            print(f"[loop] val_bpb={val_bpb:.6f} | best={best_val_bpb:.6f} | {status}")
            log_result(iteration, val_bpb, memory_gb, status, description[:60])
            save_rollout(system, user_msg, response, feedback,
                         {"status": status, "val_bpb": val_bpb, "best_val_bpb": best_val_bpb,
                          "memory_gb": memory_gb, "iteration": iteration})
            consecutive_failures = 0

    except KeyboardInterrupt:
        print("\n[loop] Interrupted.")
    finally:
        revert_train_py()
        stop_vllm_server()
        print(f"[loop] Done. {iteration} iterations.")


if __name__ == "__main__":
    main()
