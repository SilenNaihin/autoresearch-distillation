"""Qwen3-14B ICL baseline for sparse parity challenge.

Standalone script using vLLM for local inference. Same PUCT reuse buffer,
evaluation harness, and feedback loop as the Opus ICL baseline.

Requires 1x A100 80GB (or equivalent) with ~45GB GPU memory for vLLM.
No SDPO training overhead — purely inference + evaluation.

Usage:
    python baselines/qwen_icl_sparse_parity.py --max-turns 50 --run-name qwen-icl-shared
    python baselines/qwen_icl_sparse_parity.py --max-turns 20 --buffer-path /tmp/qwen_reuse_buffer.json
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

import wandb

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_config import TaskConfig
from lib.reuse_buffer import ReuseBuffer
from lib.experiment_cache import ExperimentCache, cache_path_for

TASK_CONFIG = PROJECT_ROOT / "tasks" / "sparse_parity.yaml"
MODEL = "Qwen/Qwen3-14B"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results"


def load_task():
    return TaskConfig.from_yaml(TASK_CONFIG)


def make_diff(baseline: str, modified: str) -> str:
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


def build_prompt(
    task: TaskConfig,
    solve_py: str,
    history: list[dict],
) -> list[dict]:
    """Build chat messages for Qwen3-14B."""
    system_prompt = task.prompt.system
    instance_prompt = task.build_instance_prompt(solve_py)

    if history:
        feedback_block = "\n\n## Previous attempts\n"
        for h in history[-5:]:
            status = h.get("status", "unknown")
            dmc = h.get("dmc")
            acc = h.get("accuracy")
            diff = h.get("diff", "")
            feedback_block += f"\n### Attempt (status: {status})\n"
            if dmc is not None:
                feedback_block += f"- DMC: {dmc:,.0f}\n"
            if acc is not None:
                feedback_block += f"- Accuracy: {acc:.0%}\n"
            if diff and diff != "(no changes)":
                feedback_block += f"```diff\n{diff[:1000]}\n```\n"
            feedback_block += f"- Feedback: {h.get('feedback', 'N/A')}\n"
        instance_prompt += feedback_block

    instance_prompt += (
        "\n\nOutput ONLY the complete solve.py file content (the full file, not a diff). "
        "Start with the imports and function definition. Do not include markdown fences or explanations outside the code."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]


def extract_code(content: str) -> str:
    """Extract Python code from model response."""
    # Strip thinking tokens if present
    if "</think>" in content:
        content = content[content.index("</think>") + len("</think>"):].strip()
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-14B ICL baseline for sparse parity")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--run-name", type=str, default="qwen-icl-shared")
    parser.add_argument("--buffer-path", type=str, default="/tmp/qwen_reuse_buffer.json")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    # Import vLLM here so the script can be --help'd without GPU
    from vllm import LLM, SamplingParams

    task = load_task()
    buffer = ReuseBuffer(Path(args.buffer_path), direction=task.scoring.direction)
    cache = ExperimentCache(write_path=cache_path_for(task.name, "qwen_icl"))

    # Seed buffer with default solve.py if empty
    if len(buffer) == 0:
        default_solve = (PROJECT_ROOT / task.workspace.source_dir / task.workspace.target_file).read_text()
        buffer.seed(default_solve, task.scoring.baseline)

    baseline_code = buffer._data["states"].get("0", {}).get("content", "")

    # Initialize vLLM engine
    print(f"Loading {args.model} with vLLM (gpu_mem={args.gpu_memory_utilization}, max_len={args.max_model_len})...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    print("vLLM engine ready.")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["```\n\n"],  # stop if model tries to add explanation after code
    )

    # Init wandb
    config = {
        "model": args.model,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "buffer_path": args.buffer_path,
        "method": "qwen_icl",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    wandb.init(project="sparse-parity-sdpo", name=args.run_name, config=config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.run_name}-{int(time.time())}.jsonl"

    history: list[dict] = []
    best_dmc = float("inf")

    for turn in range(args.max_turns):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Turn {turn}/{args.max_turns}  |  Best DMC: {best_dmc:,.0f}  |  Buffer: {len(buffer)} states")
        print(f"{'='*60}")

        # PUCT selection
        selections = buffer.select(1)
        if selections:
            parent_id, selected_code = selections[0]
        else:
            parent_id = 0
            selected_code = baseline_code

        print(f"  Selected parent state: {parent_id}")

        # Build prompt and generate
        try:
            messages = build_prompt(task, selected_code, history)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            outputs = llm.generate([prompt_text], sampling_params)
            raw_output = outputs[0].outputs[0].text
            modified = extract_code(raw_output)
        except Exception as e:
            print(f"  Generation failed: {e}")
            history.append({"status": "generation_error", "feedback": str(e)})
            wandb.log({"turn": turn, "status": "generation_error", "reward": 0.0})
            continue

        # Debug: show first/last lines of extracted code
        code_lines = modified.split('\n')
        print(f"  Extracted {len(code_lines)} lines. First: {code_lines[0][:80] if code_lines else '(empty)'}")
        if len(code_lines) > 1:
            print(f"  Last: {code_lines[-1][:80]}")

        # Validate syntax before eval
        try:
            compile(modified, "<solve.py>", "exec")
        except SyntaxError as e:
            print(f"  SYNTAX ERROR: {e}")
            history.append({"status": "crash", "feedback": f"Syntax error: {e}"})
            wandb.log({"turn": turn, "status": "syntax_error", "reward": 0.0})
            continue

        diff_text = make_diff(selected_code, modified)
        cache_diff = make_diff(baseline_code, modified)

        if selected_code.strip() == modified.strip():
            print("  No changes made")
            history.append({"status": "no_change", "feedback": task.feedback.no_change})
            wandb.log({"turn": turn, "status": "no_change", "reward": 0.0})
            continue

        # Check cache
        cached = cache.get(cache_diff)
        if cached is not None:
            print(f"  Cached result: {cached}")
            history.append({"status": "cached", "feedback": "duplicate", **cached})
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
            crash_info = (eval_result["stderr"] or eval_result["stdout"] or "unknown")[:1000]
            print(f"CRASH: {crash_info}")
            history.append({
                "status": "crash",
                "feedback": f"Crash: {crash_info}",
                "diff": diff_text,
            })
            cache.put(cache_diff, {"crashed": True, "tail": crash_info})
            wandb.log({"turn": turn, "status": "crash", "reward": 0.0, "eval_time": eval_time})
            continue

        reward, status, reward_feedback = task.compute_reward(dmc)

        print(f"DMC={dmc:,.0f}  accuracy={accuracy}  reward={reward:.0f}  status={status}")

        if reward > 0:
            buffer.add(modified, dmc, reward, parent_id)
            if dmc < best_dmc:
                best_dmc = dmc

        history.append({
            "status": status,
            "dmc": dmc,
            "accuracy": accuracy,
            "feedback": reward_feedback + ("\n" + task.feedback.success if reward > 0 else "\n" + task.feedback.failure),
            "diff": diff_text,
        })

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
        })

        with output_path.open("a") as f:
            f.write(json.dumps({
                "turn": turn, "dmc": dmc, "accuracy": accuracy,
                "reward": reward, "status": status, "diff": diff_text,
                "parent_id": parent_id, "eval_time": eval_time,
            }) + "\n")

    print(f"\n{'='*60}")
    print(f"Qwen ICL baseline complete: {args.max_turns} turns")
    print(f"Best DMC: {best_dmc:,.0f}")
    print(f"Buffer: {len(buffer)} states")
    print(f"Output: {output_path}")

    wandb.log({"final/best_dmc": best_dmc, "final/buffer_size": len(buffer)})
    wandb.finish()


if __name__ == "__main__":
    main()
