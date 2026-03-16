"""
Baseline comparison: SDPO-trained model vs base Qwen3-14B.

Runs both models through the autoresearch task (multi-turn bash editing),
dispatches their edits to GPU fleet, and compares val_bpb.

Usage (on box3):
    python eval_baseline.py --model /data/models/autoresearch-14b-sdpo-step160 --name sdpo-step160
    python eval_baseline.py --model Qwen/Qwen3-14B --name base-14b
    python eval_baseline.py --model Qwen/Qwen3-4B --name base-4b

Each run produces a JSONL with the trajectory + experiment results.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from openai import OpenAI

AUTORESEARCH_DIR = os.path.join(os.path.dirname(__file__), "autoresearch")
SUBMIT_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def create_workdir():
    """Copy autoresearch/ to a temp dir."""
    tmpdir = tempfile.mkdtemp(prefix="eval_workdir_", dir="/data/tmp")
    shutil.copytree(AUTORESEARCH_DIR, tmpdir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc", ".venv"))
    return tmpdir


def run_bash(workdir: str, command: str, timeout: int = 30) -> str:
    """Execute bash in workdir, return output."""
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           cwd=workdir, timeout=timeout,
                           env={**os.environ, "PAGER": "cat", "TQDM_DISABLE": "1"})
        out = r.stdout
        if r.stderr:
            out += ("\n" if out else "") + r.stderr
        return out or f"(exit code {r.returncode})"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"


def run_experiment(train_py: str, gpu_slot: str = "box5-gpu0") -> dict:
    """Dispatch train.py to a GPU and return metrics."""
    from environment import parse_metrics, compute_reward, RunOutput
    from runners import SSHRunner, FLEET

    slot = next((s for s in FLEET if s.name == gpu_slot), FLEET[0])
    runner = SSHRunner(slot, timeout=600)
    output = runner.run(train_py)

    result = {
        "returncode": output.returncode,
        "stdout_tail": output.stdout[-2000:] if output.stdout else "",
        "stderr_tail": output.stderr[-500:] if output.stderr else "",
    }

    if output.returncode != 0:
        result["status"] = "crash"
        result["val_bpb"] = None
        result["reward"] = 0.0
        return result

    metrics = parse_metrics(output.stdout)
    val_bpb = metrics.get("val_bpb")
    reward, status, feedback = compute_reward(val_bpb)

    result.update({
        "status": status,
        "val_bpb": val_bpb,
        "reward": reward,
        "feedback": feedback,
        "metrics": metrics,
    })
    return result


def _execute_text_commands(workdir: str, content: str) -> int:
    """Extract sed/bash commands from model text output and execute them.

    Some models output commands as plain text instead of tool calls.
    Returns number of commands executed.
    """
    import re
    commands_run = 0
    # Match sed commands and echo commands in text
    for pattern in [
        r"```(?:bash|sh)?\n(.*?)```",  # fenced code blocks
        r"(sed\s+-i\s+.*?)(?:\n|$)",    # inline sed commands
    ]:
        for match in re.finditer(pattern, content, re.DOTALL):
            cmd = match.group(1).strip()
            for line in cmd.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("sed ", "echo ", "cat ", "python")):
                    if SUBMIT_SIGNAL in line:
                        continue
                    run_bash(workdir, line)
                    commands_run += 1
    return commands_run


def run_agent(client: OpenAI, model: str, system_prompt: str, instance_prompt: str,
              workdir: str, max_turns: int = 5, is_vllm: bool = True) -> tuple[list[dict], bool]:
    """Run multi-turn agent loop. Returns (messages, submitted)."""
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the working directory containing train.py",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"]
            }
        }
    }]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]

    submitted = False
    for turn in range(max_turns):
        kwargs = dict(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=16384,
        )
        if is_vllm:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        response = client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                command = args.get("command", "")

                if SUBMIT_SIGNAL in command:
                    submitted = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Submission received. Your changes will be evaluated."
                    })
                    break

                output = run_bash(workdir, command)
                if SUBMIT_SIGNAL in output.split("\n")[0]:
                    submitted = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Submission received. Your changes will be evaluated."
                    })
                    break

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output[:10000]
                })

            if submitted:
                break
        else:
            # Check if content contains bash commands or submit signal as text
            # (some models output tool-like content as plain text)
            content = msg.content or ""
            if SUBMIT_SIGNAL in content:
                # Extract and run any sed/bash commands from the text before submit
                _execute_text_commands(workdir, content)
                submitted = True
                break

            # Try to extract and run inline commands from text content
            cmds_run = _execute_text_commands(workdir, content)
            if cmds_run > 0:
                # Model made edits via text — treat this turn as productive
                messages.append({
                    "role": "user",
                    "content": "Commands executed. Submit when ready: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                })
            elif turn < max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "Please make your edits using the bash tool and submit with: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                })

    return messages, submitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--name", required=True, help="Name for this eval run")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--api-key", default="dummy", help="API key (use OPENROUTER_API_KEY env var or pass directly)")
    parser.add_argument("--gpu-slot", default="box5-gpu0", help="GPU slot for experiment dispatch")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of independent trials")
    parser.add_argument("--max-turns", type=int, default=5, help="Max agent turns per trial")
    parser.add_argument("--output-dir", default="/data/eval_baselines", help="Output directory")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter API (disables vLLM-specific features)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.name}.jsonl")

    from prompts import SYSTEM_PROMPT, build_instance_prompt
    train_py = Path(os.path.join(AUTORESEARCH_DIR, "train.py")).read_text()
    instance_prompt = build_instance_prompt(train_py, [])

    api_key = args.api_key if args.api_key != "dummy" else os.environ.get("OPENROUTER_API_KEY", "dummy")
    api_base = args.api_base
    if args.openrouter and api_base == "http://localhost:8000/v1":
        api_base = "https://openrouter.ai/api/v1"
    client = OpenAI(base_url=api_base, api_key=api_key)

    # Use the model name vLLM is serving (may differ from path)
    # Try to detect from /v1/models
    try:
        models = client.models.list()
        serving_model = models.data[0].id
        print(f"vLLM serving model: {serving_model}")
    except Exception:
        serving_model = args.model
        print(f"Could not detect model, using: {serving_model}")

    results = []
    for trial in range(args.n_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.n_trials} — {args.name}")
        print(f"{'='*60}")

        workdir = create_workdir()
        try:
            # Run agent
            t0 = time.time()
            messages, submitted = run_agent(client, serving_model, SYSTEM_PROMPT, instance_prompt,
                                            workdir, max_turns=args.max_turns,
                                            is_vllm=not args.openrouter)
            agent_time = time.time() - t0
            print(f"  Agent: {len(messages)} messages, submitted={submitted}, {agent_time:.1f}s")

            # Read modified train.py
            modified = Path(os.path.join(workdir, "train.py")).read_text()
            changed = modified != train_py

            # Diff
            diff = ""
            if changed:
                orig_path = os.path.join(workdir, "train.py.orig")
                Path(orig_path).write_text(train_py)
                r = subprocess.run(["diff", "-u", orig_path, os.path.join(workdir, "train.py")],
                                   capture_output=True, text=True)
                diff = r.stdout

            if not submitted:
                print(f"  WARNING: Agent did not submit after {args.max_turns} turns")

            if not changed:
                print(f"  WARNING: train.py was not modified")
                result = {"status": "no_changes", "val_bpb": None, "reward": 0.0}
            else:
                # Dispatch experiment
                print(f"  Dispatching experiment to {args.gpu_slot}...")
                t0 = time.time()
                result = run_experiment(modified, gpu_slot=args.gpu_slot)
                exp_time = time.time() - t0
                print(f"  Experiment: status={result['status']}, val_bpb={result.get('val_bpb')}, "
                      f"reward={result.get('reward', 0):.4f}, {exp_time:.1f}s")

            record = {
                "trial": trial,
                "name": args.name,
                "model": args.model,
                "submitted": submitted,
                "changed": changed,
                "diff": diff,
                "agent_time": agent_time,
                "num_messages": len(messages),
                **result,
            }
            results.append(record)

            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")

        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {args.name}")
    print(f"{'='*60}")
    val_bpbs = [r["val_bpb"] for r in results if r.get("val_bpb") is not None]
    rewards = [r.get("reward", 0) for r in results]
    crashes = sum(1 for r in results if r.get("status") == "crash")
    no_changes = sum(1 for r in results if r.get("status") == "no_changes")

    if val_bpbs:
        print(f"  val_bpb: min={min(val_bpbs):.6f}, mean={sum(val_bpbs)/len(val_bpbs):.6f}, max={max(val_bpbs):.6f}")
    print(f"  rewards: mean={sum(rewards)/len(rewards):.4f}")
    print(f"  crashes: {crashes}/{args.n_trials}, no_changes: {no_changes}/{args.n_trials}")
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
