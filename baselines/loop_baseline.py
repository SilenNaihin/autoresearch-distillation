"""
Baseline loop: single-shot sed or multi-turn agent editing.

Two modes:
  --mode sed    (default) One LLM call, parse sed commands from text, apply in Python.
  --mode agent  Multi-turn mini-swe-agent loop with bash tool calls.

ICL mode feeds back previous results; --single-turn has no feedback.
Supports vLLM (local/remote) and Bedrock (Claude) via litellm.

    python loop_baseline.py --max-turns 50 --mode sed --run-name my-run
    python loop_baseline.py --max-turns 50 --mode agent --step-limit 5 --run-name my-run
"""

import argparse
import json
import os
import re
import subprocess
import time
from difflib import unified_diff
from pathlib import Path

import wandb

import shutil

import litellm

from lib.bash_tool import create_isolated_workdir, run_agent_episode
from lib.environment import BASELINE_VAL_BPB, compute_reward, parse_metrics
from lib.experiment_cache import BASELINE_CACHE, ExperimentCache
from lib.runners import GPUSlot, SSHRunner


# ---------------------------------------------------------------------------
# Sed command parsing (shared logic with loop_power_sampling.py)
# ---------------------------------------------------------------------------

def parse_sed_commands(text: str) -> list[str]:
    """Extract unique sed commands from model output (preserves order)."""
    seen = set()
    commands = []
    for line in text.splitlines():
        line = line.strip()
        for suffix in ["<|im_end|>", "<|endoftext|>", "```"]:
            if line.endswith(suffix):
                line = line[: -len(suffix)].rstrip()
        if line.startswith("sed -i") and line not in seen:
            seen.add(line)
            commands.append(line)
    return commands


def _parse_sed_substitution(cmd: str) -> tuple[str, str, bool] | None:
    """Parse a sed substitution command into (pattern, replacement, global)."""
    m = re.match(r"sed\s+-i\s*(?:''|\"\")\s*'s(.)", cmd)
    if not m:
        m = re.match(r"sed\s+-i\s+'s(.)", cmd)
    if not m:
        return None
    delim = m.group(1)
    rest = cmd[m.end():]
    parts = []
    current = []
    escaped = False
    for ch in rest:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == '\\':
            current.append(ch)
            escaped = True
        elif ch == delim:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    if len(parts) < 2:
        return None
    pattern = parts[0]
    replacement = parts[1]
    flags_str = parts[2] if len(parts) > 2 else ""
    is_global = "g" in flags_str
    return pattern, replacement, is_global


def validate_sed_commands(baseline_py: str, commands: list[str]) -> list[str]:
    """Filter sed commands to only those whose patterns match the baseline."""
    valid = []
    for cmd in commands:
        if "train.py" not in cmd:
            continue
        parsed = _parse_sed_substitution(cmd)
        if parsed is None:
            continue
        pattern, _replacement, _is_global = parsed
        try:
            if re.search(pattern, baseline_py):
                valid.append(cmd)
            else:
                print(f"    SKIP (no match): {cmd}")
        except re.error:
            if pattern in baseline_py:
                valid.append(cmd)
            else:
                print(f"    SKIP (no match): {cmd}")
    return valid


def apply_sed_commands(baseline_py: str, commands: list[str]) -> str:
    """Apply sed substitution commands to train.py content using Python regex."""
    content = baseline_py
    for cmd in commands:
        if "train.py" not in cmd:
            continue
        parsed = _parse_sed_substitution(cmd)
        if parsed is None:
            continue
        pattern, replacement, is_global = parsed
        try:
            if is_global:
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content, count=1)
        except re.error:
            if is_global:
                content = content.replace(pattern, replacement)
            else:
                content = content.replace(pattern, replacement, 1)
    return content


def run_single_generation(
    baseline_py: str,
    model_name: str,
    system_prompt: str,
    instance_prompt: str,
    provider_kwargs: dict,
) -> tuple[str, list[dict], list[str]]:
    """Single-shot: one LLM call, parse sed commands from text, apply in Python.

    Returns (modified_train_py, trajectory, sed_commands).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]

    response = litellm.completion(
        model=model_name,
        messages=messages,
        **provider_kwargs,
    )

    choice = response.choices[0]
    content = choice.message.content or ""

    trajectory = [{"role": "assistant", "content": content}]

    # Parse and apply sed commands from the response text
    sed_commands = parse_sed_commands(content)
    sed_commands = validate_sed_commands(baseline_py, sed_commands)
    modified = apply_sed_commands(baseline_py, sed_commands)

    return modified, trajectory, sed_commands

# Baseline-specific system prompt (separate from SDPO's prompts.py).
BASELINE_SYSTEM_PROMPT = """\
You are an ML researcher optimizing a GPT pretraining script.

## Your task
Modify train.py to lower val_bpb (bits per byte on validation data).
The training budget is fixed at 5 minutes wall-clock. Lower val_bpb is better.

## Rules
- You may ONLY modify train.py. Do not touch prepare.py or any other file.
- You can only use packages already in pyproject.toml: torch, numpy, kernels, matplotlib, \
pandas, pyarrow, requests, rustbpe, tiktoken.
- Do not modify the evaluation harness. The evaluate_bpb function in prepare.py is the ground truth.
- VRAM is a soft constraint. Some increase is OK for meaningful val_bpb gains.

## Hardware constraints
- The experiment GPU has 93 GB VRAM (H100 SXM). The default config uses ~45 GB at depth=8.
- VRAM scales with model size. If you increase depth or width, you may need to reduce \
DEVICE_BATCH_SIZE to compensate. Check that your changes fit in ~85 GB.
- FlashAttention 3 (Hopper) requires head_dim to be a power of 2 (64, 128, 256). Other values crash.
- TOTAL_BATCH_SIZE must be divisible by (DEVICE_BATCH_SIZE * sequence_length). \
Violating this triggers an assertion error.

## Tips
- The training budget is fixed at 5 minutes. A configuration that processes more training steps \
in the same time may outperform a larger model that trains fewer steps.

## How to edit
The full train.py is provided below. Study the optimizer internals, model architecture, \
and training loop to find all the levers. Think step-by-step about what changes could lower val_bpb.
Then output your changes as sed commands, one per line. Do not rewrite the entire file. Example:

sed -i 's/learning_rate = 0.001/learning_rate = 0.0005/' train.py
sed -i 's/depth = 8/depth = 12/' train.py

## Important
- You are ONLY editing the file. You do NOT run experiments.
- The full train.py is already provided — do not ask to read it.
- Think carefully about what will improve training, then output correct sed commands.\
"""

AGENT_SYSTEM_PROMPT = """\
You are an autonomous ML researcher optimizing a GPT pretraining script.

## Your task
Modify train.py to lower val_bpb (bits per byte on validation data).
The training budget is fixed at 5 minutes wall-clock. Lower val_bpb is better.

## Rules
- You may ONLY modify train.py. Do not touch prepare.py or any other file.
- You can only use packages already in pyproject.toml: torch, numpy, kernels, matplotlib, \
pandas, pyarrow, requests, rustbpe, tiktoken.
- Do not modify the evaluation harness. The evaluate_bpb function in prepare.py is the ground truth.
- VRAM is a soft constraint. Some increase is OK for meaningful val_bpb gains.

## Hardware constraints
- The experiment GPU has 93 GB VRAM (H100 SXM). The default config uses ~45 GB at depth=8.
- VRAM scales with model size. If you increase depth or width, you may need to reduce \
DEVICE_BATCH_SIZE to compensate. Check that your changes fit in ~85 GB.
- FlashAttention 3 (Hopper) requires head_dim to be a power of 2 (64, 128, 256). Other values crash.
- TOTAL_BATCH_SIZE must be divisible by (DEVICE_BATCH_SIZE * sequence_length). \
Violating this triggers an assertion error.

## Tips
- The training budget is fixed at 5 minutes. A configuration that processes more training steps \
in the same time may outperform a larger model that trains fewer steps.

## Workflow
1. Read the full file — optimizer internals, model architecture, training loop — to find all \
the levers. Think step-by-step about what changes could lower val_bpb.
2. Make targeted edits to train.py using sed. Do not rewrite the entire file. Example:
   <tool_call>
   {"name": "bash", "arguments": {"command": "sed -i 's/OLD_VALUE/NEW_VALUE/' train.py"}}
   </tool_call>
3. When complete, submit: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

## Important
- You are ONLY editing the file. You do NOT run experiments.
- We will run uv run train.py on a GPU after you submit.
- Think carefully about what will improve training, then make correct edits.\
"""

# Experiments run on h100-dev-box-2 via self-SSH (localhost).
# Inference stays on A100 (a100-backup-1) via remote vLLM.
EXPERIMENT_FLEET = [GPUSlot("localhost", "0", "box2-gpu0", "~/autoresearch")]

MODEL = "Qwen/Qwen3-14B-SDPO"
VLLM_BASE_URL = "http://20.125.45.203:8000/v1"
OUTPUT_DIR = "data/results"

# How many top results to show with full diffs in the feedback prompt
TOP_K_FULL_DIFFS = 3


def build_instance_prompt(train_py_content: str, history_lines: list[str]) -> str:
    """Build the task prompt with current train.py."""
    parts = ["## Current train.py\n```python\n" + train_py_content + "\n```"]

    parts.append(
        "You may make a single focused change or combine related changes if they work together. "
        "Feel free to try completely new approaches and explore new spaces every once in a while. "
        "Be reasonable on effort towards pushing and tweaking a result vs trying something different. "
        "You must apply your changes directly using the bash tool — do not just output a diff."
    )
    return "\n\n".join(parts)


def make_diff(baseline: str, modified: str) -> str:
    """Full unified diff — no truncation."""
    if baseline == modified:
        return "(no changes)"
    return "".join(unified_diff(
        baseline.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="a/train.py", tofile="b/train.py",
    ))


def format_feedback_prompt(turn_results: list[dict]) -> str:
    """Format accumulated results into a structured feedback block."""
    if not turn_results:
        return ""

    successful = [r for r in turn_results if r.get("val_bpb") is not None]
    failed = [r for r in turn_results if r.get("val_bpb") is None]

    successful.sort(key=lambda r: r["val_bpb"])

    parts = []

    # Summary line
    if successful:
        best = successful[0]["val_bpb"]
        if best < BASELINE_VAL_BPB:
            parts.append(
                f"**The current train.py achieves val_bpb={BASELINE_VAL_BPB}. "
                f"Your best so far: val_bpb={best:.4f}. "
                f"Make changes to continue to push it even lower than your previous changes.**"
            )
        else:
            parts.append(
                f"**The current train.py achieves val_bpb={BASELINE_VAL_BPB}. "
                f"Make changes to train.py to push this as low as you can.**"
            )

    # Stuck detection: if last 5 successful results are within 0.005 of each other
    recent_vals = [r["val_bpb"] for r in turn_results[-5:] if r.get("val_bpb") is not None]
    if len(recent_vals) >= 4 and (max(recent_vals) - min(recent_vals)) < 0.005:
        parts.append(
            "**Your last several results are all very similar. "
            "Try something completely different — a new approach you haven't explored yet.**"
        )

    # Top-K best results with full diffs
    top_k = successful[:TOP_K_FULL_DIFFS]
    if top_k:
        parts.append("## Best results so far (full diffs)")
        for i, r in enumerate(top_k):
            rank = i + 1
            marker = " *** BEST ***" if rank == 1 else ""
            parts.append(
                f"### #{rank} — val_bpb={r['val_bpb']:.6f} (attempt {r['turn']+1}){marker}\n"
                f"```diff\n{r['diff']}\n```"
            )

    # Remaining successful runs as one-line summaries
    rest = successful[TOP_K_FULL_DIFFS:]
    if rest:
        parts.append("## Other successful attempts")
        for r in rest:
            parts.append(
                f"- Attempt {r['turn']+1}: val_bpb={r['val_bpb']:.6f} "
                f"(depth={r.get('depth', '?')}, tokens={r.get('tokens_M', '?')}M)"
            )

    # Failed attempts — reframed as debugging opportunities
    if failed:
        parts.append(
            "## Failed attempts\n"
            "These crashed. Don't repeat the exact same change, but consider "
            "if the core idea could work with adjustments (e.g., reduce batch size "
            "to fit a larger model in VRAM)."
        )
        for r in failed:
            status = r.get("status", "crash")
            reason = r.get("crash_reason", "unknown")
            diff = r.get("diff", "")
            if diff and diff != "(no changes)":
                parts.append(
                    f"### Attempt {r['turn']+1}: {status} — {reason}\n"
                    f"```diff\n{diff}\n```"
                )
            else:
                parts.append(f"- Attempt {r['turn']+1}: {status} — {reason}")

    parts.append(
        "\nLearn from these results. "
        "Build on the best approaches above to further lower val_bpb."
    )

    return "\n\n".join(parts)


def classify_crash(error_text: str) -> str:
    """Extract a short crash reason from error output, including VRAM usage for OOM."""
    if not error_text:
        return "unknown error"
    lower = error_text.lower()
    if "out of memory" in lower or "oom" in lower:
        # Extract VRAM usage from PyTorch OOM message
        m = re.search(r"(\d+\.\d+)\s*GiB\s*memory\s*in\s*use", error_text)
        if m:
            return f"OOM — used {m.group(1)} GiB of 93 GiB available"
        return "OOM (out of VRAM)"
    if "syntaxerror" in lower:
        return "SyntaxError in generated code"
    if "assertionerror" in lower or "assert " in lower:
        return "assertion failed (batch size / config mismatch)"
    if "flash_attn" in lower or "flashattention" in lower:
        return "FlashAttention error (likely unsupported head_dim)"
    if "importerror" in lower or "modulenotfounderror" in lower:
        return "import error (unavailable package)"
    lines = [l.strip() for l in error_text.strip().splitlines() if l.strip()]
    if lines:
        return lines[-1][:120]
    return "unknown error"



def count_sed_commands(trajectory: list[dict]) -> int:
    """Count how many sed commands the model issued."""
    count = 0
    for msg in trajectory:
        cmd = msg.get("command", "")
        if isinstance(cmd, str) and cmd.strip().startswith("sed"):
            count += 1
    # Fallback: count in raw string if structured parsing missed them
    if count == 0:
        traj_str = str(trajectory)
        count = traj_str.count("sed -i") + traj_str.count("sed -e")
    return count


def extract_model_thinking(trajectory: list[dict], max_len: int = 1000) -> str:
    """Extract the <think> block from the assistant message."""
    for msg in trajectory:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                start = content.find("<think>")
                end = content.find("</think>")
                if start >= 0 and end > start:
                    think = content[start + len("<think>"):end].strip()
                    if len(think) > max_len:
                        think = think[:max_len] + "..."
                    return think
    return ""


def extract_changed_variables(diff_text: str) -> list[str]:
    """Extract variable names that were changed from a unified diff."""
    changed = []
    # Common config variables in train.py
    known_vars = [
        "DEVICE_BATCH_SIZE", "TOTAL_BATCH_SIZE", "MATRIX_LR", "SCALAR_LR",
        "depth", "width", "num_heads", "head_dim", "ASPECT_RATIO",
        "WINDOW_PATTERN", "sequence_length", "warmup_steps", "cooldown_fraction",
        "muon_momentum", "weight_decay", "VOCAB_SIZE", "NUM_LAYERS",
    ]
    for var in known_vars:
        if var in diff_text:
            changed.append(var)
    return changed


def dispatch_experiment(train_py: str) -> tuple[dict, int, str, str]:
    """Dispatch modified train.py to experiment GPU.

    Returns (metrics_dict, returncode, feedback_text, raw_stdout).
    """
    for slot in EXPERIMENT_FLEET:
        runner = SSHRunner(slot, timeout=600)
        output = runner.run(train_py)
        if output.returncode == 255:
            print(f"  [baseline] {slot.name} SSH failed, trying next...")
            continue

        metrics = parse_metrics(output.stdout) if output.stdout else {}
        val_bpb = metrics.get("val_bpb")
        raw_stdout = output.stdout or ""

        if output.returncode != 0:
            crash_tail = (output.stderr or output.stdout or "no output").strip()[-500:]
            return metrics, output.returncode, f"Experiment crashed (exit {output.returncode}):\n{crash_tail}", raw_stdout

        if val_bpb is None:
            tail = "\n".join((output.stdout or "").strip().splitlines()[-15:])
            return metrics, 0, f"Experiment ran but produced no val_bpb. Output tail:\n{tail}", raw_stdout

        return metrics, 0, "", raw_stdout

    return {}, -1, "All experiment GPUs unreachable.", ""


def log_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def log_trace(output_dir: Path, run_name: str, turn: int, prompt: str,
              trajectory: list[dict], diff: str = "", modified_train_py: str = "",
              experiment_stdout: str = ""):
    """Save full rollout: prompt, trajectory, diff, modified code, experiment output."""
    trace_dir = output_dir / f"{run_name}_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"turn_{turn:03d}.json"
    trace_path.write_text(json.dumps({
        "turn": turn,
        "prompt": prompt,
        "trajectory": trajectory,
        "diff": diff,
        "modified_train_py": modified_train_py,
        "experiment_stdout": experiment_stdout,
    }, ensure_ascii=True, indent=2))


def sync_cache(host: str, cache_path: Path, direction: str = "pull"):
    """Sync cache file to/from a remote host via scp, merging entries.

    Pull downloads to a temp file and merges into local (no overwrites).
    Push sends local file to remote.
    """
    remote = f"{host}:{cache_path}"
    local = str(cache_path)
    try:
        if direction == "pull":
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(cache_path) + ".remote"
            r = subprocess.run(["scp", "-o", "ConnectTimeout=5", "-q", remote, tmp],
                               capture_output=True, timeout=15)
            if r.returncode == 0 and os.path.exists(tmp):
                # Merge remote entries into local file
                try:
                    with open(tmp) as f:
                        remote_data = json.load(f)
                    local_data = {"diffs": {}, "best_val_bpb": 1e9, "best_diff": ""}
                    if cache_path.exists():
                        with open(cache_path) as f:
                            local_data = json.load(f)
                    if "diffs" in remote_data:
                        local_data.setdefault("diffs", {}).update(remote_data["diffs"])
                    rb = remote_data.get("best_val_bpb", 1e9)
                    if rb < local_data.get("best_val_bpb", 1e9):
                        local_data["best_val_bpb"] = rb
                        local_data["best_diff"] = remote_data.get("best_diff", "")
                    with open(cache_path, "w") as f:
                        json.dump(local_data, f, ensure_ascii=True)
                except (json.JSONDecodeError, OSError):
                    pass
                finally:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        else:
            subprocess.run(["ssh", "-o", "ConnectTimeout=5", host,
                           f"mkdir -p {cache_path.parent}"],
                           capture_output=True, timeout=10)
            subprocess.run(["scp", "-o", "ConnectTimeout=5", "-q", local, remote],
                           capture_output=True, timeout=15)
    except Exception:
        pass  # cache sync is best-effort


def main():
    parser = argparse.ArgumentParser(description="Multi-turn baseline (in-context learning)")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vllm-base-url", type=str, default=VLLM_BASE_URL)
    parser.add_argument("--run-name", type=str, default="qwen3-14b-baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single-turn", action="store_true",
                        help="No feedback between turns — measures baseline sampling entropy")
    parser.add_argument("--experiment-host", type=str, default=None,
                        help="SSH host for experiment GPU (default: localhost)")
    parser.add_argument("--experiment-gpu", type=str, default="0",
                        help="CUDA device index on experiment host")
    parser.add_argument("--cache-sync-host", type=str, default=None,
                        help="SSH host to sync shared cache to/from (e.g. 74.179.57.153)")
    parser.add_argument("--provider", type=str, default="vllm",
                        choices=["vllm", "bedrock"],
                        help="LLM provider: vllm (local/remote vLLM) or bedrock (AWS)")
    parser.add_argument("--mode", type=str, default="sed",
                        choices=["sed", "agent"],
                        help="sed: single-shot text parsing; agent: mini-swe-agent loop")
    parser.add_argument("--step-limit", type=int, default=5,
                        help="Max bash tool calls per episode (agent mode only)")
    parser.add_argument("--verl-compat", action="store_true",
                        help="Treat no-tool-call as submission (agent mode, for VERL-trained models)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous JSONL file (loads turn_results and best_val_bpb)")
    args = parser.parse_args()

    # Override experiment fleet if --experiment-host is set
    global EXPERIMENT_FLEET
    if args.experiment_host:
        host = args.experiment_host
        gpu = args.experiment_gpu
        EXPERIMENT_FLEET = [GPUSlot(host, gpu, f"{host}-gpu{gpu}", "~/autoresearch")]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / f"{args.run_name}-{int(time.time())}.jsonl"

    baseline = Path("autoresearch/train.py").read_text()

    # Build litellm model name and provider kwargs
    if args.provider == "bedrock":
        litellm_model = f"bedrock/{args.model}"
        provider_kwargs = {"temperature": args.temperature}
    else:
        litellm_model = f"openai/{args.model}"
        provider_kwargs = {
            "api_base": args.vllm_base_url,
            "api_key": "dummy",
            "temperature": args.temperature,
        }

    # Create mini-swe-agent model for agent mode
    swe_model = None
    if args.mode == "agent":
        from minisweagent.models.litellm_model import LitellmModel
        if args.provider == "bedrock":
            swe_model = LitellmModel(
                model_name=f"bedrock/{args.model}",
                model_kwargs={"temperature": args.temperature},
                cost_tracking="ignore_errors",
            )
        else:
            swe_model = LitellmModel(
                model_name=f"openai/{args.model}",
                model_kwargs={
                    "api_base": args.vllm_base_url,
                    "api_key": "dummy",
                    "temperature": args.temperature,
                },
                cost_tracking="ignore_errors",
            )

    print(f"Model:       {litellm_model}")
    print(f"Mode:        {args.mode}")
    print(f"Max turns:   {args.max_turns}")
    print(f"Temperature: {args.temperature}")
    print(f"Single-turn: {args.single_turn}")
    if args.mode == "agent":
        print(f"Step limit:  {args.step_limit}")
        print(f"VERL compat: {args.verl_compat}")
    print(f"Fleet:       {[s.name for s in EXPERIMENT_FLEET]}")
    print(f"Output:      {output_path}")
    print()

    # Init wandb
    config = {
        "model": args.model,
        "mode": args.mode,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "single_turn": args.single_turn,
        "seed": args.seed,
        "fleet": [s.name for s in EXPERIMENT_FLEET],
        "baseline_val_bpb": BASELINE_VAL_BPB,
        "method": f"{args.mode} / {'single-turn' if args.single_turn else 'ICL'}",
    }
    wandb.init(project="autoresearch-baseline", name=args.run_name, config=config)
    log_jsonl(output_path, {"config": config})

    # Wandb table for per-turn details (diffs, thinking, crash info)
    turns_table = wandb.Table(columns=[
        "turn", "status", "val_bpb", "reward", "diff", "crash_reason",
        "model_thinking", "used_search", "num_sed_commands", "changed_variables",
        "agent_time", "experiment_time", "prompt_len", "trajectory_len",
    ])

    # Multi-turn loop
    best_val_bpb = float("inf")
    turn_results: list[dict] = []
    start_turn = 0

    # Resume from previous JSONL if provided
    if args.resume:
        with open(args.resume) as f:
            for line in f:
                entry = json.loads(line)
                if "config" in entry:
                    continue
                # Skip errors/generation_errors — only keep real results
                status = entry.get("status", "")
                if status in ("generation_error", "agent_error"):
                    continue
                turn_results.append({
                    "turn": entry.get("turn", len(turn_results)),
                    "val_bpb": entry.get("val_bpb"),
                    "diff": entry.get("diff", ""),
                    "status": status,
                    "crash_reason": entry.get("crash_reason", ""),
                    "depth": entry.get("depth"),
                    "tokens_M": entry.get("tokens_M"),
                    "memory_gb": entry.get("memory_gb"),
                })
                v = entry.get("val_bpb")
                if v is not None and v < best_val_bpb:
                    best_val_bpb = v
        start_turn = len(turn_results)
        print(f"Resumed from {args.resume}: {start_turn} turns, best={best_val_bpb}")

    cache_sync_host = args.cache_sync_host
    if cache_sync_host:
        sync_cache(cache_sync_host, BASELINE_CACHE, "pull")
    cache = ExperimentCache(write_path=BASELINE_CACHE)
    print(f"Experiment cache: {len(cache)} entries loaded from disk")
    cumulative_crashes = 0
    cumulative_successes = 0

    for turn in range(start_turn, start_turn + args.max_turns):
        t0 = time.time()
        # Sync cache from remote before checking for duplicates
        if cache_sync_host:
            sync_cache(cache_sync_host, BASELINE_CACHE, "pull")
            cache = ExperimentCache(write_path=BASELINE_CACHE)
        print(f"\n{'='*60}")
        print(f"Turn {turn}/{args.max_turns}")
        print(f"{'='*60}")

        # Build instance prompt (with feedback for ICL, without for single-turn)
        instance_prompt = build_instance_prompt(baseline, [])
        if not args.single_turn:
            feedback_block = format_feedback_prompt(turn_results)
            if feedback_block:
                instance_prompt += "\n\n" + feedback_block

        # Generate edits: sed mode (single-shot text) or agent mode (mini-swe-agent)
        agent_t0 = time.time()
        try:
            if args.mode == "agent":
                sys_prompt = AGENT_SYSTEM_PROMPT
                workdir = create_isolated_workdir()
                try:
                    modified, trajectory = run_agent_episode(
                        workdir=workdir,
                        model=swe_model,
                        system_prompt=sys_prompt,
                        instance_prompt=instance_prompt,
                        step_limit=args.step_limit,
                        verl_compat=args.verl_compat,
                    )
                finally:
                    shutil.rmtree(workdir, ignore_errors=True)
                sed_commands = []  # not applicable for agent mode
            else:
                modified, trajectory, sed_commands = run_single_generation(
                    baseline_py=baseline,
                    model_name=litellm_model,
                    system_prompt=BASELINE_SYSTEM_PROMPT,
                    instance_prompt=instance_prompt,
                    provider_kwargs=provider_kwargs,
                )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  Generation failed: {e}")
            print(f"  Traceback: {tb}")
            cumulative_crashes += 1
            turn_results.append({
                "turn": turn,
                "val_bpb": None,
                "diff": "",
                "status": "generation_error",
                "crash_reason": str(e)[:200],
            })
            wandb.log({"turn": turn, "status": "generation_error", "reward": 0.0,
                        "prompt_len": len(instance_prompt),
                        "cumulative_crashes": cumulative_crashes,
                        "cumulative_successes": cumulative_successes})
            log_jsonl(output_path, {
                "turn": turn, "status": "generation_error",
                "error": str(e), "traceback": tb,
                "experiment_time": time.time() - t0,
                "prompt_len": len(instance_prompt),
            })
            continue

        agent_time = time.time() - agent_t0
        experiment_stdout = ""  # populated after dispatch

        diff_text = make_diff(baseline, modified)
        trajectory_len = len(trajectory)
        diff_size = len(diff_text)
        model_thinking = extract_model_thinking(trajectory)

        if args.mode == "agent":
            n_tool_calls = sum(1 for m in trajectory if m.get("role") == "assistant"
                              and "tool_calls" in str(m.get("content", "")))
            used_search = any(
                "search.py" in str(a.get("command", ""))
                for m in trajectory
                for a in (m.get("extra", {}) or {}).get("actions", [])
                if isinstance(a, dict)
            )
            num_sed = count_sed_commands(trajectory)
        else:
            n_tool_calls = len(sed_commands)
            used_search = False
            num_sed = len(sed_commands)

        print(f"  Tool calls: {n_tool_calls}  Sed: {num_sed}")
        print(f"  Diff size:  {diff_size} chars")
        print(f"  Gen time:   {agent_time:.1f}s  Trajectory: {trajectory_len} msgs")

        if baseline == modified:
            print(f"  Status: no_changes")
            cumulative_crashes += 1
            turn_results.append({
                "turn": turn,
                "val_bpb": None,
                "diff": "",
                "status": "no_changes",
                "crash_reason": "no changes made to train.py",
            })
            turns_table.add_data(
                turn, "no_changes", None, 0.0, "", "no changes made",
                model_thinking, used_search, num_sed, "",
                agent_time, 0.0, len(instance_prompt), trajectory_len,
            )
            wandb.log({"turn": turn, "status": "no_changes", "reward": 0.0,
                        "tool_calls": n_tool_calls, "prompt_len": len(instance_prompt),
                        "trajectory_len": trajectory_len, "used_search": used_search,
                        "agent_time": agent_time, "diff_size": 0,
                        "cumulative_crashes": cumulative_crashes,
                        "cumulative_successes": cumulative_successes})
            log_jsonl(output_path, {"turn": turn, "status": "no_changes",
                                    "tool_calls": n_tool_calls,
                                    "prompt_len": len(instance_prompt),
                                    "trajectory_len": trajectory_len,
                                    "used_search": used_search,
                                    "agent_time": agent_time})
            log_trace(Path(OUTPUT_DIR), args.run_name, turn, instance_prompt, trajectory,
                      diff=diff_text, modified_train_py=modified)
            continue

        # Cache: replay cached result if we've seen this exact diff before
        cached = cache.get(diff_text)
        if cached is not None:
            c_status = cached.get("status", "duplicate")
            c_vbpb = cached.get("val_bpb")
            print(f"  Status: cached ({c_status}, val_bpb={c_vbpb})")
            turn_results.append({
                "turn": turn,
                "val_bpb": c_vbpb,
                "diff": diff_text,
                "status": "cached_" + c_status,
                "crash_reason": cached.get("crash_reason", "exact same changes already tried"),
                "depth": cached.get("depth"),
                "tokens_M": cached.get("tokens_M"),
                "memory_gb": cached.get("memory_gb"),
            })
            turns_table.add_data(
                turn, "cached_" + c_status, c_vbpb, 0.0, diff_text, "cached result",
                model_thinking, used_search, num_sed, ", ".join(extract_changed_variables(diff_text)),
                agent_time, 0.0, len(instance_prompt), trajectory_len,
            )
            wandb.log({"turn": turn, "status": "cached_" + c_status, "reward": 0.0,
                        "val_bpb": c_vbpb,
                        "tool_calls": n_tool_calls, "prompt_len": len(instance_prompt),
                        "trajectory_len": trajectory_len, "used_search": used_search,
                        "agent_time": agent_time, "diff_size": diff_size,
                        "is_duplicate": True, "is_novel": False,
                        "cumulative_crashes": cumulative_crashes,
                        "cumulative_successes": cumulative_successes})
            log_jsonl(output_path, {"turn": turn, "status": "cached_" + c_status,
                                    "val_bpb": c_vbpb,
                                    "diff": diff_text, "tool_calls": n_tool_calls,
                                    "prompt_len": len(instance_prompt),
                                    "trajectory_len": trajectory_len,
                                    "used_search": used_search,
                                    "agent_time": agent_time})
            log_trace(Path(OUTPUT_DIR), args.run_name, turn, instance_prompt, trajectory,
                      diff=diff_text, modified_train_py=modified)
            continue

        # Dispatch to experiment GPU
        print(f"  Dispatching experiment to {EXPERIMENT_FLEET[0].name}...")
        metrics, returncode, error_feedback, experiment_stdout = dispatch_experiment(modified)
        val_bpb = metrics.get("val_bpb")
        experiment_time = time.time() - t0

        if returncode != 0 or (returncode == 0 and val_bpb is None and error_feedback):
            status = "crash" if returncode != 0 else "no_metric"
            crash_reason = classify_crash(error_feedback)
            print(f"  Status: {status} — {crash_reason}")
            turn_results.append({
                "turn": turn,
                "val_bpb": None,
                "diff": diff_text,
                "status": status,
                "crash_reason": crash_reason,
            })
            # Cache OOM crashes (deterministic), but not transient crashes
            if "OOM" in crash_reason or "out of memory" in crash_reason.lower():
                cache.put(diff_text, {
                    "status": "crash", "val_bpb": None,
                    "crash_reason": crash_reason,
                })
                if cache_sync_host:
                    sync_cache(cache_sync_host, BASELINE_CACHE, "push")
            cumulative_crashes += 1
            turns_table.add_data(
                turn, status, None, 0.0, diff_text, crash_reason,
                model_thinking, used_search, num_sed, ", ".join(extract_changed_variables(diff_text)),
                agent_time, experiment_time, len(instance_prompt), trajectory_len,
            )
            wandb.log({"turn": turn, "status": status, "reward": 0.0,
                        "experiment_time": experiment_time, "tool_calls": n_tool_calls,
                        "prompt_len": len(instance_prompt),
                        "trajectory_len": trajectory_len, "used_search": used_search,
                        "agent_time": agent_time, "diff_size": diff_size,
                        "is_duplicate": False, "is_novel": True,
                        "cumulative_crashes": cumulative_crashes,
                        "cumulative_successes": cumulative_successes})
            log_jsonl(output_path, {"turn": turn, "status": status,
                                    "diff": diff_text, "feedback": error_feedback,
                                    "experiment_time": experiment_time,
                                    "prompt_len": len(instance_prompt),
                                    "trajectory_len": trajectory_len,
                                    "used_search": used_search,
                                    "agent_time": agent_time})
            log_trace(Path(OUTPUT_DIR), args.run_name, turn, instance_prompt, trajectory,
                      diff=diff_text, modified_train_py=modified, experiment_stdout=experiment_stdout)
            continue

        # Compute reward
        reward, status, reward_feedback = compute_reward(val_bpb, best_val_bpb)
        if status == "improvement":
            best_val_bpb = val_bpb

        memory_gb = metrics.get("peak_vram_mb", 0) / 1024
        depth = metrics.get("depth")
        tokens_M = metrics.get("total_tokens_M")

        turn_results.append({
            "turn": turn,
            "val_bpb": val_bpb,
            "diff": diff_text,
            "status": status,
            "depth": int(depth) if depth else None,
            "tokens_M": int(tokens_M) if tokens_M else None,
            "memory_gb": round(memory_gb, 1),
        })

        # Cache successful result
        cache.put(diff_text, {
            "status": status, "val_bpb": val_bpb,
            "depth": int(depth) if depth else None,
            "tokens_M": int(tokens_M) if tokens_M else None,
            "memory_gb": round(memory_gb, 1),
        }, val_bpb=val_bpb, diff_text_raw=diff_text)
        if cache_sync_host:
            sync_cache(cache_sync_host, BASELINE_CACHE, "push")

        cumulative_successes += 1
        turns_table.add_data(
            turn, status, val_bpb, reward, diff_text, "",
            model_thinking, used_search, num_sed, ", ".join(extract_changed_variables(diff_text)),
            agent_time, experiment_time, len(instance_prompt), trajectory_len,
        )
        print(f"  val_bpb={val_bpb:.6f}  best={best_val_bpb:.6f}  "
              f"reward={reward:.4f}  status={status}  time={experiment_time:.0f}s")

        wandb.log({
            "turn": turn, "val_bpb": val_bpb, "best_val_bpb": best_val_bpb,
            "reward": reward, "memory_gb": memory_gb,
            "experiment_time": experiment_time, "status": status,
            "tool_calls": n_tool_calls, "prompt_len": len(instance_prompt),
            "trajectory_len": trajectory_len, "used_search": used_search,
            "agent_time": agent_time, "diff_size": diff_size,
            "is_duplicate": False, "is_novel": True,
            "cumulative_crashes": cumulative_crashes,
            "cumulative_successes": cumulative_successes,
            **{f"env/{k}": v for k, v in metrics.items()},
        })
        log_jsonl(output_path, {
            "turn": turn, "val_bpb": val_bpb, "best_val_bpb": best_val_bpb,
            "reward": reward, "status": status, "memory_gb": memory_gb,
            "experiment_time": experiment_time, "diff": diff_text,
            "metrics": metrics, "tool_calls": n_tool_calls,
            "prompt_len": len(instance_prompt),
            "trajectory_len": trajectory_len, "used_search": used_search,
            "agent_time": agent_time,
        })
        log_trace(Path(OUTPUT_DIR), args.run_name, turn, instance_prompt, trajectory,
                  diff=diff_text, modified_train_py=modified, experiment_stdout=experiment_stdout)

    # Summary
    print(f"\n{'='*60}")
    print(f"Baseline complete: {args.max_turns} turns")
    print(f"Best val_bpb: {best_val_bpb:.6f} (baseline: {BASELINE_VAL_BPB})")
    print(f"Output: {output_path}")

    wandb.log({
        "final/best_val_bpb": best_val_bpb,
        "final/improvement": max(0, BASELINE_VAL_BPB - best_val_bpb),
        "turns_table": turns_table,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
