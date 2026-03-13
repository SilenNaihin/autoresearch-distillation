"""
Power sampling baseline: single-shot generation with MCMC refinement.

Same task as loop_baseline.py (modify train.py to lower val_bpb), but instead
of multi-turn agent editing, the model generates sed commands in a single shot.
Power sampling (MCMC from p^α) produces higher-likelihood outputs that should
correlate with better reasoning.

Ablation comparison:
    loop_baseline.py         — multi-turn agent, vanilla sampling
    loop_power_sampling.py   — single-shot, MCMC power sampling (this file)

Usage:
    python loop_power_sampling.py --max-turns 30 --method power --alpha 4.0
    python loop_power_sampling.py --max-turns 30 --method best-of-n --n 8
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import wandb

from environment import BASELINE_VAL_BPB, compute_reward, parse_metrics
from loop_baseline import (
    EXPERIMENT_FLEET,
    MODEL,
    OUTPUT_DIR,
    TOP_K_FULL_DIFFS,
    VLLM_BASE_URL,
    classify_crash,
    dispatch_experiment,
    format_feedback_prompt,
    log_jsonl,
    log_trace,
    make_diff,
)
from power_sampling import PowerSampleResult, best_of_n, power_sample

# ---------------------------------------------------------------------------
# Prompt for single-shot generation (no tool calling)
# ---------------------------------------------------------------------------

POWER_SYSTEM_PROMPT = """\
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

## Output format
Think step-by-step about what changes could lower val_bpb.
Then output your changes as sed commands, one per line.
Each sed command must start with "sed -i" and edit train.py.

Example output:
I'll increase model depth and adjust learning rate for the larger model.

sed -i 's/NUM_LAYERS = 8/NUM_LAYERS = 12/' train.py
sed -i 's/learning_rate = 0.001/learning_rate = 0.0005/' train.py\
"""


def build_power_prompt(train_py: str, feedback_block: str) -> str:
    """Build the user prompt for power sampling (single-shot)."""
    parts = ["## Current train.py\n```python\n" + train_py + "\n```"]

    if feedback_block:
        parts.append(feedback_block)

    parts.append(
        "Make a single focused change or combine related changes. "
        "Feel free to try completely new approaches and explore new spaces. "
        "Output your reasoning followed by sed commands."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Sed parsing and application
# ---------------------------------------------------------------------------


def parse_sed_commands(text: str) -> list[str]:
    """Extract unique sed commands from model output (preserves order)."""
    seen = set()
    commands = []
    for line in text.splitlines():
        line = line.strip()
        # Strip trailing special tokens / markdown artifacts
        for suffix in ["<|im_end|>", "<|endoftext|>", "```"]:
            if line.endswith(suffix):
                line = line[: -len(suffix)].rstrip()
        if line.startswith("sed -i") and line not in seen:
            seen.add(line)
            commands.append(line)
    return commands


def _parse_sed_substitution(cmd: str) -> tuple[str, str, bool] | None:
    """Parse a sed substitution command into (pattern, replacement, global).

    Handles: sed -i 's/PAT/REPL/' file  and  sed -i 's/PAT/REPL/g' file
    Uses the first non-alphanumeric char after 's' as delimiter.
    """
    # Strip 'sed -i' prefix (with optional '' for macOS compat)
    m = re.match(r"sed\s+-i\s*(?:''|\"\")\s*'s(.)", cmd)
    if not m:
        m = re.match(r"sed\s+-i\s+'s(.)", cmd)
    if not m:
        return None

    delim = m.group(1)
    rest = cmd[m.end():]  # everything after the delimiter

    # Split by delimiter, handling escaped delimiters
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
    """Apply sed substitution commands to train.py content using Python regex.

    Portable (no platform-specific sed) and safe (no shell execution).
    """
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
            # If regex fails, try literal string replacement
            if is_global:
                content = content.replace(pattern, replacement)
            else:
                content = content.replace(pattern, replacement, 1)
    return content


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Power sampling baseline")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--vllm-base-url", type=str, default=VLLM_BASE_URL)
    parser.add_argument("--run-name", type=str, default="qwen3-14b-power-sampling")
    parser.add_argument("--seed", type=int, default=0)

    # Sampling method
    parser.add_argument("--method", type=str, default="power",
                        choices=["power", "best-of-n"],
                        help="Sampling method: 'power' (MCMC) or 'best-of-n'")
    # Power sampling params
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="Power exponent α (higher = sharper)")
    parser.add_argument("--block-num", type=int, default=16,
                        help="Number of MCMC blocks")
    parser.add_argument("--mcmc-steps", type=int, default=10,
                        help="MCMC refinement steps per block")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Proposal temperature (1.0 for exact MH ratio)")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max response tokens (match 64k context window, model EOS's naturally)")
    # Best-of-N params
    parser.add_argument("--n", type=int, default=8,
                        help="Number of candidates for best-of-n")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / f"{args.run_name}-{int(time.time())}.jsonl"

    baseline = Path("autoresearch/train.py").read_text()

    from openai import OpenAI
    client = OpenAI(base_url=args.vllm_base_url, api_key="dummy")

    print(f"Model:       {args.model}")
    print(f"vLLM:        {args.vllm_base_url}")
    print(f"Method:      {args.method}")
    if args.method == "power":
        print(f"  alpha:       {args.alpha}")
        print(f"  block_num:   {args.block_num}")
        print(f"  mcmc_steps:  {args.mcmc_steps}")
        print(f"  temperature: {args.temperature}")
    else:
        print(f"  n:           {args.n}")
    print(f"Max turns:   {args.max_turns}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Fleet:       {[s.name for s in EXPERIMENT_FLEET]}")
    print(f"Output:      {output_path}")
    print()

    config = {
        "model": args.model,
        "max_turns": args.max_turns,
        "method": args.method,
        "alpha": args.alpha,
        "block_num": args.block_num,
        "mcmc_steps": args.mcmc_steps,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "seed": args.seed,
        "fleet": [s.name for s in EXPERIMENT_FLEET],
        "baseline_val_bpb": BASELINE_VAL_BPB,
    }
    wandb.init(project="autoresearch-baseline", name=args.run_name, config=config)
    log_jsonl(output_path, {"config": config})

    best_val_bpb = float("inf")
    turn_results: list[dict] = []
    best_trajectory_summary: str = ""
    previous_trajectory_summary: str = ""

    for turn in range(args.max_turns):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Turn {turn}/{args.max_turns}")
        print(f"{'='*60}")

        # Build prompt
        feedback_block = format_feedback_prompt(
            turn_results, best_trajectory_summary, previous_trajectory_summary)
        user_prompt = build_power_prompt(baseline, feedback_block)

        # Generate with power sampling or best-of-n
        print(f"  Generating ({args.method})...")
        if args.method == "power":
            result = power_sample(
                client=client,
                model=args.model,
                system_prompt=POWER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                alpha=args.alpha,
                max_tokens=args.max_tokens,
                block_num=args.block_num,
                mcmc_steps=args.mcmc_steps,
                temperature=args.temperature,
            )
        else:
            result = best_of_n(
                client=client,
                model=args.model,
                system_prompt=POWER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                n=args.n,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

        gen_time = time.time() - t0
        print(f"  Generation: {len(result.tokens)} tokens, "
              f"{result.total_tokens_generated} total generated, "
              f"accept={result.acceptance_rate:.2f}, {gen_time:.1f}s")

        if not result.text.strip():
            print("  Status: empty_generation")
            turn_results.append({
                "turn": turn, "val_bpb": None, "diff": "",
                "status": "empty_generation",
                "crash_reason": "model produced empty output",
            })
            wandb.log({"turn": turn, "status": "empty_generation", "reward": 0.0})
            log_jsonl(output_path, {"turn": turn, "status": "empty_generation",
                                    "gen_time": gen_time})
            continue

        # Extract reasoning (everything before the first sed command)
        reasoning = result.text
        first_sed = reasoning.find("sed -i")
        if first_sed > 0:
            reasoning = reasoning[:first_sed].strip()
        previous_trajectory_summary = reasoning

        # Log the full generation as trace
        log_trace(Path(OUTPUT_DIR), args.run_name, turn, user_prompt,
                  [{"role": "assistant", "content": result.text}])

        # Parse and validate sed commands against baseline
        sed_commands = parse_sed_commands(result.text)
        print(f"  Sed commands found: {len(sed_commands)}")
        sed_commands = validate_sed_commands(baseline, sed_commands)
        print(f"  Sed commands valid: {len(sed_commands)}")

        if not sed_commands:
            print("  Status: no_sed_commands")
            turn_results.append({
                "turn": turn, "val_bpb": None, "diff": "",
                "status": "no_sed_commands",
                "crash_reason": "no sed commands matched baseline train.py",
            })
            wandb.log({"turn": turn, "status": "no_sed_commands", "reward": 0.0})
            log_jsonl(output_path, {"turn": turn, "status": "no_sed_commands",
                                    "gen_time": gen_time,
                                    "model_output": result.text[:2000]})
            continue

        # Apply validated sed commands
        modified = apply_sed_commands(baseline, sed_commands)
        diff_text = make_diff(baseline, modified)

        print(f"  Diff size: {len(diff_text)} chars")

        if baseline == modified:
            print("  Status: no_changes (sed commands had no effect)")
            turn_results.append({
                "turn": turn, "val_bpb": None, "diff": "",
                "status": "no_changes",
                "crash_reason": "sed commands did not modify train.py",
            })
            wandb.log({"turn": turn, "status": "no_changes", "reward": 0.0})
            log_jsonl(output_path, {"turn": turn, "status": "no_changes",
                                    "gen_time": gen_time,
                                    "sed_commands": sed_commands})
            continue

        # Dispatch experiment
        print(f"  Dispatching experiment to {EXPERIMENT_FLEET[0].name}...")
        metrics, returncode, error_feedback = dispatch_experiment(modified)
        val_bpb = metrics.get("val_bpb")
        experiment_time = time.time() - t0

        if returncode != 0 or (returncode == 0 and val_bpb is None and error_feedback):
            status = "crash" if returncode != 0 else "no_metric"
            crash_reason = classify_crash(error_feedback)
            print(f"  Status: {status} — {crash_reason}")
            turn_results.append({
                "turn": turn, "val_bpb": None, "diff": diff_text,
                "status": status, "crash_reason": crash_reason,
            })
            wandb.log({"turn": turn, "status": status, "reward": 0.0,
                        "experiment_time": experiment_time,
                        "gen_time": gen_time,
                        "acceptance_rate": result.acceptance_rate})
            log_jsonl(output_path, {
                "turn": turn, "status": status, "diff": diff_text,
                "feedback": error_feedback, "experiment_time": experiment_time,
                "gen_time": gen_time, "sed_commands": sed_commands,
            })
            continue

        # Compute reward
        reward, status, reward_feedback = compute_reward(val_bpb, best_val_bpb)
        if status == "improvement":
            best_val_bpb = val_bpb
            best_trajectory_summary = previous_trajectory_summary

        memory_gb = metrics.get("peak_vram_mb", 0) / 1024
        depth = metrics.get("depth")
        tokens_M = metrics.get("total_tokens_M")

        turn_results.append({
            "turn": turn, "val_bpb": val_bpb, "diff": diff_text,
            "status": status,
            "depth": int(depth) if depth else None,
            "tokens_M": int(tokens_M) if tokens_M else None,
            "memory_gb": round(memory_gb, 1),
        })

        print(f"  val_bpb={val_bpb:.6f}  best={best_val_bpb:.6f}  "
              f"reward={reward:.4f}  status={status}  time={experiment_time:.0f}s")

        wandb.log({
            "turn": turn, "val_bpb": val_bpb, "best_val_bpb": best_val_bpb,
            "reward": reward, "memory_gb": memory_gb,
            "experiment_time": experiment_time, "status": status,
            "gen_time": gen_time,
            "acceptance_rate": result.acceptance_rate,
            "total_tokens_generated": result.total_tokens_generated,
            **{f"env/{k}": v for k, v in metrics.items()},
        })
        log_jsonl(output_path, {
            "turn": turn, "val_bpb": val_bpb, "best_val_bpb": best_val_bpb,
            "reward": reward, "status": status, "memory_gb": memory_gb,
            "experiment_time": experiment_time, "diff": diff_text,
            "metrics": metrics, "gen_time": gen_time,
            "sed_commands": sed_commands,
            "acceptance_rate": result.acceptance_rate,
            "total_tokens_generated": result.total_tokens_generated,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"Power sampling baseline complete: {args.max_turns} turns")
    print(f"Method: {args.method}")
    print(f"Best val_bpb: {best_val_bpb:.6f} (baseline: {BASELINE_VAL_BPB})")
    print(f"Output: {output_path}")

    wandb.log({"final/best_val_bpb": best_val_bpb,
               "final/improvement": max(0, BASELINE_VAL_BPB - best_val_bpb)})
    wandb.finish()


if __name__ == "__main__":
    main()
