"""
Shared prompt templates for the autoresearch distillation loop.

Used by both loop_swe.py (data collection) and agent_loop.py (VERL RL training).
"""

SYSTEM_PROMPT = """\
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
- Simpler is better. A small improvement that adds ugly complexity is not worth it.
- The script runs on a single GPU via: uv run train.py

## Workflow
1. Read train.py to understand the current implementation
2. Make targeted modifications using bash commands (sed, cat heredoc, python scripts, etc.)
3. Verify your changes by reading the modified file
4. When satisfied, submit: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

## Important
- You are ONLY editing the file. You do NOT run experiments.
- We will run uv run train.py on a GPU after you submit.
- Focus on making correct, clean edits.\
"""


def build_instance_prompt(train_py_content: str, history_lines: list[str]) -> str:
    """Build the task prompt with current train.py and experiment history."""
    parts = ["## Current train.py\n```python\n" + train_py_content + "\n```"]

    if history_lines:
        parts.append("## Experiment history (recent)\n```\n" + "\n".join(history_lines) + "\n```")
        parts.append(
            "Each row shows: iteration, val_bpb, memory_gb, status, description.\n"
            "Learn from past experiments. Don't repeat things that didn't work. "
            "Build on ideas that improved val_bpb."
        )
    parts.append(
        "Make a single focused change to train.py to lower val_bpb. "
        "You must apply your changes directly using the bash tool — do not just output a diff."
    )
    return "\n\n".join(parts)
