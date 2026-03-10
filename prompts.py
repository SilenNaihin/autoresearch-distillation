"""
Shared prompt templates for autoresearch SDPO training.

Used by agent_loop.py (VERL RL training) and data/prepare_autoresearch.py.
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

## Workflow
1. Think step-by-step about what changes could lower val_bpb (architecture, hyperparameters, optimization, etc.)
2. Make targeted edits to train.py using sed. Do not rewrite the entire file. Example:
   <tool_call>
   {"name": "bash", "arguments": {"command": "sed -i 's/MATRIX_LR = 0.04/MATRIX_LR = 0.06/' train.py"}}
   </tool_call>
3. When complete, submit: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

## Important
- You are ONLY editing the file. You do NOT run experiments.
- We will run uv run train.py on a GPU after you submit.
- Think carefully about what will improve training, then make correct edits.\
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
        "Prefer a single focused change to train.py to lower val_bpb, "
        "but you may combine related changes if they work together. "
        "You must apply your changes directly using the bash tool — do not just output a diff."
    )
    return "\n\n".join(parts)
