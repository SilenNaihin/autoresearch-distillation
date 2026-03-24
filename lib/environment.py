"""
Shared utilities for autoresearch experiment evaluation.

Used by agent_loop.py for metric parsing and reward computation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunOutput:
    stdout: str
    stderr: str
    returncode: int


BASELINE_VAL_BPB = 1.056


def parse_metrics(output: str) -> dict:
    """Parse train.py output for key metrics."""
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


def compute_reward(val_bpb: float | None, best_val_bpb: float = None) -> tuple[float, str, str]:
    """Reward = max(0, baseline - val_bpb). Failures get 0."""
    if val_bpb is None:
        return 0.0, "crash", "No val_bpb in output."
    reward = max(0.0, BASELINE_VAL_BPB - val_bpb)
    if reward > 0:
        return reward, "improvement", f"val_bpb={val_bpb:.6f} (baseline={BASELINE_VAL_BPB}, delta={-reward:.6f})"
    return 0.0, "no_improvement", f"val_bpb={val_bpb:.6f} (baseline={BASELINE_VAL_BPB}, delta=+{val_bpb - BASELINE_VAL_BPB:.6f})"
