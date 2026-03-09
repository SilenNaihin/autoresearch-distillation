"""
Experiment environment for the autoresearch distillation loop.

Clean interface between two concerns:
  - ExperimentRunner (you implement): HOW to run train.py on a GPU
  - ExperimentEnvironment: WHAT to do — prompt construction, diff handling, reward computation

Your friend plugs ExperimentEnvironment into VERL's reward pipeline.

Usage:
    from runners import GPUPoolRunner
    env = ExperimentEnvironment(runner=GPUPoolRunner())

    # VERL calls these:
    system, user = env.get_prompt()
    result = env.step(model_response)  # result.reward is the RL signal
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RunOutput:
    stdout: str
    stderr: str
    returncode: int


@dataclass
class ExperimentResult:
    status: str  # "improvement" | "no_improvement" | "crash" | "patch_error" | "parse_error"
    reward: float
    val_bpb: float | None = None
    best_val_bpb: float | None = None
    memory_gb: float | None = None
    diff: str | None = None
    description: str = ""
    feedback: str = ""
    metrics: dict = field(default_factory=dict)
    iteration: int = 0


# ---------------------------------------------------------------------------
# Runner protocol — implement this for your GPU setup
# ---------------------------------------------------------------------------

class ExperimentRunner(Protocol):
    """Run train.py on a GPU and return the output.

    Receives the modified train.py content (not a file path), so multiple
    experiments can be dispatched concurrently without racing on local files.

    Implement this to control WHERE experiments execute:
      - LocalRunner: subprocess on a local GPU (included below)
      - SSHRunner: send content + run on remote box (see runners.py)
      - GPUPoolRunner: auto-allocate across fleet (see runners.py)
    """

    def run(self, train_py: str) -> RunOutput: ...


class LocalRunner:
    """Runs experiments on a local GPU via subprocess."""

    def __init__(self, cwd: str, gpu_id: str = "1", timeout: int = 600):
        self.cwd = cwd
        self.gpu_id = gpu_id
        self.timeout = timeout

    def run(self, train_py: str) -> RunOutput:
        train_py_path = os.path.join(self.cwd, "train.py")
        Path(train_py_path).write_text(train_py)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        try:
            r = subprocess.run(
                ["uv", "run", "train.py"],
                capture_output=True, text=True,
                env=env, cwd=self.cwd, timeout=self.timeout,
            )
            return RunOutput(r.stdout, r.stderr, r.returncode)
        except subprocess.TimeoutExpired:
            return RunOutput("", f"TIMEOUT: exceeded {self.timeout}s", -1)
        finally:
            subprocess.run(["git", "checkout", "--", "train.py"],
                           cwd=self.cwd, capture_output=True, timeout=10)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ExperimentEnvironment:
    """Manages prompt construction, diff application, experiment execution, and reward computation.

    Interface for VERL integration:
        env = ExperimentEnvironment(runner=GPUPoolRunner())

        system, user = env.get_prompt()       # → feed to policy model
        result = env.step(model_response)     # → result.reward for RL
    """

    def __init__(
        self,
        runner: ExperimentRunner,
        autoresearch_dir: str = "autoresearch",
        results_file: str = "results.tsv",
        rollout_file: str = "rollouts/rollouts.jsonl",
        history_tail: int = 20,
    ):
        self.runner = runner
        self.autoresearch_dir = os.path.abspath(autoresearch_dir)
        self.train_py_path = os.path.join(self.autoresearch_dir, "train.py")
        self.results_file = results_file
        self.rollout_file = rollout_file
        self.history_tail = history_tail

        self.baseline = Path(self.train_py_path).read_text()
        self.best_val_bpb = float("inf")
        self.iteration = 0

        self._init_results_file()

    # -- Public interface --

    def get_prompt(self) -> tuple[str, str]:
        """Returns (system_prompt, user_message) for the current environment state."""
        history = self._read_history()
        return SYSTEM_PROMPT, _build_user_message(self.baseline, history)

    def step(self, model_response: str) -> ExperimentResult:
        """Run one experiment cycle from a model response.

        Extract diff → apply patch → delegate to runner → parse → compute reward → revert.
        """
        self.iteration += 1

        # 1. Extract diff
        diff_text = extract_diff(model_response)
        if diff_text is None:
            return self._finish("parse_error", model_response, reward=-1.0,
                                feedback="No unified diff found. Output a ```diff ... ``` block.")

        # 2. Apply patch, snapshot modified content, revert immediately
        #    (fast — holds no lock during the 5-min experiment)
        ok, err = self._apply_patch(diff_text)
        if not ok:
            self._revert()
            return self._finish("patch_error", model_response, reward=-1.0,
                                diff=diff_text, feedback=f"Patch failed: {err}")
        modified_train_py = Path(self.train_py_path).read_text()
        self._revert()

        description = _extract_description(model_response)

        # 3. Run experiment (the slow part — delegates to your runner)
        output = self.runner.run(modified_train_py)

        # 5. Handle crash
        if output.returncode != 0:
            crash_tail = _tail(output.stderr or output.stdout or "no output", 30)
            return self._finish("crash", model_response, reward=-1.0,
                                diff=diff_text, description=description,
                                feedback=f"CRASH (exit {output.returncode}):\n{crash_tail}")

        # 6. Parse metrics
        metrics = parse_metrics(output.stdout)
        val_bpb = metrics.get("val_bpb")
        memory_gb = metrics.get("peak_vram_mb", 0) / 1024

        if val_bpb is None:
            tail = _tail(output.stdout, 20)
            return self._finish("crash", model_response, reward=-1.0,
                                diff=diff_text, description=description,
                                metrics=metrics, feedback=f"No val_bpb in output:\n{tail}")

        # 7. Compute reward from val_bpb
        reward, status, feedback = compute_reward(val_bpb, self.best_val_bpb)
        if status == "improvement":
            self.best_val_bpb = val_bpb
        feedback += f" Memory: {memory_gb:.1f} GB."

        return self._finish(status, model_response, reward=reward,
                            val_bpb=val_bpb, best_val_bpb=self.best_val_bpb,
                            memory_gb=memory_gb, diff=diff_text,
                            description=description, feedback=feedback, metrics=metrics)

    # -- Internals --

    def _finish(self, status: str, model_response: str, **kwargs) -> ExperimentResult:
        """Build result, log it, save rollout."""
        result = ExperimentResult(status=status, iteration=self.iteration, **kwargs)
        val = result.val_bpb or 0.0
        mem = result.memory_gb or 0.0
        desc = result.description or status
        self._log_result(val, mem, status, desc[:60])
        self._save_rollout(model_response, result)
        return result

    def _init_results_file(self):
        if not os.path.exists(self.results_file):
            with open(self.results_file, "w") as f:
                f.write("iteration\tval_bpb\tmemory_gb\tstatus\tdescription\n")

    def _read_history(self) -> list[str]:
        if not os.path.exists(self.results_file):
            return []
        with open(self.results_file) as f:
            lines = f.read().strip().splitlines()
        if len(lines) <= 1:
            return []
        return [lines[0]] + lines[1:][-self.history_tail:]

    def _log_result(self, val_bpb: float, memory_gb: float, status: str, description: str):
        with open(self.results_file, "a") as f:
            f.write(f"{self.iteration}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")

    def _save_rollout(self, model_response: str, result: ExperimentResult):
        system, user = self.get_prompt()
        rollout = {
            "conversations": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": model_response},
                {"role": "user", "content": result.feedback},
            ],
            "result": asdict(result),
        }
        os.makedirs(os.path.dirname(self.rollout_file), exist_ok=True)
        with open(self.rollout_file, "a") as f:
            f.write(json.dumps(rollout) + "\n")

    def _apply_patch(self, diff_text: str) -> tuple[bool, str]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(diff_text)
            patch_path = f.name
        try:
            # Try clean apply first
            r = subprocess.run(["git", "apply", "--check", patch_path],
                               capture_output=True, text=True,
                               cwd=self.autoresearch_dir, timeout=10)
            if r.returncode == 0:
                r = subprocess.run(["git", "apply", patch_path],
                                   capture_output=True, text=True,
                                   cwd=self.autoresearch_dir, timeout=10)
                if r.returncode == 0:
                    return True, ""
                return False, r.stderr.strip()
            # Fallback to 3-way merge
            r = subprocess.run(["git", "apply", "--3way", patch_path],
                               capture_output=True, text=True,
                               cwd=self.autoresearch_dir, timeout=10)
            if r.returncode == 0:
                return True, ""
            return False, r.stderr.strip()
        finally:
            os.unlink(patch_path)

    def _revert(self):
        subprocess.run(["git", "checkout", "--", "train.py"],
                       cwd=self.autoresearch_dir, check=True, timeout=10)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous ML researcher optimizing a GPT pretraining script.

## Your task
Propose a modification to train.py that will lower val_bpb (bits per byte on validation data). \
The training budget is fixed at 5 minutes wall-clock. Lower val_bpb is better.

## Rules
- You may ONLY modify train.py. Do not touch prepare.py or any other file.
- You can only use packages already in pyproject.toml: torch, numpy, kernels, matplotlib, \
pandas, pyarrow, requests, rustbpe, tiktoken.
- Do not modify the evaluation harness. The evaluate_bpb function in prepare.py is the ground truth.
- VRAM is a soft constraint. Some increase is OK for meaningful val_bpb gains.
- Simpler is better. A small improvement that adds ugly complexity is not worth it.
- The script runs on a single GPU via: uv run train.py

## Output format
Output your reasoning, then a single unified diff (git patch) that can be applied with \
`git apply`. The diff must modify only train.py.

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


def _build_user_message(train_py_content: str, history_lines: list[str]) -> str:
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
# Utilities (public — your friend may use these directly)
# ---------------------------------------------------------------------------

def extract_diff(response: str) -> str | None:
    """Pull a unified diff from a model response."""
    m = re.search(r"```diff\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    m = re.search(r"(--- a/.*?\n\+\+\+ b/.*?\n@@.*?)(?:\n```|$)", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return None


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


def _extract_description(response: str, max_len: int = 80) -> str:
    for line in response.splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "---", "+++", "@@", "-", "+")):
            continue
        line = re.sub(r"^#+\s*", "", line)
        if len(line) > 10:
            return line[:max_len]
    return "no description"


BASELINE_VAL_BPB = 0.9979


def compute_reward(val_bpb: float | None, best_val_bpb: float = None) -> tuple[float, str, str]:
    """Reward = max(0, baseline - val_bpb). Failures get 0."""
    if val_bpb is None:
        return 0.0, "crash", "No val_bpb in output."
    reward = max(0.0, BASELINE_VAL_BPB - val_bpb)
    if reward > 0:
        return reward, "improvement", f"val_bpb={val_bpb:.6f} (baseline={BASELINE_VAL_BPB}, improved by {reward:.6f})"
    return 0.0, "no_improvement", f"val_bpb={val_bpb:.6f} (baseline={BASELINE_VAL_BPB}, no improvement)"


def _tail(text: str, n: int) -> str:
    return "\n".join(text.strip().splitlines()[-n:])
