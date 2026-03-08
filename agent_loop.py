"""
Custom agent loop for autoresearch SDPO training.

Each rollout: model generates a diff → we run a 5-min experiment → score it.
The reward is computed inline and returned via AgentLoopOutput.reward_score.
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from uuid import uuid4

# Ensure SDPO's verl is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SDPO"))

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TIMEOUT = 600  # 5 min + buffer
AUTORESEARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoresearch")
TRAIN_PY = "train.py"

# GPU pool: 8 experiment GPUs (indices 2-9, since 0-1 are for verl)
EXPERIMENT_GPU_IDS = list(range(2, 10))
_gpu_semaphore: asyncio.Semaphore | None = None
_gpu_queue: asyncio.Queue | None = None


def _init_gpu_pool():
    """Lazily initialize the GPU pool (must be called inside an event loop)."""
    global _gpu_semaphore, _gpu_queue
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(len(EXPERIMENT_GPU_IDS))
        _gpu_queue = asyncio.Queue()
        for gpu_id in EXPERIMENT_GPU_IDS:
            _gpu_queue.put_nowait(gpu_id)


# ---------------------------------------------------------------------------
# Diff parsing and patch application (adapted from loop.py)
# ---------------------------------------------------------------------------

def extract_diff(response: str) -> str | None:
    m = re.search(r"```diff\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    m = re.search(r"(--- a/.*?\n\+\+\+ b/.*?\n@@.*?)(?:\n```|$)", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return None


def apply_patch(diff_text: str, workdir: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff_text)
        patch_path = f.name
    try:
        r = subprocess.run(
            ["git", "apply", "--check", patch_path],
            capture_output=True, text=True, cwd=workdir, timeout=10,
        )
        if r.returncode == 0:
            r = subprocess.run(
                ["git", "apply", patch_path],
                capture_output=True, text=True, cwd=workdir, timeout=10,
            )
            if r.returncode == 0:
                return True, ""
            return False, r.stderr.strip()
        # Try 3-way merge as fallback
        r = subprocess.run(
            ["git", "apply", "--3way", patch_path],
            capture_output=True, text=True, cwd=workdir, timeout=10,
        )
        if r.returncode == 0:
            return True, ""
        return False, r.stderr.strip()
    finally:
        os.unlink(patch_path)


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
# Agent loop
# ---------------------------------------------------------------------------

@register("autoresearch_agent")
class AutoresearchAgentLoop(AgentLoopBase):
    """Agent loop that generates diffs, runs real experiments, and scores them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        _init_gpu_pool()

        # 1. Standard single-turn generation (mirrors SingleTurnAgentLoop)
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)

        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

        response_ids = output.token_ids[:self.response_length]
        response_mask = [1] * len(response_ids)
        response_logprobs = output.log_probs[:self.response_length] if output.log_probs else None

        # 2. Decode response to text
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 3. Run experiment and compute reward
        reward = await self._run_experiment(response_text)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward,
            num_turns=2,
            metrics={},
        )

    async def _run_experiment(self, response_text: str) -> float:
        """Extract diff, apply patch, run experiment, return reward score."""

        # Extract diff from model response
        diff_text = extract_diff(response_text)
        if diff_text is None:
            logger.info("No diff found in response")
            return -1.0  # Same as crash/patch_error

        # Acquire a GPU from the pool
        await _gpu_semaphore.acquire()
        gpu_id = await _gpu_queue.get()

        workdir = None
        try:
            # Create isolated working directory
            workdir = tempfile.mkdtemp(prefix="autoresearch_exp_")
            shutil.copytree(AUTORESEARCH_DIR, os.path.join(workdir, "autoresearch"))
            exp_dir = os.path.join(workdir, "autoresearch")

            # Initialize git in the copy so git apply works
            subprocess.run(["git", "init"], cwd=exp_dir, capture_output=True, timeout=10)
            subprocess.run(["git", "add", "."], cwd=exp_dir, capture_output=True, timeout=10)
            subprocess.run(
                ["git", "commit", "-m", "init", "--allow-empty"],
                cwd=exp_dir, capture_output=True, timeout=10,
                env={**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
                     "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"},
            )

            # Apply patch
            success, error = apply_patch(diff_text, exp_dir)
            if not success:
                logger.info(f"Patch failed: {error}")
                return -1.0

            # Run experiment as async subprocess
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")

            proc = await asyncio.create_subprocess_exec(
                "uv", "run", "train.py",
                cwd=exp_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=EXPERIMENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.info(f"Experiment timed out on GPU {gpu_id}")
                return -1.0

            stdout = stdout_bytes.decode(errors="replace")
            returncode = proc.returncode

            if returncode != 0:
                logger.info(f"Experiment crashed (exit {returncode}) on GPU {gpu_id}")
                return -1.0

            # Parse metrics
            metrics = parse_results(stdout)
            val_bpb = metrics.get("val_bpb")

            if val_bpb is None:
                logger.info("Could not parse val_bpb from experiment output")
                return -1.0

            # Compute reward based on val_bpb
            # Baseline val_bpb is ~1.187 (from initial train.py runs)
            # Lower is better, so improvement = baseline - val_bpb
            baseline_bpb = 1.187
            delta = baseline_bpb - val_bpb

            if delta > 0:
                # Improvement: positive reward scaled by magnitude
                return 1.0 + 10.0 * abs(delta)
            else:
                # No improvement: small negative reward
                return -0.1

        except Exception:
            logger.exception("Experiment execution failed")
            return -1.0

        finally:
            # Release GPU back to pool
            _gpu_queue.put_nowait(gpu_id)
            _gpu_semaphore.release()

            # Clean up working directory
            if workdir and os.path.exists(workdir):
                shutil.rmtree(workdir, ignore_errors=True)
