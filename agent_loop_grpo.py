"""
Custom agent loop for autoresearch GRPO baseline with multi-turn bash tool.

Simplified version of agent_loop.py (SDPO) that removes:
- Reuse buffer (PUCT tree search) — always starts from baseline train.py
- Feedback pipeline — no teacher reprompting, no self-distillation
- Parent tracking — no state tree

Each rollout:
  1. Model receives prompt (baseline train.py from dataset)
  2. Model uses bash tool to read/edit train.py (multi-turn)
  3. Model submits via: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  4. We read modified train.py from the isolated workdir
  5. We dispatch to GRPO experiment fleet (box6-gpu1)
  6. We parse metrics, compute reward
  7. reward_score is set on AgentLoopOutput for RL training

GRPO computes advantages from within-group comparison of rewards (n=4).
No self-distillation, no teacher EMA, no feedback reprompting.
"""

import asyncio
import json
import logging
import os
import sys
import threading
from difflib import unified_diff
from pathlib import Path
from typing import Any

from experiment_cache import GRPO_CACHE, ExperimentCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SDPO"))

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# GRPO experiment fleet (dedicated — does not share with SDPO)
# ---------------------------------------------------------------------------

from runners import GPUSlot, GPUPoolRunner

GRPO_FLEET = [
    GPUSlot("h100-dev-box-6", "1", "box6-gpu1", "~/autoresearch-gpu1"),
]

_pool = None
_pool_lock = threading.Lock()


def _make_diff(baseline: str, modified: str) -> str:
    if baseline == modified:
        return "No changes were made to train.py."
    return "".join(unified_diff(
        baseline.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="a/train.py", tofile="b/train.py",
    ))


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = GPUPoolRunner(slots=GRPO_FLEET)
                logger.info(f"GRPO: Initialized GPUPoolRunner with {_pool.total} GPU slots")
    return _pool


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@register("autoresearch_grpo_agent")
class GRPOAgentLoop(ToolAgentLoop):
    """Multi-turn agent loop for GRPO baseline.

    Simpler than AutoresearchAgentLoop (SDPO):
    - No reuse buffer — always starts from baseline train.py in the prompt
    - No feedback pipeline — reward comes purely from group comparison
    - Experiment cache still used to avoid redundant GPU dispatches
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sed_failed: str | None = None
        self._cache = ExperimentCache(write_path=GRPO_CACHE)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        global_step = kwargs.get("global_step")
        assert global_step is not None and global_step >= 0, (
            f"global_step not forwarded to agent loop (got {global_step!r})"
        )
        self._global_step = global_step

        self._total_tool_calls = 0
        self._failed_tool_calls = 0
        self._noop_tool_calls = 0
        self._sed_failed = None

        # Read baseline train.py for diff comparison
        bash_tool = self.tools.get("bash")
        self._baseline_code = ""
        if bash_tool:
            self._baseline_code = Path(bash_tool.autoresearch_dir, "train.py").read_text()

        # Pre-create persistent bash tool instance
        instance_id = None
        if bash_tool:
            instance_id, _ = await bash_tool.create()
            kwargs.setdefault("tools_kwargs", {})
            kwargs["tools_kwargs"]["_bash_instance_id"] = instance_id

        try:
            output = await super().run(sampling_params, **kwargs)

            # Dispatch experiment and compute reward
            reward, feedback = await self._dispatch_experiment(bash_tool, instance_id)
            output.reward_score = reward

            env_metrics = {}
            for k in ("val_bpb", "peak_vram_mb", "training_seconds", "total_seconds",
                      "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth"):
                env_metrics[f"env_{k}"] = float(getattr(self, '_last_env_metrics', {}).get(k, float('nan')))
            env_metrics["env_novel"] = self._is_novel
            output.extra_fields["reward_extra_info"] = {"feedback": feedback, **env_metrics}

            return output
        finally:
            if bash_tool and instance_id:
                await bash_tool.release(instance_id)

    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        bash_tool = self.tools.get("bash")
        instance_id = agent_data.tools_kwargs.get("_bash_instance_id")
        if bash_tool and instance_id and bash_tool.is_submitted(instance_id):
            return AgentState.TERMINATED
        return state

    _TRAIN_PY_CMDS = ("sed", "cat", "echo", "python", "python3", "printf", "tee", "patch")

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data
    ) -> tuple[ToolResponse, float, dict]:
        if tool_call.name == "bash":
            bash_tool = self.tools["bash"]
            instance_id = tools_kwargs.get("_bash_instance_id")
            if instance_id is None:
                return ToolResponse(text="Error: no bash instance"), 0.0, {}

            try:
                tool_args = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                self._failed_tool_calls += 1
                self._total_tool_calls += 1
                return ToolResponse(text="Error: invalid JSON in tool arguments"), 0.0, {}

            if not isinstance(tool_args, dict):
                self._failed_tool_calls += 1
                self._total_tool_calls += 1
                return ToolResponse(text="Error: tool arguments must be a JSON object"), 0.0, {}

            try:
                self._total_tool_calls += 1
                cmd = tool_args.get("command", "")
                cmd_stripped = cmd.strip().split()[0] if cmd.strip() else ""
                if cmd_stripped and cmd_stripped not in self._TRAIN_PY_CMDS and "train.py" not in cmd:
                    self._noop_tool_calls += 1

                response, reward, metrics = await bash_tool.execute(
                    instance_id, tool_args, agent_data=agent_data
                )

                resp_text = (response.text or "").strip()
                if cmd_stripped == "sed" and "sed:" in resp_text:
                    self._sed_failed = resp_text

                if response.text and len(response.text) > self.max_tool_response_length:
                    text = response.text
                    if self.tool_response_truncate_side == "left":
                        text = text[:self.max_tool_response_length] + "...(truncated)"
                    elif self.tool_response_truncate_side == "right":
                        text = "(truncated)..." + text[-self.max_tool_response_length:]
                    else:
                        half = self.max_tool_response_length // 2
                        text = text[:half] + "...(truncated)..." + text[-half:]
                    response = ToolResponse(text=text)

                return response, reward, metrics
            except Exception as e:
                logger.warning(f"Error when executing bash tool: {e}")
                return ToolResponse(text=f"Error when executing tool: {e}"), 0.0, {}
        else:
            return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    async def _dispatch_experiment(self, bash_tool, instance_id: str | None) -> tuple[float, str]:
        """Read modified train.py, dispatch to GPU fleet, compute reward."""
        self._is_novel = 0.0
        self._last_env_metrics = {}

        if bash_tool is None or instance_id is None:
            return 0.0, "No bash tool or instance available."

        modified = bash_tool.read_train_py(instance_id)
        if modified is None:
            return 0.0, "No modified train.py found."

        diff_text = _make_diff(self._baseline_code, modified)

        from environment import BASELINE_VAL_BPB, compute_reward, parse_metrics

        if self._baseline_code == modified:
            if self._sed_failed:
                return 0.0, f"FAILURE: your sed command failed: {self._sed_failed}"
            return 0.0, "No changes were made to train.py."

        # Check experiment cache
        cached = self._cache.get(diff_text, current_step=self._global_step)
        if cached is not None:
            return cached["reward"], cached["feedback"]

        # Dispatch to GPU fleet
        self._is_novel = 1.0
        pool = _get_pool()
        output = await asyncio.to_thread(pool.run, modified)

        metrics = parse_metrics(output.stdout) if output.stdout else {}
        self._last_env_metrics = metrics

        if output.returncode != 0:
            parts = []
            if output.stderr and output.stderr.strip():
                parts.append(output.stderr.strip()[:1000])
            if output.stdout and output.stdout.strip():
                parts.append(output.stdout.strip()[-1000:])
            crash_info = "\n".join(parts) if parts else "no output"
            feedback = f"Experiment crashed (exit {output.returncode}):\n{crash_info}"
            if "CUDA out of memory" in crash_info:
                self._cache.put(diff_text, {"reward": 0.0, "feedback": feedback, "crashed": True}, step=self._global_step)
            return 0.0, feedback

        val_bpb = metrics.get("val_bpb")
        if val_bpb is None:
            tail = "\n".join(output.stdout.strip().splitlines()[-20:]) if output.stdout else "empty output"
            return 0.0, f"Experiment ran but produced no val_bpb. Output tail:\n{tail}"

        reward, status, reward_feedback = compute_reward(val_bpb)

        metrics_line = " | ".join(
            f"{k}: {metrics[k]:g}" for k in ("num_steps", "num_params_M", "peak_vram_mb", "mfu_percent")
            if k in metrics
        )
        feedback = f"{reward_feedback}\n{metrics_line}"

        self._cache.put(diff_text, {"reward": reward, "feedback": feedback},
                        step=self._global_step, val_bpb=val_bpb, diff_text_raw=diff_text)
        return reward, feedback
