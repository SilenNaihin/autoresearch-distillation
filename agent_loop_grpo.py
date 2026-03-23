"""
Custom agent loop for autoresearch GRPO with multi-turn bash tool + PUCT reuse buffer.

Each rollout:
  1. PUCT selects a parent state (train.py version) from the reuse buffer
  2. Model receives prompt with selected train.py
  3. Model uses bash tool to read/edit train.py (multi-turn)
  4. Model submits via: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  5. We read modified train.py from the isolated workdir
  6. We dispatch to GPU fleet (box2)
  7. We parse metrics, compute reward
  8. Successful experiments are added back to the reuse buffer
  9. reward_score is set on AgentLoopOutput for RL training

GRPO computes advantages from within-group comparison of rewards (n=4).
PUCT reuse buffer steers exploration toward promising states.
Entropy bonus prevents policy collapse.
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
from reuse_buffer import ReuseBuffer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SDPO"))

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# GRPO experiment fleet
# ---------------------------------------------------------------------------

from runners import GPUSlot, GPUPoolRunner

GRPO_FLEET = [
    GPUSlot("h100-dev-box-2", "0", "box2-gpu0", "~/autoresearch"),
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
# Shared reuse buffer (singleton, thread-safe)
# ---------------------------------------------------------------------------

_buffer = None
_buffer_lock = threading.Lock()
_buffer_kwargs: dict = {}


def _get_buffer() -> ReuseBuffer:
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                _buffer = ReuseBuffer(Path("/data/reuse_buffer_grpo.json"), **_buffer_kwargs)
    return _buffer


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@register("autoresearch_grpo_agent")
class GRPOAgentLoop(ToolAgentLoop):
    """Multi-turn agent loop for GRPO with PUCT reuse buffer.

    Like SDPO's AutoresearchAgentLoop but without self-distillation or feedback.
    PUCT steers exploration; entropy bonus prevents policy collapse.
    """

    def __init__(self, *args, c_puct: float = 1.0, max_states: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        global _buffer_kwargs
        _buffer_kwargs = {"c_puct": c_puct, "max_states": max_states}

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

        # Select a state from the reuse buffer
        from prompts import replace_train_py_in_prompt
        from environment import BASELINE_VAL_BPB

        buffer = _get_buffer()
        bash_tool = self.tools.get("bash")

        # Seed with baseline train.py if buffer is empty
        if bash_tool and len(buffer) == 0:
            baseline_code = Path(bash_tool.autoresearch_dir, "train.py").read_text()
            buffer.seed(baseline_code, BASELINE_VAL_BPB)

        # Store original baseline for cache keying
        self._baseline_code = buffer._data["states"].get("0", {}).get("code", "")

        # PUCT selection
        selections = buffer.select(1)
        if selections:
            self._parent_id, self._selected_code = selections[0]
        else:
            self._parent_id = 0
            self._selected_code = Path(bash_tool.autoresearch_dir, "train.py").read_text() if bash_tool else ""

        # Replace train.py in prompt with the selected state's code
        if "raw_prompt" in kwargs:
            kwargs["raw_prompt"] = list(kwargs["raw_prompt"])
            last_msg = dict(kwargs["raw_prompt"][-1])
            last_msg["content"] = replace_train_py_in_prompt(last_msg["content"], self._selected_code)
            # Append best val_bpb target
            best_bpb = buffer.get_best_val_bpb()
            if best_bpb < BASELINE_VAL_BPB:
                last_msg["content"] += f"\n\n## Best val_bpb achieved so far: {best_bpb:.6f} (baseline={BASELINE_VAL_BPB})"
            kwargs["raw_prompt"][-1] = last_msg

        # Pre-create persistent bash tool instance
        instance_id = None
        if bash_tool:
            instance_id, _ = await bash_tool.create()
            # Overwrite workdir train.py with the selected state's code
            workdir = bash_tool.get_workdir(instance_id)
            if workdir and self._selected_code:
                Path(workdir, "train.py").write_text(self._selected_code)
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
            output.extra_fields["reward_extra_info"] = {
                "feedback": feedback, "parent_id": int(self._parent_id), **env_metrics
            }

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

        # Two diffs: cache_diff keys on original baseline, feedback_diff shows changes from selected parent
        cache_diff = _make_diff(self._baseline_code, modified)
        feedback_diff = _make_diff(self._selected_code, modified)

        from environment import BASELINE_VAL_BPB, compute_reward, parse_metrics

        if self._selected_code == modified:
            if self._sed_failed:
                return 0.0, f"FAILURE: your sed command failed: {self._sed_failed}"
            return 0.0, "No changes were made to train.py."

        # Check reuse buffer — if this exact code is already a known state, skip GPU
        buffer = _get_buffer()
        known = buffer.find_by_code(modified)
        if known is not None:
            return known["reward"], f"This code has already been evaluated: val_bpb={known['val_bpb']:.6f}."

        # Check experiment cache
        cached = self._cache.get(cache_diff, current_step=self._global_step)
        if cached is not None:
            if cached.get("crashed"):
                tail = cached.get("tail", cached.get("feedback", "unknown error"))
                return 0.0, f"These changes caused the experiment to crash:\n{tail}"
            # Handle both old format (reward/feedback) and new format (val_bpb/metrics_line)
            val_bpb = cached.get("val_bpb")
            if val_bpb is not None:
                reward, status, reward_feedback = compute_reward(val_bpb)
                metrics_line = cached.get("metrics_line", "")
                if reward > 0 and buffer.find_by_code(modified) is None:
                    buffer.add(modified, val_bpb, reward, self._parent_id)
                return reward, f"{reward_feedback}\n{metrics_line}"
            # Old GRPO cache format: has reward + feedback directly
            return cached.get("reward", 0.0), cached.get("feedback", "")

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
                self._cache.put(cache_diff, {"crashed": True, "tail": crash_info}, step=self._global_step)
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

        # Cache for future dedup
        self._cache.put(cache_diff, {"val_bpb": val_bpb, "metrics_line": metrics_line},
                        step=self._global_step, val_bpb=val_bpb, diff_text_raw=cache_diff)

        # Add successful experiments to reuse buffer
        if reward > 0:
            buffer.add(modified, val_bpb, reward, self._parent_id)

        return reward, feedback
