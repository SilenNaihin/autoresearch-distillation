"""
Custom agent loop for GRPO with multi-turn bash tool + PUCT reuse buffer.

Each rollout:
  1. PUCT selects a parent state from the reuse buffer
  2. Model receives prompt with selected file content
  3. Model uses bash tool to edit (multi-turn)
  4. Model submits via: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  5. We read modified file from the isolated workdir
  6. We dispatch to experiment fleet
  7. We parse metrics, compute reward
  8. Successful experiments are added back to the reuse buffer
  9. reward_score is set on AgentLoopOutput for RL training

Generic — all task-specific config comes from TaskConfig.

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
from pathlib import Path
from typing import Any

from lib.experiment_cache import ExperimentCache, cache_path_for
from lib.reuse_buffer import ReuseBuffer
from task_config import TaskConfig

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

from lib.runners import GPUSlot, GPUPoolRunner

_pool = None
_pool_lock = threading.Lock()


def _get_pool(task: TaskConfig, fleet_override: list[dict] | None = None):
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                if fleet_override:
                    slots = [GPUSlot(**s) for s in fleet_override]
                    _pool = GPUPoolRunner(slots=slots, task=task)
                else:
                    _pool = GPUPoolRunner(task=task)
                logger.info(f"GRPO: Initialized GPUPoolRunner with {_pool.total} slots")
    return _pool


# ---------------------------------------------------------------------------
# Shared reuse buffer (singleton, thread-safe)
# ---------------------------------------------------------------------------

_buffer = None
_buffer_lock = threading.Lock()
_buffer_kwargs: dict = {}


def _get_buffer(task: TaskConfig) -> ReuseBuffer:
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                buf_path = Path(f"/data/reuse_buffer_{task.name}_grpo.json")
                _buffer = ReuseBuffer(buf_path, direction=task.scoring.direction,
                                      **_buffer_kwargs)
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

    def __init__(self, *args, c_puct: float = 1.0, max_states: int = 1000,
                 task_config: str = "tasks/autoresearch.yaml",
                 fleet_override: list[dict] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = TaskConfig.from_yaml(task_config)
        self._fleet_override = fleet_override

        global _buffer_kwargs
        _buffer_kwargs = {"c_puct": c_puct, "max_states": max_states}

        self._sed_failed: str | None = None
        self._cache = ExperimentCache(
            write_path=cache_path_for(self.task.name, "grpo"),
            direction=self.task.scoring.direction,
        )

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

        task = self.task
        target_file = task.workspace.target_file

        buffer = _get_buffer(task)
        bash_tool = self.tools.get("bash")

        # Seed with baseline file if buffer is empty
        if bash_tool and len(buffer) == 0:
            baseline_code = Path(bash_tool.source_dir, target_file).read_text()
            buffer.seed(baseline_code, task.scoring.baseline)

        # Store original baseline for cache keying
        self._baseline_code = buffer._data["states"].get("0", {}).get(
            "content", buffer._data["states"].get("0", {}).get("code", ""))

        # PUCT selection
        selections = buffer.select(1)
        if selections:
            self._parent_id, self._selected_code = selections[0]
        else:
            self._parent_id = 0
            self._selected_code = Path(bash_tool.source_dir, target_file).read_text() if bash_tool else ""

        # Replace target file in prompt with the selected state's content
        if "raw_prompt" in kwargs:
            kwargs["raw_prompt"] = list(kwargs["raw_prompt"])
            last_msg = dict(kwargs["raw_prompt"][-1])
            last_msg["content"] = task.replace_file_in_prompt(last_msg["content"], self._selected_code)
            # Append best metric target
            best = buffer.get_best_metric()
            if task.is_improvement(best):
                metric = task.scoring.metric
                baseline = task.scoring.baseline
                last_msg["content"] += f"\n\n## Best {metric} achieved so far: {best:.6f} (baseline={baseline})"
            kwargs["raw_prompt"][-1] = last_msg

        # Pre-create persistent bash tool instance
        instance_id = None
        if bash_tool:
            instance_id, _ = await bash_tool.create()
            workdir = bash_tool.get_workdir(instance_id)
            if workdir and self._selected_code:
                Path(workdir, target_file).write_text(self._selected_code)
            kwargs.setdefault("tools_kwargs", {})
            kwargs["tools_kwargs"]["_bash_instance_id"] = instance_id

        try:
            output = await super().run(sampling_params, **kwargs)

            # Dispatch experiment and compute reward
            reward, feedback = await self._dispatch_experiment(bash_tool, instance_id)
            output.reward_score = reward

            env_metrics = {}
            for k in task.scoring.metrics:
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

    _EDIT_CMDS = ("sed", "cat", "echo", "python", "python3", "printf", "tee", "patch")

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
                target_file = self.task.workspace.target_file
                cmd_stripped = cmd.strip().split()[0] if cmd.strip() else ""
                if cmd_stripped and cmd_stripped not in self._EDIT_CMDS and target_file not in cmd:
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
        """Read modified file, dispatch to fleet, compute reward."""
        self._is_novel = 0.0
        self._last_env_metrics = {}
        task = self.task
        metric_name = task.scoring.metric

        if bash_tool is None or instance_id is None:
            return 0.0, "No bash tool or instance available."

        modified = bash_tool.read_target_file(instance_id)
        if modified is None:
            return 0.0, f"No modified {task.workspace.target_file} found."

        # Two diffs: cache_diff keys on original baseline, feedback_diff shows changes from selected parent
        cache_diff = task.make_diff(self._baseline_code, modified)
        feedback_diff = task.make_diff(self._selected_code, modified)

        if self._selected_code == modified:
            if self._sed_failed:
                return 0.0, task.fmt_feedback("sed_failed", error=self._sed_failed)
            return 0.0, f"No changes were made to {task.workspace.target_file}."

        # Check reuse buffer
        buffer = _get_buffer(task)
        known = buffer.find_by_content(modified)
        if known is not None:
            mv = known.get("metric_value", known.get("val_bpb"))
            return known["reward"], f"This code has already been evaluated: {metric_name}={mv:.6f}."

        # Check experiment cache
        cached = self._cache.get(cache_diff, current_step=self._global_step)
        if cached is not None:
            if cached.get("crashed"):
                tail = cached.get("tail", cached.get("feedback", "unknown error"))
                return 0.0, f"These changes caused the experiment to crash:\n{tail}"
            val = cached.get("val_bpb", cached.get("metric_value"))
            if val is not None:
                reward, status, reward_feedback = task.compute_reward(val)
                metrics_line = cached.get("metrics_line", "")
                if reward > 0 and buffer.find_by_content(modified) is None:
                    buffer.add(modified, val, reward, self._parent_id)
                return reward, f"{reward_feedback}\n{metrics_line}"
            # Old cache format
            return cached.get("reward", 0.0), cached.get("feedback", "")

        # Dispatch to fleet
        self._is_novel = 1.0
        pool = _get_pool(task, self._fleet_override)
        output = await asyncio.to_thread(pool.run, modified)

        metrics = task.parse_metrics(output.stdout) if output.stdout else {}
        self._last_env_metrics = metrics

        if output.returncode != 0:
            parts = []
            if output.stderr and output.stderr.strip():
                parts.append(output.stderr.strip()[:1000])
            if output.stdout and output.stdout.strip():
                parts.append(output.stdout.strip()[-1000:])
            crash_info = "\n".join(parts) if parts else "no output"
            feedback = f"Experiment crashed (exit {output.returncode}):\n{crash_info}"
            if task.is_cacheable_crash(crash_info):
                self._cache.put(cache_diff, {"crashed": True, "tail": crash_info}, step=self._global_step)
            return 0.0, feedback

        val = metrics.get(metric_name)
        if val is None:
            tail = "\n".join(output.stdout.strip().splitlines()[-20:]) if output.stdout else "empty output"
            return 0.0, f"Experiment ran but produced no {metric_name}. Output tail:\n{tail}"

        reward, status, reward_feedback = task.compute_reward(val)

        display = task.scoring.display_metrics
        metrics_line = " | ".join(
            f"{k}: {metrics[k]:g}" for k in display if k in metrics
        )
        feedback = f"{reward_feedback}\n{metrics_line}"

        # Cache for future dedup
        self._cache.put(cache_diff, {"val_bpb": val, "metric_value": val, "metrics_line": metrics_line},
                        step=self._global_step, metric_value=val, diff_text_raw=cache_diff)

        # Add successful experiments to reuse buffer
        if reward > 0:
            buffer.add(modified, val, reward, self._parent_id)

        return reward, feedback
