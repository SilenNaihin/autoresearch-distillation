"""
Custom agent loop for SDPO training with multi-turn bash tool.

Each rollout:
  1. Model receives prompt (target file + experiment history)
  2. Model uses bash tool to read/edit target file (multi-turn via ToolAgentLoop state machine)
  3. Model submits via: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  4. We read modified target file from the isolated workdir
  5. We dispatch to GPU fleet via GPUPoolRunner
  6. We parse metrics, compute reward
  7. reward_score is set on AgentLoopOutput for RL training

Generic — all task-specific config comes from TaskConfig.

Extends VERL's ToolAgentLoop which implements the multi-turn state machine:
PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED

Tool response tokens are masked (response_mask=0), so we only train on
the model's bash commands, not environment output.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from lib.experiment_cache import ExperimentCache, cache_path_for, SDPO_CACHE
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
# Local evaluation
# ---------------------------------------------------------------------------


def _evaluate_locally(modified_code: str, task: TaskConfig) -> subprocess.CompletedProcess:
    """Run evaluation in an isolated temp dir. Returns CompletedProcess."""
    tmpdir = tempfile.mkdtemp(prefix="eval_")
    try:
        src_dir = Path(task.workspace.source_dir)
        for f in src_dir.iterdir():
            if f.name == "__pycache__":
                continue
            if f.is_file():
                shutil.copy2(f, tmpdir)
            elif f.is_dir():
                shutil.copytree(f, Path(tmpdir) / f.name,
                                ignore=shutil.ignore_patterns("__pycache__"))

        (Path(tmpdir) / task.workspace.target_file).write_text(modified_code)

        env = os.environ.copy()
        run_cmd = task.execution.run_command
        result = subprocess.run(
            run_cmd, shell=True, cwd=tmpdir, capture_output=True, text=True,
            timeout=task.execution.timeout, env=env,
        )
        return result
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args="", returncode=-1, stdout="", stderr="TIMEOUT"
        )
    except Exception as e:
        return subprocess.CompletedProcess(
            args="", returncode=-1, stdout="", stderr=str(e)
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

_buffer = None
_buffer_lock = threading.Lock()
_buffer_kwargs: dict = {}


def _get_buffer(task: TaskConfig) -> ReuseBuffer:
    """Lazily initialize the shared ReuseBuffer singleton."""
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                data_dir = os.environ.get("DATA_DIR", "/data")
                buf_path = Path(data_dir) / f"reuse_buffer_{task.name}.json"
                _buffer = ReuseBuffer(buf_path, direction=task.scoring.direction,
                                      **_buffer_kwargs)
    return _buffer


@register("autoresearch_agent")
class AutoresearchAgentLoop(ToolAgentLoop):
    """Multi-turn agent loop: bash editing → experiment dispatch → reward.

    Extends ToolAgentLoop to:
      1. Keep a persistent BashTool instance per trajectory (not per tool call)
      2. After the model submits, dispatch modified file to experiment fleet
      3. Compute reward from experiment results
    """

    def __init__(self, *args, c_puct: float = 1.0, max_states: int = 1000,
                 task_config: str = "tasks/autoresearch.yaml", **kwargs):
        super().__init__(*args, **kwargs)
        self.task = TaskConfig.from_yaml(task_config)

        global _buffer_kwargs
        _buffer_kwargs = {"c_puct": c_puct, "max_states": max_states}

        self._sed_failed: str | None = None
        self._cache = ExperimentCache(
            write_path=cache_path_for(self.task.name, "sdpo"),
            direction=self.task.scoring.direction,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Override run() to add pre-creation and post-submission logic."""
        global_step = kwargs.get("global_step")
        assert global_step is not None and global_step >= 0, (
            f"global_step not forwarded to agent loop (got {global_step!r}). "
            "Check VERL agent_loop.py forwards global_steps from batch.meta_info."
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

        # Store original baseline for cache keying (state 0 = seed)
        self._baseline_code = buffer._data["states"].get("0", {}).get(
            "content", buffer._data["states"].get("0", {}).get("code", ""))

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
            # Append best metric target if better than baseline
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

            # Detect empty chain of thought
            self._empty_thinking = False
            try:
                prefix = self.tokenizer.decode(output.response_ids[:50], skip_special_tokens=False)
                think_start = prefix.find("<think>")
                think_end = prefix.find("</think>")
                if think_start >= 0 and think_end > think_start:
                    think_content = prefix[think_start + len("<think>"):think_end].strip()
                    self._empty_thinking = len(think_content) < 20
            except Exception:
                pass

            # Post-submission: dispatch experiment
            reward, feedback = await self._dispatch_experiment(bash_tool, instance_id)
            output.reward_score = reward

            # Pack env metrics for wandb logging
            env_metrics = {}
            for k in task.scoring.metrics:
                env_metrics[f"env_{k}"] = float(getattr(self, '_last_env_metrics', {}).get(k, float('nan')))
            env_metrics["env_novel"] = self._is_novel
            output.extra_fields["reward_extra_info"] = {"feedback": feedback, "parent_id": int(self._parent_id), **env_metrics}

            # Store modified raw_prompt so teacher prompt uses the same file version
            if "raw_prompt" in kwargs:
                output.extra_fields["agent_raw_prompt"] = kwargs["raw_prompt"]

            return output
        finally:
            if bash_tool and instance_id:
                await bash_tool.release(instance_id)

    async def _handle_processing_tools_state(self, agent_data):
        """Override to terminate the loop after the model submits."""
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
        """Override to reuse persistent bash tool instance across calls."""
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

                # Detect sed failures
                resp_text = (response.text or "").strip()
                if cmd_stripped == "sed" and "sed:" in resp_text:
                    self._sed_failed = resp_text

                # Truncate long output
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
        """Read modified file, dispatch to fleet, compute reward.

        Returns (reward, feedback) tuple. Feedback is passed through to SDPO's
        self-distillation pipeline via extra_fields["reward_extra_info"]["feedback"].
        """
        self._is_novel = 0.0
        task = self.task
        metric_name = task.scoring.metric

        if bash_tool is None or instance_id is None:
            return 0.0, "No bash tool or instance available."

        modified = bash_tool.read_target_file(instance_id)
        if modified is None:
            logger.warning(f"No modified {task.workspace.target_file} found for instance {instance_id}")
            return 0.0, f"No modified {task.workspace.target_file} found."

        # Two diffs: cache_diff keys on original baseline, feedback_diff shows changes from selected parent
        cache_diff = task.make_diff(self._baseline_code, modified)
        feedback_diff = task.make_diff(self._selected_code, modified)

        if self._selected_code == modified:
            if self._sed_failed:
                feedback = task.fmt_feedback("sed_failed", error=self._sed_failed)
            else:
                feedback = task.feedback.no_change
            return 0.0, feedback

        # Check reuse buffer — if this exact content is already known, skip dispatch
        buffer = _get_buffer(task)
        known = buffer.find_by_content(modified)
        if known is not None:
            mv = known.get("metric_value", known.get("val_bpb"))
            feedback = (f"Changes from previous attempt:\n{feedback_diff}\n\n"
                        f"{task.fmt_feedback('duplicate', value=f'{mv:.6f}')}")
            return known["reward"], feedback

        # Check experiment cache
        cached = self._cache.get(cache_diff, current_step=self._global_step)
        if cached is not None:
            if cached.get("crashed"):
                feedback = (f"Changes from previous attempt:\n{feedback_diff}\n\n"
                            f"These changes caused the experiment to crash:\n{cached['tail']}\n\n"
                            f"{task.feedback.crash}")
                return 0.0, feedback
            val = cached.get("val_bpb", cached.get("metric_value"))
            if val is not None:
                reward, status, reward_feedback = task.compute_reward(val)
                metrics_line = cached.get("metrics_line", "")
                deg = task.check_degradation(val)
                if deg:
                    reward_feedback += f" {deg}"
                feedback = f"Changes from previous attempt:\n{feedback_diff}\n\n{reward_feedback}\n{metrics_line}"
                if reward > 0:
                    if buffer.find_by_content(modified) is None:
                        buffer.add(modified, val, reward, self._parent_id)
                    return reward, f"{feedback}\n\n{task.feedback.success}"
                return reward, f"{feedback}\n\n{task.feedback.failure}"

        # Evaluate locally (blocking I/O → run in thread)
        self._is_novel = 1.0
        output = await asyncio.to_thread(_evaluate_locally, modified, task)

        # Parse metrics
        metrics = task.parse_metrics(output.stdout) if output.stdout else {}
        self._last_env_metrics = metrics

        if output.returncode != 0:
            parts = []
            if output.stderr and output.stderr.strip():
                parts.append(output.stderr.strip()[:1000])
            if output.stdout and output.stdout.strip():
                parts.append(output.stdout.strip()[-1000:])
            crash_info = "\n".join(parts) if parts else "no output"
            logger.warning(f"Experiment crashed (exit {output.returncode}): {crash_info}")
            feedback = (f"Changes from previous attempt:\n{feedback_diff}\n\n"
                        f"These changes caused the experiment to crash (exit {output.returncode}):\n{crash_info}")
            if task.is_cacheable_crash(crash_info):
                self._cache.put(cache_diff, {"crashed": True, "tail": crash_info}, step=self._global_step)
            return 0.0, f"{feedback}\n\n{task.feedback.crash}"

        val = metrics.get(metric_name)

        if val is None:
            tail = "\n".join(output.stdout.strip().splitlines()[-20:]) if output.stdout else "empty output"
            logger.warning(f"No {metric_name} in experiment output. Tail:\n{tail}")
            feedback = (f"Changes from previous attempt:\n{feedback_diff}\n\n"
                        f"We were not able to run your experiment. Output tail:\n{tail}")
            return 0.0, feedback

        reward, status, reward_feedback = task.compute_reward(val)

        logger.info(f"Experiment: {metric_name}={val}, reward={reward:.4f}, status={status}")
        display = task.scoring.display_metrics
        metrics_line = " | ".join(f"{k}: {metrics[k]:g}" for k in display if k in metrics)
        deg = task.check_degradation(val)
        if deg:
            reward_feedback += f" {deg}"

        feedback = f"Changes from previous attempt:\n{feedback_diff}\n\n{reward_feedback}\n{metrics_line}"

        # Cache for future dedup
        self._cache.put(cache_diff, {"val_bpb": val, "metric_value": val, "metrics_line": metrics_line},
                        step=self._global_step, metric_value=val, diff_text_raw=cache_diff)
        if reward > 0:
            buffer.add(modified, val, reward, self._parent_id)
            return reward, f"{feedback}\n\n{task.feedback.success}"
        return reward, f"{feedback}\n\n{task.feedback.failure}"
