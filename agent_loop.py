"""
Custom agent loop for autoresearch SDPO training with multi-turn bash tool.

Each rollout:
  1. Model receives prompt (train.py + experiment history)
  2. Model uses bash tool to read/edit train.py (multi-turn via ToolAgentLoop state machine)
  3. Model submits via: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  4. We read modified train.py from the isolated workdir
  5. We dispatch to GPU fleet via GPUPoolRunner
  6. We parse metrics, compute reward
  7. reward_score is set on AgentLoopOutput for RL training

Extends VERL's ToolAgentLoop which implements the multi-turn state machine:
PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED

Tool response tokens are masked (response_mask=0), so we only train on
the model's bash commands, not environment output.
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

from experiment_cache import SDPO_CACHE, ExperimentCache
from reuse_buffer import ReuseBuffer

# Ensure SDPO's verl is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SDPO"))

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Shared GPU pool (singleton, thread-safe)
# ---------------------------------------------------------------------------

_pool = None
_pool_lock = threading.Lock()


def _make_diff(baseline: str, modified: str) -> str:
    """Generate a unified diff between baseline and modified train.py."""
    if baseline == modified:
        return "No changes were made to train.py."
    return "".join(unified_diff(
        baseline.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="a/train.py", tofile="b/train.py",
    ))


def _get_pool():
    """Lazily initialize the shared GPUPoolRunner."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from runners import GPUPoolRunner
                _pool = GPUPoolRunner()
                logger.info(f"Initialized GPUPoolRunner with {_pool.total} GPU slots")
    return _pool


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

_buffer = None
_buffer_lock = threading.Lock()
_buffer_kwargs: dict = {}  # set from agent loop __init__


def _get_buffer() -> ReuseBuffer:
    """Lazily initialize the shared ReuseBuffer singleton."""
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                _buffer = ReuseBuffer(Path("/data/reuse_buffer.json"), **_buffer_kwargs)
    return _buffer


@register("autoresearch_agent")
class AutoresearchAgentLoop(ToolAgentLoop):
    """Multi-turn agent loop: bash editing → GPU experiment → reward.

    Extends ToolAgentLoop to:
      1. Keep a persistent BashTool instance per trajectory (not per tool call)
      2. After the model submits, dispatch modified train.py to GPU fleet
      3. Compute reward from experiment results
    """

    def __init__(self, *args, c_puct: float = 1.0, max_states: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        global _buffer_kwargs
        _buffer_kwargs = {"c_puct": c_puct, "max_states": max_states}

        self._sed_failed: str | None = None  # tracked but no longer triggers early termination
        self._cache = ExperimentCache(write_path=SDPO_CACHE)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Override run() to add pre-creation and post-submission logic."""
        # Validate that VERL forwarded the step number
        global_step = kwargs.get("global_step")
        assert global_step is not None and global_step >= 0, (
            f"global_step not forwarded to agent loop (got {global_step!r}). "
            "Check VERL agent_loop.py forwards global_steps from batch.meta_info."
        )
        self._global_step = global_step

        # Reset per-trajectory tool call stats
        self._total_tool_calls = 0
        self._failed_tool_calls = 0  # malformed JSON, errors
        self._noop_tool_calls = 0    # commands that don't touch train.py
        self._sed_failed = None

        # Select a state from the reuse buffer for this rollout
        from prompts import replace_train_py_in_prompt
        from environment import BASELINE_VAL_BPB

        buffer = _get_buffer()
        # Seed with baseline train.py if buffer is empty
        bash_tool = self.tools.get("bash")
        if bash_tool and len(buffer) == 0:
            baseline_code = Path(bash_tool.autoresearch_dir, "train.py").read_text()
            buffer.seed(baseline_code, BASELINE_VAL_BPB)

        selections = buffer.select(1)
        if selections:
            self._parent_id, self._selected_code = selections[0]
        else:
            # Fallback: use baseline directly
            self._parent_id = 0
            self._selected_code = Path(bash_tool.autoresearch_dir, "train.py").read_text() if bash_tool else ""

        # Replace train.py in prompt with the selected state's code
        if "raw_prompt" in kwargs:
            kwargs["raw_prompt"] = list(kwargs["raw_prompt"])  # don't mutate dataset
            last_msg = dict(kwargs["raw_prompt"][-1])
            last_msg["content"] = replace_train_py_in_prompt(last_msg["content"], self._selected_code)
            # Append best val_bpb target if better than baseline
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
            # Store instance_id where _call_tool can find it
            kwargs.setdefault("tools_kwargs", {})
            kwargs["tools_kwargs"]["_bash_instance_id"] = instance_id

        try:
            # Run the standard ToolAgentLoop state machine
            output = await super().run(sampling_params, **kwargs)

            # Detect empty chain of thought (e.g. "<think>\n</think>")
            self._empty_thinking = False
            try:
                # Decode just the first 50 tokens to check for thinking
                prefix = self.tokenizer.decode(output.response_ids[:50], skip_special_tokens=False)
                think_start = prefix.find("<think>")
                think_end = prefix.find("</think>")
                if think_start >= 0 and think_end > think_start:
                    think_content = prefix[think_start + len("<think>"):think_end].strip()
                    self._empty_thinking = len(think_content) < 20
            except Exception:
                pass

            # Post-submission: dispatch experiment to GPU fleet
            reward, feedback = await self._dispatch_experiment(bash_tool, instance_id)
            output.reward_score = reward
            # Pack env metrics from the last dispatch for wandb logging
            env_metrics = {}
            for k in ("val_bpb", "peak_vram_mb", "training_seconds", "total_seconds",
                      "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth"):
                env_metrics[f"env_{k}"] = float(getattr(self, '_last_env_metrics', {}).get(k, float('nan')))
            env_metrics["env_novel"] = self._is_novel
            output.extra_fields["reward_extra_info"] = {"feedback": feedback, "parent_id": int(self._parent_id), **env_metrics}

            return output
        finally:
            # Clean up workdir
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

    _TRAIN_PY_CMDS = ("sed", "cat", "echo", "python", "python3", "printf", "tee", "patch")

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
                # Track commands that likely don't modify train.py
                cmd_stripped = cmd.strip().split()[0] if cmd.strip() else ""
                if cmd_stripped and cmd_stripped not in self._TRAIN_PY_CMDS and "train.py" not in cmd:
                    self._noop_tool_calls += 1

                response, reward, metrics = await bash_tool.execute(
                    instance_id, tool_args, agent_data=agent_data
                )

                # Detect sed failures (no input files, bad expression, etc.)
                # All sed errors produce output starting with "sed:"; success is silent.
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
            # For any other tool, use default create/execute/release per call
            return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    async def _dispatch_experiment(self, bash_tool, instance_id: str | None) -> tuple[float, str]:
        """Read modified train.py, dispatch to GPU fleet, compute reward.

        Returns (reward, feedback) tuple. Feedback is passed through to SDPO's
        self-distillation pipeline via extra_fields["reward_extra_info"]["feedback"].
        """
        self._is_novel = 0.0

        if bash_tool is None or instance_id is None:
            return 0.0, "No bash tool or instance available."

        modified = bash_tool.read_train_py(instance_id)
        if modified is None:
            logger.warning(f"No modified train.py found for instance {instance_id}")
            return 0.0, "No modified train.py found."

        # Diff against the selected parent state (not fixed baseline)
        diff_text = _make_diff(self._selected_code, modified)

        from environment import BASELINE_VAL_BPB, compute_reward, parse_metrics

        if self._selected_code == modified:
            if self._sed_failed:
                feedback = (f"FAILURE: your sed command failed: {self._sed_failed}\n\n"
                            "You must specify the target file and ensure the pattern matches exactly.")
            else:
                feedback = "No changes were made to train.py. You must make new creative edits to reduce validation loss."
            return 0.0, feedback

        # Check experiment cache — avoid re-dispatching identical diffs
        cached = self._cache.get(diff_text, current_step=self._global_step)
        if cached is not None:
            reward, feedback = cached["reward"], cached["feedback"]
            if cached.get("crashed"):
                feedback += "\n\nThis exact set of changes has been tried before and crashed. Do not re-attempt this change."
            elif reward > 0:
                feedback += "\n\nSUCCESS: These changes reduced validation loss. Combine with additional modifications."
            else:
                feedback += "\n\nThis exact set of changes has been tried before and did not reduce validation loss."
            return reward, feedback

        # Dispatch to GPU fleet (blocking I/O → run in thread)
        self._is_novel = 1.0
        pool = _get_pool()
        output = await asyncio.to_thread(pool.run, modified)

        # Parse whatever metrics are available (even on crash, stdout may have partial output)
        metrics = parse_metrics(output.stdout) if output.stdout else {}
        self._last_env_metrics = metrics

        if output.returncode != 0:
            parts = []
            if output.stderr and output.stderr.strip():
                parts.append(output.stderr.strip()[:1000])
            if output.stdout and output.stdout.strip():
                parts.append(output.stdout.strip()[-1000:])
            crash_info = "\n".join(parts) if parts else "no output"
            logger.warning(f"Experiment crashed (exit {output.returncode}): {crash_info}")
            feedback = (f"Changes from previous attempt:\n{diff_text}\n\n"
                        f"These changes caused the experiment to crash (exit {output.returncode}):\n{crash_info}")
            # Cache CUDA OOMs (deterministic), but not transient crashes (SSH, segfault)
            if "CUDA out of memory" in crash_info:
                self._cache.put(diff_text, {"reward": 0.0, "feedback": feedback, "crashed": True}, step=self._global_step)
            return 0.0, f"{feedback}\n\nDo not re-attempt this change."

        val_bpb = metrics.get("val_bpb")

        if val_bpb is None:
            tail = "\n".join(output.stdout.strip().splitlines()[-20:]) if output.stdout else "empty output"
            logger.warning(f"No val_bpb in experiment output. Tail:\n{tail}")
            feedback = (f"Changes from previous attempt:\n{diff_text}\n\n"
                        f"We were not able to run your experiment. Output tail:\n{tail}")
            return 0.0, feedback

        reward, status, reward_feedback = compute_reward(val_bpb)

        logger.info(f"Experiment: val_bpb={val_bpb}, reward={reward:.4f}, status={status}")
        metrics_line = " | ".join(f"{k}: {metrics[k]:g}" for k in ("num_steps", "num_params_M", "peak_vram_mb", "mfu_percent"))
        degradation = val_bpb - BASELINE_VAL_BPB
        if degradation > 0.05:
            reward_feedback += f" These changes made validation loss significantly worse (degraded by {degradation:.6f})."

        feedback = f"Changes from previous attempt:\n{diff_text}\n\n{reward_feedback}\n{metrics_line}"

        # Cache result and add successful states to the reuse buffer
        self._cache.put(diff_text, {"reward": reward, "feedback": feedback},
                        step=self._global_step, val_bpb=val_bpb, diff_text_raw=diff_text)
        if reward > 0:
            buffer = _get_buffer()
            buffer.add(modified, val_bpb, reward, self._parent_id)
            return reward, f"{feedback}\n\nSUCCESS: These changes reduced validation loss. This is a good result. Repeat this approach and combine it with additional modifications to reduce validation loss further."
        return reward, f"{feedback}\n\nThese changes did not reduce validation loss. Continue exploring new and creative modifications that reduce validation loss."
