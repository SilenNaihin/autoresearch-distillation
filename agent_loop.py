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

@register("autoresearch_agent")
class AutoresearchAgentLoop(ToolAgentLoop):
    """Multi-turn agent loop: bash editing → GPU experiment → reward.

    Extends ToolAgentLoop to:
      1. Keep a persistent BashTool instance per trajectory (not per tool call)
      2. After the model submits, dispatch modified train.py to GPU fleet
      3. Compute reward from experiment results
    """

    def __init__(self, *args, behavioral_feedback: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_val_bpb = float("inf")
        self._best_lock = threading.Lock()
        self._behavioral_feedback = behavioral_feedback

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Override run() to add pre-creation and post-submission logic."""
        # Reset per-trajectory tool call stats
        self._total_tool_calls = 0
        self._failed_tool_calls = 0  # malformed JSON, errors
        self._noop_tool_calls = 0    # commands that don't touch train.py

        # Pre-create persistent bash tool instance
        bash_tool = self.tools.get("bash")
        instance_id = None

        if bash_tool:
            instance_id, _ = await bash_tool.create()
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
            output.extra_fields["reward_extra_info"] = {"feedback": feedback, **env_metrics}

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
        if bash_tool is None or instance_id is None:
            return 0.0, "No bash tool or instance available."

        modified = bash_tool.read_train_py(instance_id)
        if modified is None:
            logger.warning(f"No modified train.py found for instance {instance_id}")
            return 0.0, "No modified train.py found."

        # Generate diff between baseline and modified train.py for teacher feedback
        baseline = Path(bash_tool.autoresearch_dir, "train.py").read_text()
        diff_text = _make_diff(baseline, modified)

        # Build behavioral notes for teacher feedback (gated by config)
        notes_text = ""
        if self._behavioral_feedback:
            notes = []
            if baseline == modified:
                notes.append("Warning: No changes were made to train.py before submitting.")
            if self._noop_tool_calls > 0:
                notes.append(f"Warning: {self._noop_tool_calls} of {self._total_tool_calls} tool calls did not modify train.py.")
            if self._failed_tool_calls > 0:
                notes.append(f"Warning: {self._failed_tool_calls} tool calls had malformed arguments.")
            if self._empty_thinking:
                notes.append("Warning: You did not use chain of thought. You must reason step-by-step inside <think> tags before making changes.")
            notes_text = "\n".join(notes)

        from environment import compute_reward, parse_metrics

        if baseline == modified:
            feedback = ("FAILURE: train.py was unchanged from the original. No experiment was run. Reward: 0.0.\n\n"
                        "You MUST modify train.py to lower val_bpb. "
                        "Use the bash tool with sed commands to make targeted edits, e.g.:\n"
                        "  sed -i 's/n_head: int = 6/n_head: int = 8/' train.py\n"
                        "  sed -i 's/sequence_len: int = 256/sequence_len: int = 512/' train.py\n\n"
                        "Do NOT echo or cat the entire file into train.py. "
                        "Make small, targeted changes with sed, then submit.")
            return 0.0, f"{feedback}\n\n{notes_text}" if notes_text else feedback

        # Dispatch to GPU fleet (blocking I/O → run in thread)
        pool = _get_pool()
        output = await asyncio.to_thread(pool.run, modified)

        # Parse whatever metrics are available (even on crash, stdout may have partial output)
        metrics = parse_metrics(output.stdout) if output.stdout else {}
        self._last_env_metrics = metrics

        if output.returncode != 0:
            # Combine stderr and stdout tail for maximum crash context.
            # Remote cmd uses 2>&1, so experiment errors are in stdout;
            # stderr is SSH-level errors only.
            parts = []
            if output.stderr and output.stderr.strip():
                parts.append(output.stderr.strip()[:1000])
            if output.stdout and output.stdout.strip():
                parts.append(output.stdout.strip()[-1000:])
            crash_info = "\n".join(parts) if parts else "no output"
            logger.warning(f"Experiment crashed (exit {output.returncode}): {crash_info}")
            feedback = f"Changes:\n{diff_text}\n\nExperiment crashed (exit {output.returncode}):\n{crash_info}"
            return 0.0, f"{feedback}\n\n{notes_text}" if notes_text else feedback
        val_bpb = metrics.get("val_bpb")

        if val_bpb is None:
            tail = "\n".join(output.stdout.strip().splitlines()[-20:]) if output.stdout else "empty output"
            logger.warning(f"No val_bpb in experiment output. Tail:\n{tail}")
            feedback = f"Changes:\n{diff_text}\n\nExperiment ran but produced no val_bpb metric. Output tail:\n{tail}"
            return 0.0, f"{feedback}\n\n{notes_text}" if notes_text else feedback

        with self._best_lock:
            reward, status, reward_feedback = compute_reward(val_bpb, self._best_val_bpb)
            if status == "improvement" and val_bpb is not None:
                self._best_val_bpb = val_bpb

        logger.info(f"Experiment: val_bpb={val_bpb}, reward={reward:.4f}, status={status}")
        feedback = f"Changes:\n{diff_text}\n\n{reward_feedback}"
        return reward, f"{feedback}\n\n{notes_text}" if notes_text else feedback
