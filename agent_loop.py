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
from typing import Any

# Ensure SDPO's verl is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SDPO"))

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Shared GPU pool (singleton, thread-safe)
# ---------------------------------------------------------------------------

_pool = None
_pool_lock = threading.Lock()


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_val_bpb = float("inf")
        self._best_lock = threading.Lock()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Override run() to add pre-creation and post-submission logic."""
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

            # Post-submission: dispatch experiment to GPU fleet
            reward = await self._dispatch_experiment(bash_tool, instance_id)
            output.reward_score = reward

            return output
        finally:
            # Clean up workdir
            if bash_tool and instance_id:
                await bash_tool.release(instance_id)

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
                return ToolResponse(text="Error: invalid JSON in tool arguments"), 0.0, {}

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
        else:
            # For any other tool, use default create/execute/release per call
            return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    async def _dispatch_experiment(self, bash_tool, instance_id: str | None) -> float:
        """Read modified train.py, dispatch to GPU fleet, compute reward."""
        if bash_tool is None or instance_id is None:
            return -1.0

        modified = bash_tool.read_train_py(instance_id)
        if modified is None:
            logger.warning(f"No modified train.py found for instance {instance_id}")
            return -1.0

        from environment import compute_reward, parse_metrics

        # Dispatch to GPU fleet (blocking I/O → run in thread)
        pool = _get_pool()
        output = await asyncio.to_thread(pool.run, modified)

        if output.returncode != 0:
            logger.warning(f"Experiment crashed (exit {output.returncode}): {output.stderr[:2000] if output.stderr else output.stdout[-2000:] if output.stdout else 'no output'}")
            return -1.0

        metrics = parse_metrics(output.stdout)
        val_bpb = metrics.get("val_bpb")

        with self._best_lock:
            reward, status, feedback = compute_reward(val_bpb, self._best_val_bpb)
            if status == "improvement" and val_bpb is not None:
                self._best_val_bpb = val_bpb

        logger.info(f"Experiment: val_bpb={val_bpb}, reward={reward:.4f}, status={status}")
        return reward
