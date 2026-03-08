"""
Custom agent loop for autoresearch SDPO training.

Each rollout: model generates a diff → ExperimentEnvironment dispatches it
to a remote GPU via GPUPoolRunner → ~5 min experiment → reward score returned.

Uses environment.py and runners.py for all experiment logic.
"""

import asyncio
import logging
import os
import sys
import threading
from typing import Any
from uuid import uuid4

# Ensure SDPO's verl is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SDPO"))

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Shared ExperimentEnvironment (singleton, thread-safe via GPUPoolRunner)
# ---------------------------------------------------------------------------

_env = None
_env_lock = threading.Lock()


def _get_env():
    """Lazily initialize the shared ExperimentEnvironment."""
    global _env
    if _env is None:
        with _env_lock:
            if _env is None:
                from environment import ExperimentEnvironment
                from runners import GPUPoolRunner

                pool = GPUPoolRunner()
                _env = ExperimentEnvironment(pool)
                logger.info(f"Initialized ExperimentEnvironment with {pool.total} GPU slots")
    return _env


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@register("autoresearch_agent")
class AutoresearchAgentLoop(AgentLoopBase):
    """Agent loop that generates diffs, runs real experiments on remote GPUs, and scores them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
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

        # 3. Run experiment via ExperimentEnvironment (blocking call → run in thread)
        env = _get_env()
        result = await asyncio.to_thread(env.step, response_text)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=result.reward,
            num_turns=2,
            metrics={},
        )
