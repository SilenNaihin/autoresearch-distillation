"""
Power Sampling: MCMC sampling from p^α for autoregressive LLMs.

Based on "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
(Karan & Du, 2025). https://arxiv.org/abs/2510.14901

Uses vLLM's OpenAI-compatible chat completions API with logprobs.
With temperature=1 proposal, the Metropolis-Hastings acceptance ratio simplifies to:
    log_ratio = (α - 1) * [Σ log_p(new_suffix) - Σ log_p(old_suffix)]

Algorithm:
    1. Generate complete initial sequence (with thinking if available)
    2. Find the </think> boundary — MCMC only resamples tokens AFTER it
    3. For each of block_num blocks:
       a. Pick a random resample position in the post-think region
       b. For each of mcmc_steps MH steps:
          - Resample everything from that position onward
          - Accept/reject via Metropolis-Hastings ratio
    Total MCMC API calls = block_num × mcmc_steps

Requirements:
    - vLLM server with continue_final_message support (v0.4+)
    - openai Python client
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class PowerSampleResult:
    text: str
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    acceptance_rate: float = 0.0
    total_tokens_generated: int = 0
    wall_time: float = 0.0
    think_end_idx: int = 0


def _chat_generate(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 1.0,
    stop: list[str] | None = None,
    continue_final: bool = False,
) -> tuple[list[str], list[float]]:
    """Generate via chat completions with logprobs.

    Returns (token_strings, base_model_logprobs).
    """
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        logprobs=True,
        top_logprobs=1,
    )
    if stop:
        kwargs["stop"] = stop

    extra_body: dict = {}
    if continue_final:
        extra_body["continue_final_message"] = True
        extra_body["add_generation_prompt"] = False
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]

    if choice.logprobs is None or not choice.logprobs.content:
        return [], []

    tokens = [item.token for item in choice.logprobs.content]
    lps = [item.logprob for item in choice.logprobs.content]
    return tokens, lps


def _build_messages(system_prompt: str, user_prompt: str, assistant_prefix: str = ""):
    """Build chat messages, optionally with a partial assistant response."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_prefix:
        msgs.append({"role": "assistant", "content": assistant_prefix})
    return msgs


def _find_think_end(tokens: list[str]) -> int:
    """Find the token index right after </think> in the token sequence.

    Returns 0 if no </think> found (no thinking block present).
    """
    text = ""
    for i, tok in enumerate(tokens):
        text += tok
        if "</think>" in text:
            return i + 1
    return 0


def best_of_n(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    n: int = 8,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    stop: list[str] | None = None,
) -> PowerSampleResult:
    """Generate N completions, return the one with highest average log-likelihood."""
    t0 = time.time()
    best_result = None
    best_score = float("-inf")
    total_gen = 0

    messages = _build_messages(system_prompt, user_prompt)

    for _ in range(n):
        tokens, logprobs = _chat_generate(
            client, model, messages, max_tokens, temperature, stop
        )
        total_gen += len(tokens)
        if not tokens:
            continue
        score = sum(logprobs) / len(logprobs)
        if score > best_score:
            best_score = score
            best_result = PowerSampleResult(
                text="".join(tokens),
                tokens=tokens,
                logprobs=logprobs,
            )

    if best_result is None:
        return PowerSampleResult(text="")

    best_result.total_tokens_generated = total_gen
    best_result.acceptance_rate = 1.0 / n
    best_result.wall_time = time.time() - t0
    return best_result


def power_sample(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    alpha: float = 4.0,
    max_tokens: int = 32768,
    block_num: int = 16,
    mcmc_steps: int = 10,
    temperature: float = 1.0,
    stop: list[str] | None = None,
) -> PowerSampleResult:
    """Block-wise MCMC power sampling from p^α (Algorithm 1).

    1. Generate a complete initial sequence with the full max_tokens budget.
       If the model uses <think>...</think>, the thinking block is preserved
       and MCMC only resamples tokens AFTER </think>.
    2. Refine via block_num rounds of MCMC, each with mcmc_steps MH proposals.
       Each block picks a random resample position in the post-think region.

    With temperature=1 proposal, the MH acceptance ratio simplifies to:
        log_ratio = (α - 1) * [Σ log_p(new) - Σ log_p(old)]

    Args:
        client: OpenAI-compatible client pointing at vLLM.
        model: Model name (e.g. "Qwen/Qwen3-14B").
        system_prompt: System message.
        user_prompt: User message (train.py + feedback).
        alpha: Power exponent (higher = sharper). Paper uses 4.0.
        max_tokens: Max tokens per generation (full context budget).
        block_num: Number of MCMC refinement rounds. Paper uses 16.
        mcmc_steps: MH steps per round. Paper uses 10.
        temperature: Proposal temperature. 1.0 gives exact MH ratio.
        stop: Stop sequences.
    """
    if temperature != 1.0:
        import warnings
        warnings.warn(
            f"temperature={temperature} but MH ratio assumes temperature=1.0. "
            f"The paper uses temp=1/α={1/alpha:.2f} with the full 4-term MH ratio "
            f"(requires temperature-scaled logprobs, not available via vLLM API). "
            f"With temp≠1, proposals may be better but acceptance ratio is approximate.",
            stacklevel=2,
        )

    t0 = time.time()

    total_generated = 0
    total_accepted = 0
    total_proposed = 0

    # --- Step 1: Generate complete initial sequence (with thinking) ---
    msgs = _build_messages(system_prompt, user_prompt)
    suffix_tokens, suffix_logprobs = _chat_generate(
        client, model, msgs, max_tokens, temperature, stop,
    )
    total_generated += len(suffix_tokens)

    if not suffix_tokens:
        return PowerSampleResult(text="", wall_time=time.time() - t0)

    seq_len = len(suffix_tokens)

    # --- Find </think> boundary ---
    # MCMC only resamples tokens after this index, preserving the thinking block.
    think_end = _find_think_end(suffix_tokens)
    resample_start = think_end  # first token eligible for resampling

    # --- Step 2: Block-wise MCMC refinement (post-think only) ---
    for block_idx in range(block_num):
        # Need at least 1 token after think block to resample
        if seq_len <= resample_start:
            break

        # Pick a random resample position AFTER the thinking block
        idx = random.randint(resample_start, seq_len - 1)
        remaining = seq_len - idx

        for step in range(mcmc_steps):
            # Build prefix for proposal (always includes full <think>...</think>)
            prop_prefix = "".join(suffix_tokens[:idx])
            if prop_prefix:
                prop_msgs = _build_messages(system_prompt, user_prompt, prop_prefix)
                prop_cont = True
            else:
                prop_msgs = _build_messages(system_prompt, user_prompt)
                prop_cont = False

            try:
                prop_tokens, prop_logprobs = _chat_generate(
                    client, model, prop_msgs, remaining, temperature, stop,
                    continue_final=prop_cont,
                )
            except Exception:
                # Chat template may reject certain prefixes. Skip this proposal.
                continue
            total_generated += len(prop_tokens)
            total_proposed += 1

            if not prop_tokens:
                continue

            # Compare over overlapping length
            cmp_len = min(len(prop_tokens), remaining)

            old_lps = suffix_logprobs[idx : idx + cmp_len]
            new_lps = prop_logprobs[:cmp_len]

            old_sum = sum(old_lps)
            new_sum = sum(new_lps)

            # MH ratio (temp=1 proposal simplification)
            log_ratio = (alpha - 1.0) * (new_sum - old_sum)

            u = random.random()
            if u == 0:
                u = 1e-30
            if math.log(u) < log_ratio:
                suffix_tokens = suffix_tokens[:idx] + list(prop_tokens[:cmp_len])
                suffix_logprobs = suffix_logprobs[:idx] + list(new_lps)
                seq_len = len(suffix_tokens)
                remaining = seq_len - idx
                total_accepted += 1

    acceptance_rate = total_accepted / max(total_proposed, 1)
    output_text = "".join(suffix_tokens)

    return PowerSampleResult(
        text=output_text,
        tokens=suffix_tokens,
        logprobs=suffix_logprobs,
        acceptance_rate=acceptance_rate,
        total_tokens_generated=total_generated,
        wall_time=time.time() - t0,
        think_end_idx=think_end,
    )
