"""
Power Sampling: MCMC sampling from p^α for autoregressive LLMs.

Based on "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
(Karan & Du, 2025). https://arxiv.org/abs/2510.14901

Uses vLLM's OpenAI-compatible chat completions API with logprobs.
vLLM always returns base model logprobs log_p regardless of sampling temperature,
so we can use low-temperature proposals (τ=1/α) while still computing the full
MH acceptance ratio by approximating log_q from top-k base logprobs:

    log_ratio = α·[Σlog_p(new) - Σlog_p(old)] + [Σlog_q(old) - Σlog_q(new)]

where log_q is approximated via: log_q(token) ≈ log_p(token)/τ - logsumexp(top_k_log_p/τ)

Algorithm (constrained MCMC with biased resampling):
    1. Generate complete initial sequence (with thinking if available)
    2. For each of block_num blocks:
       a. Pick a random resample position biased toward reasoning conclusion + output
       b. For each of mcmc_steps MH steps:
          - Resample everything from that position onward (at temperature τ=1/α)
          - If validator provided, reject structurally invalid proposals
          - Accept/reject valid proposals via full MH ratio
    Total MCMC API calls = block_num × mcmc_steps

Requirements:
    - vLLM server with continue_final_message support (v0.4+)
    - openai Python client
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import OpenAI

# Number of top logprobs to request for log_q approximation.
# At τ=0.25, top-20 captures >99.9% of the probability mass.
_TOP_K = 20


@dataclass
class PowerSampleResult:
    text: str
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    acceptance_rate: float = 0.0
    validity_rate: float = 0.0
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
) -> tuple[list[str], list[float], list[list[float]]]:
    """Generate via chat completions with logprobs.

    Returns (token_strings, base_model_logprobs, top_k_base_logprobs_per_position).
    vLLM always returns base model logprobs regardless of sampling temperature.
    """
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        logprobs=True,
        top_logprobs=_TOP_K,
    )
    if stop:
        kwargs["stop"] = stop

    extra_body: dict = {}
    if continue_final:
        extra_body["continue_final_message"] = True
        extra_body["add_generation_prompt"] = False
    if extra_body:
        kwargs["extra_body"] = extra_body

    import subprocess, json as _json
    hard_timeout = 900  # 15 min hard wall-clock limit per attempt
    for attempt in range(5):
        try:
            # Use subprocess to guarantee clean timeout — no zombie threads/connections.
            # The child process makes the API call and prints JSON result to stdout.
            script = f"""
import json, sys
from openai import OpenAI
import httpx
client = OpenAI(
    base_url="{client.base_url}",
    api_key="dummy",
    timeout=httpx.Timeout(connect=60, read=600, write=60, pool=60),
)
resp = client.chat.completions.create(**json.loads(sys.stdin.read()))
# Extract what we need
items = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
result = {{
    "tokens": [it.token for it in items],
    "logprobs": [it.logprob for it in items],
    "top_logprobs": [[t.logprob for t in it.top_logprobs] for it in items],
}}
print(json.dumps(result))
"""
            proc = subprocess.run(
                ["python3", "-c", script],
                input=_json.dumps(kwargs),
                capture_output=True, text=True, timeout=hard_timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"subprocess failed: {proc.stderr[-500:]}")
            data = _json.loads(proc.stdout)
            tokens = data["tokens"]
            lps = data["logprobs"]
            top_lps = data["top_logprobs"]
            return tokens, lps, top_lps
        except subprocess.TimeoutExpired:
            if attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"    _chat_generate attempt {attempt+1} hard timeout ({hard_timeout}s); retrying in {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise TimeoutError(f"_chat_generate failed after {attempt+1} attempts (hard timeout)")
        except Exception as exc:
            if attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"    _chat_generate attempt {attempt+1} failed: {exc}; retrying in {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise
    return [], [], []


def _approx_log_q(log_p_token: float, top_base_lps: list[float], temperature: float) -> float:
    """Approximate log_q(token) from top-k base model logprobs.

    log_q(token) = log_p(token)/τ - logsumexp(top_k_log_p / τ)

    At low τ (e.g. 0.25), the softmax is very peaked so top-20
    captures nearly all the probability mass, making this tight.
    """
    scaled = [lp / temperature for lp in top_base_lps]
    # Include the sampled token if not already in top-k
    scaled_token = log_p_token / temperature
    if scaled_token not in scaled:
        scaled.append(scaled_token)
    max_s = max(scaled)
    log_Z = max_s + math.log(sum(math.exp(s - max_s) for s in scaled))
    return scaled_token - log_Z


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
        tokens, logprobs, _top_lps = _chat_generate(
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
    temperature: float | None = None,
    stop: list[str] | None = None,
    validator: Callable[[str], bool] | None = None,
) -> PowerSampleResult:
    """Block-wise constrained MCMC power sampling from p^α.

    1. Generate a complete initial sequence with the full max_tokens budget.
    2. Refine via block_num rounds of MCMC, each with mcmc_steps MH proposals.
       Resample positions are biased toward the last 30% of the think block
       onward. Proposals are generated at temperature τ=1/α (paper's optimal).
       Proposals that fail structural validation are rejected before MH.

    Full MH acceptance ratio using approximate log_q:
        log_ratio = α·[Σlog_p(new) - Σlog_p(old)] + [Σlog_q(old) - Σlog_q(new)]

    Args:
        client: OpenAI-compatible client pointing at vLLM.
        model: Model name (e.g. "Qwen/Qwen3-14B").
        system_prompt: System message.
        user_prompt: User message (train.py + feedback).
        alpha: Power exponent (higher = sharper). Paper uses 4.0.
        max_tokens: Max tokens per generation (full context budget).
        block_num: Number of MCMC refinement rounds. Paper uses 16.
        mcmc_steps: MH steps per round. Paper uses 10.
        temperature: Proposal temperature. Defaults to 1/α (paper's optimal).
        stop: Stop sequences.
        validator: Optional callable(text) -> bool. Rejects structurally
            invalid proposals before MH ratio (constrained MCMC).
    """
    if temperature is None:
        temperature = 1.0 / alpha

    t0 = time.time()

    total_generated = 0
    total_accepted = 0
    total_proposed = 0
    total_invalid = 0

    # --- Step 1: Generate complete initial sequence (with thinking) ---
    msgs = _build_messages(system_prompt, user_prompt)
    suffix_tokens, suffix_logprobs, suffix_top_lps = _chat_generate(
        client, model, msgs, max_tokens, temperature, stop,
    )
    total_generated += len(suffix_tokens)
    print(f"    Initial generation: {len(suffix_tokens)} tokens in {time.time() - t0:.1f}s", flush=True)

    if not suffix_tokens:
        return PowerSampleResult(text="", wall_time=time.time() - t0)

    seq_len = len(suffix_tokens)

    # --- Find </think> boundary ---
    think_end = _find_think_end(suffix_tokens)

    # Biased resampling: start from last 30% of think block onward.
    resample_start = max(0, int(think_end * 0.7))
    print(f"    think_end={think_end}, resample_start={resample_start}, seq_len={seq_len}", flush=True)

    # --- Step 2: Block-wise constrained MCMC refinement ---
    for block_idx in range(block_num):
        if seq_len <= resample_start:
            break

        # Pick a random resample position (biased toward conclusion + output)
        idx = random.randint(resample_start, seq_len - 1)
        remaining = seq_len - idx

        block_t0 = time.time()
        for step in range(mcmc_steps):
            prop_prefix = "".join(suffix_tokens[:idx])
            if prop_prefix:
                prop_msgs = _build_messages(system_prompt, user_prompt, prop_prefix)
                prop_cont = True
            else:
                prop_msgs = _build_messages(system_prompt, user_prompt)
                prop_cont = False

            try:
                prop_tokens, prop_logprobs, prop_top_lps = _chat_generate(
                    client, model, prop_msgs, remaining, temperature, stop,
                    continue_final=prop_cont,
                )
            except Exception as exc:
                print(f"    MCMC block {block_idx} step {step}: proposal failed: {exc}")
                continue
            total_generated += len(prop_tokens)
            total_proposed += 1

            if not prop_tokens:
                continue

            # Compare over overlapping length
            cmp_len = min(len(prop_tokens), remaining)

            # Structural validation (constrained MCMC)
            if validator is not None:
                proposed_text = "".join(suffix_tokens[:idx]) + "".join(prop_tokens[:cmp_len])
                if not validator(proposed_text):
                    total_invalid += 1
                    continue

            old_lps = suffix_logprobs[idx : idx + cmp_len]
            new_lps = prop_logprobs[:cmp_len]

            # Full 4-term MH ratio with approximate log_q
            old_sum_p = sum(old_lps)
            new_sum_p = sum(new_lps)

            old_sum_q = sum(
                _approx_log_q(old_lps[i], suffix_top_lps[idx + i], temperature)
                for i in range(cmp_len)
            )
            new_sum_q = sum(
                _approx_log_q(new_lps[i], prop_top_lps[i], temperature)
                for i in range(cmp_len)
            )

            log_ratio = alpha * (new_sum_p - old_sum_p) + (old_sum_q - new_sum_q)

            u = random.random()
            if u == 0:
                u = 1e-30
            if math.log(u) < log_ratio:
                suffix_tokens = suffix_tokens[:idx] + list(prop_tokens[:cmp_len])
                suffix_logprobs = suffix_logprobs[:idx] + list(new_lps)
                suffix_top_lps = suffix_top_lps[:idx] + list(prop_top_lps[:cmp_len])
                seq_len = len(suffix_tokens)
                remaining = seq_len - idx
                total_accepted += 1
                # Update think_end since sequence changed
                think_end = _find_think_end(suffix_tokens)

        print(f"    Block {block_idx}/{block_num}: idx={idx} accepted={total_accepted}/{total_proposed} invalid={total_invalid} ({time.time()-block_t0:.1f}s)", flush=True)

    total_valid = total_proposed - total_invalid
    acceptance_rate = total_accepted / max(total_valid, 1)
    validity_rate = total_valid / max(total_proposed, 1)
    output_text = "".join(suffix_tokens)

    return PowerSampleResult(
        text=output_text,
        tokens=suffix_tokens,
        logprobs=suffix_logprobs,
        acceptance_rate=acceptance_rate,
        validity_rate=validity_rate,
        total_tokens_generated=total_generated,
        wall_time=time.time() - t0,
        think_end_idx=think_end,
    )
