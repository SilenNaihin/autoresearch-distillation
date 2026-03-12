# Power Sampling Ablation

Ablation baseline using MCMC power sampling from
["Reasoning with Sampling"](https://arxiv.org/abs/2510.14901) (Karan & Du, 2025).

## What is this

The paper shows that RL post-training (GRPO, SDPO, etc.) primarily *sharpens* the
base model distribution — upweighting high-likelihood reasoning traces that the model
could already generate. They propose sampling from **p^α** (the power distribution)
using MCMC at inference time, matching RL gains with zero training.

We add this as a third baseline alongside:

| Method | Weight updates | Inference cost | Memory |
|--------|---------------|----------------|--------|
| **ICL baseline** (`loop_baseline.py`) | None | 1x | Context window (feedback history) |
| **Power sampling** (`loop_power_sampling.py`) | None | ~9x | None (stateless per turn) |
| **SDPO** | Yes (RL) | 1x | Weights |

## How it works

### The algorithm

Standard sampling picks tokens greedily based on next-token likelihood. Power sampling
instead targets the *joint* sequence distribution p(x)^α, which upweights tokens that
lead to high-likelihood *complete sequences* — an implicit form of planning.

This is NOT the same as low-temperature sampling. Low temp exponentiates each
conditional `p(x_t | x_{<t})^α` independently (greedy). The power distribution
exponentiates the *joint* `p(x_{0:T})^α`, which accounts for how current tokens
affect future token quality.

**Algorithm (block-wise MCMC):**

1. Generate an initial sequence from the base model
2. For each block of tokens:
   a. Pick a random position in the sequence
   b. Resample everything from that position onward
   c. Accept/reject via Metropolis-Hastings: higher p^α sequences accepted more often
   d. Repeat for `mcmc_steps` iterations
3. The accepted sequence converges toward a sample from p^α

With temperature=1 proposal, the MH acceptance ratio simplifies to:

```
log_ratio = (α - 1) * [Σ log_p(new_suffix) - Σ log_p(old_suffix)]
```

Only base model logprobs needed — vLLM provides these directly.

### How it fits our setup

The ICL baseline uses mini-swe-agent (multi-turn bash tool calls). Power sampling
can't apply to multi-turn agent trajectories (tool responses break the autoregressive
chain). So this ablation uses **single-shot generation**: the model outputs reasoning
+ sed commands in one pass, we parse and apply the sed commands.

This means the comparison is:

- **ICL baseline**: multi-turn agent editing, vanilla sampling, accumulated feedback
- **Power sampling**: single-shot sed generation, MCMC-refined sampling, same feedback
- **SDPO**: multi-turn agent editing, vanilla sampling, weight updates from RL

The power sampling ablation tests whether *smarter sampling* can compensate for not
having the iterative agent loop.

## Files

| File | Role |
|------|------|
| `power_sampling.py` | Core MCMC algorithm + best-of-N. Uses vLLM chat completions with logprobs. |
| `loop_power_sampling.py` | Experiment loop: generate → parse sed → apply → dispatch → feedback. |
| `smoke_test_power.py` | End-to-end smoke test (see below). |

## Smoke test

Verifies every stage of the pipeline before committing to a full 30-turn run.

```bash
python smoke_test_power.py --vllm-base-url http://20.125.45.203:8000/v1
```

**Steps:**

| # | What | Time | Failure means |
|---|------|------|---------------|
| 1 | vLLM connectivity + logprobs returned | ~1s | Server down or model not loaded |
| 2 | `continue_final_message` support | ~1s | vLLM too old for MCMC resampling (best-of-n still works) |
| 3 | `power_sample()` with small params (2 blocks, 2 MCMC steps, 4096 tokens) | ~3.5min | Bug in MCMC logic or API mismatch |
| 4 | `best_of_n()` with n=2, 4096 tokens | ~3min | Basic generation broken |
| 5 | Sed parsing + application from actual model output | instant | Model not producing sed commands (prompt needs tuning) |
| 6 | Experiment dispatch to H100 (15s then kill) | ~15s | SSH/file sync broken |

Step 6 does NOT run a full 5-minute experiment. It dispatches the modified train.py,
lets the experiment start for 15 seconds, then kills it. This confirms SSH connectivity,
file sync, and process launch all work. The `SSHRunner` timeout mechanism handles the
kill and cleanup.

## Running

```bash
# MCMC power sampling (paper's algorithm)
python loop_power_sampling.py \
    --method power \
    --alpha 4.0 \
    --mcmc-steps 10 \
    --block-num 16 \
    --max-tokens 4096 \
    --max-turns 30 \
    --run-name qwen3-14b-power-a4

# Simpler best-of-N (good sanity check)
python loop_power_sampling.py \
    --method best-of-n \
    --n 8 \
    --max-tokens 4096 \
    --max-turns 30 \
    --run-name qwen3-14b-best-of-8
```

## Hyperparameters

| Param | Default | Paper | Notes |
|-------|---------|-------|-------|
| `alpha` | 4.0 | 4.0 | Higher = sharper. Try 2.0 if acceptance rate is near 0 |
| `block_num` | 16 | 16 | Number of sequential MCMC blocks |
| `mcmc_steps` | 10 | 10 | Steps per block. 2 already gives most of the gain |
| `temperature` | 1.0 | 0.25 | Paper uses 1/α but that needs temperature-scaled logprobs. We use 1.0 for exact MH ratio with the vLLM API |
| `max_tokens` | 4096 | 3072 | Qwen3 uses ~3K tokens for `<think>` blocks before outputting sed commands |
| `n` (best-of-n) | 8 | — | Paper doesn't test this but it's a simpler baseline |

## Qwen3 thinking behavior

Qwen3 generates `<think>...</think>` blocks before its actual response. This uses
~3K tokens of the budget before any sed commands appear. The chat completions API
strips the `<think>` tags from `message.content` but the tokens still count against
`max_tokens`. This is why we default to `max_tokens=4096`.

Smoke test showed: with 4096 max_tokens, the model reliably produces reasoning + sed
commands. With 2 blocks × 2 MCMC steps, acceptance rate was 25% and wall time was
~3.5 minutes per generation.

For full runs (16 blocks × 10 MCMC steps), expect ~10-15 minutes per turn for the
generation phase, plus ~5 minutes for the experiment. Total ~15-20 min/turn.

## Differences from the reference implementation

Verified against https://github.com/aakaran/reasoning-with-sampling

The reference uses HuggingFace `model.generate(output_logits=True, output_scores=True)`
which returns BOTH raw logits and temperature-scaled logits. This enables the full
4-term MH acceptance ratio with a low-temperature proposal (τ=1/α=0.25):

```
log_r = α·Σlog_p(new) + Σlog_q(old) − α·Σlog_p(old) − Σlog_q(new)
```

where `log_q` = `log_softmax(logits/τ)` (temperature-scaled).

We use the vLLM API which only exposes base model logprobs `log_p`. This forces us
to use τ=1 (proposal = base model), which simplifies the ratio to:

```
log_r = (α−1) · [Σlog_p(new) − Σlog_p(old)]
```

This is **mathematically exact** for τ=1. The tradeoff vs the paper:
- **Proposals are more random** (temp=1 vs 0.25), so acceptance rate is lower
- **Convergence is slower** — may need more MCMC steps
- Smoke test showed 25% acceptance with α=4, which is workable
- The paper's Figure 6 shows gains plateau around NMCMC=10; we may need more

Everything else matches the reference: block-wise progression, resampling index over
all generated tokens (not just current block), accept/reject logic, EOS handling.

## Known limitations

1. **Proposal temperature**: See "Differences from reference" above. We use τ=1.0;
   paper uses τ=1/α. A warning is emitted if temperature≠1.0 is passed.

2. **Single-shot vs agent**: The ICL baseline uses multi-turn tool calling (20 bash
   calls per turn). Power sampling uses single-shot generation. This is a confound —
   if power sampling underperforms, it might be the single-shot format, not the
   sampling method.

3. **Variable-length sequences**: The paper assumes fixed-length sequences. Our
   generations vary in length (model stops at EOS). Proposals shorter than the
   current sequence are handled by comparing over the overlap.

4. **`continue_final_message`**: Required for MCMC resampling. If vLLM doesn't
   support it, fall back to `--method best-of-n`.

## What to look for in results

- **acceptance_rate**: Logged to wandb. Should be 0.1-0.5. Near 0 = α too high
  or proposals too random. Near 1 = α too low (barely sharpening).
- **val_bpb vs ICL baseline**: Does smarter single-shot sampling beat iterative
  agent editing?
- **val_bpb vs SDPO**: Does inference-time sharpening match learned weight updates?
- **gen_time**: How much overhead does power sampling add per turn?
- **no_sed_commands rate**: If high, the model isn't following the prompt format.
