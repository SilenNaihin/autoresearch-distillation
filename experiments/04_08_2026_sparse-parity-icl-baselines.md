# Sparse Parity ICL Baselines

**Branch:** `feat/sparse-parity-challenge`
**Date:** 2026-04-08
**Box:** a100-backup-1 (1x A100 80GB, 216GB RAM)

## Purpose

Establish ICL baselines (no weight updates) for the sparse parity DMC optimization challenge
before running SDPO training. Two baselines:
- **B1**: Qwen3-14B via vLLM local inference
- **B2**: Opus 4.6 via Bedrock API

Both use the same PUCT reuse buffer, evaluation harness, and feedback loop.
The leaderboard best prior to this experiment was Sequential Elimination at ~19,153 DMC.

## Setup

### B2: Opus ICL (complete)
- Script: `baselines/opus_icl_sparse_parity.py`
- Model: `bedrock/us.anthropic.claude-opus-4-6-v1` via litellm
- Config: 20 turns, temperature=0.8, buffer at `/tmp/opus_reuse_buffer.json`
- wandb: https://wandb.ai/kfallah/sparse-parity-sdpo/runs/7nrmfvbh
- Commit: `1068e9a` (initial), `a520941` (PYTHONPATH fix)

### B1: Qwen ICL (running)
- Script: `baselines/qwen_icl_sparse_parity.py`
- Model: Qwen/Qwen3-14B via vLLM v0.11.2, bf16, gpu_mem=0.55
- Config: 30 turns, temperature=1.0, 16K max_tokens (to accommodate thinking)
- wandb: https://wandb.ai/kfallah/sparse-parity-sdpo/runs/dhy0ghid
- Commit: `fe59a74`

### Why standalone scripts instead of SDPO infrastructure?
The SDPO training pipeline allocates Adam optimizer states (~118GB for 14B model) even with
lr=0, causing OOM on the 216GB RAM box. Standalone vLLM inference uses only ~46GB GPU memory
and minimal CPU RAM.

## Hiccups

1. **sparse-parity-challenge repo not pip-installable** - Has no setup.py/pyproject.toml.
   Fixed by cloning to `~/sparse-parity-challenge` and setting PYTHONPATH.

2. **evaluate.py `import importlib` vs `import importlib.util`** - Python 3.12 (conda env)
   doesn't auto-import `importlib.util`. Fixed explicitly.

3. **SDPO pipeline OOM at 183GB** - Loading actor (fp32) + ref + vLLM + Adam states exhausted
   216GB RAM. Even with bf16 actor (model_dtype: bf16 in fsdp_config), still 179GB OOM due
   to optimizer states. Pivoted to standalone vLLM scripts.

4. **Qwen3 thinking tokens** - With `enable_thinking=True`, model's `<think>` block consumed
   all 4096 max_tokens leaving no code output. Fixed by increasing to 16384.

5. **TrackedArray extremely slow** - LRU stack tracking is O(n) per access. GF(2) gaussian
   seed took 173s for 3 eval seeds. Some seeds (sequential_elimination) hung indefinitely.
   Seed evaluation not fully completed.

## Results

### B2: Opus ICL (20 turns)
| Turn | DMC | Status |
|------|-----|--------|
| 0 | 997,874 | improvement |
| 3 | 1,317,363 | improvement |
| 5 | 959,535 | improvement |
| 6 | 733,318 | improvement |
| 7 | 18,344 | improvement |
| 8 | **15,724** | improvement (best) |
| 12 | 20,965 | improvement |
| 13 | 22,818 | improvement |
| 14 | 26,206 | improvement |

**Best: DMC = 15,724** (18% better than leaderboard's 19,153)

Key observation: Opus converged fast — found sub-20K DMC by turn 7 (GF(2) elimination approach).
14 out of 20 turns succeeded. Crashes were from 0% accuracy (broken algorithms).

### B1: Qwen ICL (in progress, ~4/30 turns)
- Turn 1: DMC = 1,474,082 (first success)
- Remaining turns in progress (~6 min/turn)

Early observation: Qwen's first valid solution is 66x worse than Opus's best. The 14B model
may need more turns to discover efficient algorithms.

## Open Questions

1. Will Qwen converge to sub-100K DMC with 30 turns?
2. Should we increase turn count or add seed solutions to the PUCT buffer?
3. For SDPO runs: can we use the standalone vLLM approach for rollouts while doing
   separate gradient updates, or must we fix the SDPO OOM?
