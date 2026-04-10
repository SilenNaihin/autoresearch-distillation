# Sparse Parity: SDPO Training + Opus Single-Shot Baseline

**Branch:** `feat/sparse-parity-challenge`
**Date:** 2026-04-08
**Box:** a100-backup-1 (1x A100 80GB, 216GB RAM)

## Purpose

Run SDPO training with Qwen3-14B and Opus 4.6 single-shot baseline (no ICL feedback)
for the sparse parity DMC challenge. Follows ICL baselines from earlier experiment.

## B3: Opus Single-Shot (COMPLETE)

- Script: `baselines/opus_single_shot_sparse_parity.py`
- Model: `bedrock/us.anthropic.claude-opus-4-6-v1` via litellm
- Config: 50 turns, temperature=1.0, buffer pre-seeded with 7 seed solutions
- wandb: https://wandb.ai/kfallah/sparse-parity-sdpo/runs/xal9gwhb
- Commit: `5e880ed`

**Result: Best DMC = 28,103 over 50 turns, 54 buffer states**

Worse than Opus ICL (15,724) since single-shot has no feedback history.
Each turn is independent — model sees a seed solution from PUCT buffer and
generates a modification without context on what worked/failed before.

## B4: SDPO Qwen3-14B Full FT (KILLED — too slow)

### Config
- Model: Qwen/Qwen3-14B (14.77B params)
- Optimizer: torchao AdamW8bit (CPU-compatible 8-bit Adam)
- FSDP2 with CPU offload (actor + ref)
- vLLM rollout with enforce_eager=True
- Agent loop: multi-turn bash tool, local evaluation
- wandb: https://wandb.ai/kfallah/sparse-parity-sdpo/runs/c7vm9b45
- Commit: `a337532`

### Launch Command
```bash
cd /home/azureuser/autoresearch-distillation/SDPO && \
TASK=sparse_parity DATA_DIR=/home/azureuser/data \
PYTHONPATH=/home/azureuser/sparse-parity-challenge/src:/home/azureuser/autoresearch-distillation:$PYTHONPATH \
RAY_memory_monitor_refresh_ms=0 \
python training/run_sdpo.py --config-name sparse_parity_sdpo \
  vars.dir=/home/azureuser/autoresearch-distillation \
  vars.ckpt_dir=/home/azureuser/data/checkpoints \
  trainer.rollout_data_dir=/home/azureuser/data/rollout_dumps \
  trainer.group_name=sparse-parity-sdpo \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.optim.optimizer=AdamW8bit \
  actor_rollout_ref.actor.optim.optimizer_impl=torchao.optim
```

### Step 1 Metrics
- Total step time: 1660s (~28 min)
- Rollout time: 368s (6 min) — multi-turn with mean 4.875 turns
- Optimizer step: 1266s (21 min) — CPU offload dominates
- Response length mean: 10,235 tokens
- CPU memory: 191.7 GB / 216 GB
- GPU memory: 77.2 GB / 80 GB
- actor/lr: 0.0 (warmup phase, lr_warmup_steps=5)

### Hiccups Encountered

1. **EXPERIMENT env var missing** — `user.yaml` defaults `group_name: ${oc.env:EXPERIMENT}`.
   Fixed: added `trainer.group_name` override. Hydra structured config issue requires
   CLI overrides, not YAML.

2. **FileNotFoundError: sparse_parity/solve.py** — `workspace.source_dir` was relative,
   resolved against CWD (SDPO/) instead of repo root.
   Fixed: `TaskConfig.from_yaml()` now resolves source_dir to absolute path at load time.
   Commit: `432140a`

3. **PermissionError: /data** — `/data` directory doesn't exist on a100-backup-1.
   Fixed: agent_loop.py and experiment_cache.py now use DATA_DIR env var.
   Commit: `ea4f6a2`

4. **vLLM EngineDeadError: 'NoneType' has no 'result'** — VERL's
   `ExternalZeroMQDistributedExecutor.execute_model` was gated behind vLLM >= 0.12.0
   but box has 0.11.2. Base class's `execute_model` passes `non_block=True` to
   `collective_rpc` which silently drops it — returns raw result instead of Future.
   Fixed: lowered version gate to >= 0.11.0. Commit: `67ec74a`

5. **ModuleNotFoundError: 'runners'** — GPUPoolRunner import fails because
   `runners.py` is at `lib/runners.py`. Sparse parity evaluates locally (CPU-only).
   Fixed: replaced with local `_evaluate_locally()` function. Commit: `4d202f8`

6. **bitsandbytes AdamW8bit requires GPU tensors** — FSDP2 offload keeps params on
   CPU during optimizer step. bitsandbytes CUDA kernels can't operate on CPU tensors.
   Fixed: switched to torchao AdamW8bit which supports CPU. Commit: `a337532`
   Standard fp32 AdamW was tried first but OOM'd (~118GB optimizer states + 30GB model).

7. **Hydra structured config: YAML overrides silently ignored** — `optimizer`,
   `optimizer_impl`, `group_name` overrides in YAML were not applied. Must use CLI
   overrides. This is a known issue documented in memory.

## Resource Profile
- GPU: 77.2 / 80 GB (vLLM 0.55 × 80 = 44GB + model forward pass)
- CPU: 191.7 / 216 GB (FSDP2 offloaded params + 8-bit optimizer states)
- Step time: ~28 min/step (dominated by CPU optimizer step at 21 min)

## B4 → B5: Switch to LoRA rank 32

Full fine-tuning with CPU offload was unacceptably slow: 21 min optimizer step (66% of
31 min total). Root cause: FSDP2 offload_policy=true forces all 14.77B param updates to
CPU. No way to selectively keep optimizer on GPU in FSDP2.

**Fix: LoRA rank 32** — reduces trainable params from 14.77B to ~50M. Eliminates need
for CPU offload entirely. Optimizer runs on GPU.

Changes:
- `actor_rollout_ref.model.lora_rank: 32` (VERL native PEFT support)
- `actor_rollout_ref.model.lora_alpha: 16`
- `actor_rollout_ref.model.target_modules: all-linear`
- `actor_rollout_ref.actor.fsdp_config.offload_policy: false`
- `actor_rollout_ref.ref.fsdp_config.offload_policy: false`
- Switched back to standard AdamW (8-bit unnecessary for 50M params)
- New experiment_name: `qwen3-14b-sdpo-lora32`
- Ref model: VERL auto-uses base model (adapter disabled) as reference when LoRA enabled

Expected memory profile:
- Training: ~30GB model (bf16) + ~300MB LoRA adapter + optimizer ≈ 31GB
- Rollout: vLLM 0.55 × 80GB = 44GB (model + KV cache)
- CPU RAM: minimal (no offload)

## B5: SDPO Qwen3-14B + LoRA rank 32 (RUNNING — stalled, LR too low)

### Hiccups
8. **bitsandbytes optimizer_impl default** — CLI override changed `optimizer` to AdamW
   but `optimizer_impl` fell back to `bitsandbytes.optim` from base sdpo config.
   `bitsandbytes.optim.AdamW` still requires CUDA tensors for its `is_on_gpu()` check,
   which fails because some LoRA parameters have `grad=None`.
   Fixed: added CLI override `actor_rollout_ref.actor.optim.optimizer_impl=torch.optim`
   and YAML `optimizer_impl: torch.optim`. Commit: `c066243`

### Config
- Model: Qwen/Qwen3-14B + LoRA rank 32 (14.90B total, ~130M trainable)
- Optimizer: torch.optim.AdamW (on GPU)
- FSDP2 without CPU offload
- wandb: https://wandb.ai/kfallah/sparse-parity-sdpo/runs/j08tcp9s
- Commit: `c066243`

### Launch Command
```bash
cd /home/azureuser/autoresearch-distillation/SDPO && \
source /home/azureuser/miniconda3/bin/activate verl && \
TASK=sparse_parity DATA_DIR=/home/azureuser/data \
PYTHONPATH=/home/azureuser/sparse-parity-challenge/src:/home/azureuser/autoresearch-distillation:$PYTHONPATH \
RAY_memory_monitor_refresh_ms=0 HYDRA_FULL_ERROR=1 \
python /home/azureuser/autoresearch-distillation/training/run_sdpo.py \
  --config-name sparse_parity_sdpo \
  vars.dir=/home/azureuser/autoresearch-distillation \
  vars.ckpt_dir=/home/azureuser/data/checkpoints \
  trainer.rollout_data_dir=/home/azureuser/data/rollout_dumps \
  trainer.group_name=sparse-parity-sdpo \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim
```

### Step-by-step timing breakdown (step 1 → steady state)

| Component | Step 1 | Steps 2-20 (avg) | Notes |
|-----------|--------|-------------------|-------|
| gen (rollout) | 702s | ~1600s | Bottleneck. Slowest trajectory dominates batch |
| update_actor | 123s | ~140s | LoRA optimizer on GPU — 8.7x faster than CPU FT |
| old_log_prob | 29s | ~31s | Ref log probs (base model, adapter disabled) |
| adv | 1s | ~20s | Advantage computation |
| **Total** | **855s (14m)** | **~1900s (32m)** | High variance: 14m to 47m |

### Results after 20 steps (11.5 hours)

- **Best DMC: 637,131** (first achieved at step 5, never improved after)
- Baseline DMC for comparison: 28,103 (Opus single-shot) / 15,724 (Opus ICL)
- `success_sample_fraction: 0.0` across all 20 steps — model never beat baseline
- Rewards non-zero on ~60% of steps (max ~9.4M) but same cached values repeat
- `rollout_probs_diff_mean: ~0.005` — policy only 0.5% different from base model
- `env/novel` trending down — model increasingly hitting cached/duplicate results

### Why it stalled

1. **LR 1e-6 is far too low for LoRA.** Typical LoRA LR is 1e-4 to 2e-4. At 1e-6,
   the adapter weights barely move — policy is <1% different from base after 20 steps.
2. **SDPO teacher (EMA, rate 0.01) is essentially identical to actor.** The teacher's
   reprompted responses are indistinguishable from the base model, providing no useful
   distillation signal.
3. **Rollout dominates step time.** Multi-turn agent loop with max 10 turns × 2 (user +
   assistant) = 20 total turns. Slowest trajectory blocks the batch. One trajectory at
   24K tokens / 20 turns takes 43 min while others finish in 10 min.
4. **GPU memory oversubscription.** `max_memory_allocated: 113.8 GB` on 80GB A100.
   CUDA UVM spilling to host RAM, slowing all GPU operations.

### Key resource profile (LoRA)
- GPU peak: 113.8 GB allocated (80GB VRAM + UVM spill to host)
- GPU reserved: 114.3 GB
- CPU RAM: 130 GB / 216 GB
- update_actor: ~140s (was 1223s with full FT — 8.7x improvement)

## Next Steps

Need to address all four issues before restarting:
1. Increase LR to ~5e-5 (50x current)
2. Reduce max_user_turns/max_assistant_turns to limit rollout time
3. Consider reducing max_response_length or gpu_memory_utilization
4. May need to reduce batch size or model context to fit within 80GB VRAM
