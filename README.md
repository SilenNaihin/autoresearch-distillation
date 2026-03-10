# autoresearch-distillation

Boost a model's single-shot performance at coming up with novel ML research ideas using test-time training. An open-source model proposes modifications to a training script, runs real experiments, and learns from the outcomes via [Self-Distillation Policy Optimization (SDPO)](https://github.com/lasgroup/SDPO).

Built on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and [SDPO](https://github.com/lasgroup/SDPO).

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                     SDPO TRAINING LOOP                           │
│                                                                  │
│   1. Model receives prompt (train.py + experiment history)       │
│   2. Model thinks step-by-step, then edits train.py via sed      │
│   3. Model submits: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT   │
│   4. Modified train.py dispatched to a remote H100 via SSH       │
│   5. uv run train.py executes (5 min fixed budget)               │
│   6. val_bpb parsed → reward signal computed                     │
│   7. SDPO updates model weights from the rollout                 │
│   8. GOTO 1 — model improves at proposing experiments            │
│                                                                  │
│   No separate reward model. No offline data collection.          │
│   The agent trains on live experiment outcomes.                   │
└──────────────────────────────────────────────────────────────────┘
```

The model sees `train.py`, uses bash commands (primarily `sed`) to make targeted edits in an isolated workdir, and submits when complete. The modified script is dispatched to a GPU for evaluation, and the model gets a scalar reward based on whether `val_bpb` improved. SDPO (Self-Distillation Policy Optimization) uses the model's own successful rollouts as the teacher signal — no separate reward model or critic needed.

Chain-of-thought is enabled (`enable_thinking: true`), so the model reasons in `<think>` blocks before acting.

## Architecture

```
box3 (h100-dev-box-3)           — vLLM inference + FSDP2 training (2 GPUs)
box2, box4, box5                — experiment execution (5 H100s total)
```

**VERL** (via SDPO fork) handles the RL training loop: rollout generation, advantage estimation, policy updates. Our custom `AutoresearchAgentLoop` extends VERL's `ToolAgentLoop` — the model uses a bash tool to edit `train.py` across multiple turns, then the modified script is dispatched to a GPU and the experiment reward flows back into training. Tool response tokens are masked (`response_mask=0`), so we only train on the model's decisions.

**GPUPoolRunner** manages 5 remote H100s via SSH. Thread-safe with file-based locking — multiple VERL workers dispatch experiments concurrently.

## GPU Fleet

| Slot | Host | GPU | Role |
|------|------|-----|------|
| box2-gpu0 | h100-dev-box-2 | 0 | Experiment |
| box3 | h100-dev-box-3 | 0,1 | vLLM + FSDP2 (VERL) |
| box4-gpu0 | h100-dev-box-4 | 0 | Experiment |
| box4-gpu1 | h100-dev-box-4 | 1 | Experiment |
| box5-gpu0 | h100-dev-box-5 | 0 | Experiment |
| box5-gpu1 | h100-dev-box-5 | 1 | Experiment |

## Reward Structure

Reward is computed against a hardcoded baseline (`BASELINE_VAL_BPB = 1.056`):

| Status | Reward | When |
|--------|--------|------|
| improvement | `max(0, 1.056 - val_bpb)` | val_bpb below baseline |
| no_improvement | `0.0` | val_bpb at or above baseline |
| crash | `0.0` | any failure mode |

No negative rewards. The SDPO `success_reward_threshold` is set to `0.0`, so any rollout that beats baseline is treated as a "success" for self-distillation.

## Project Structure

```
autoresearch-distillation/
├── agent_loop.py              # VERL agent loop — multi-turn bash editing + experiment dispatch
├── bash_tool.py               # VERL BashTool — isolated workdir + bash execution
├── environment.py             # RunOutput, parse_metrics(), compute_reward()
├── runners.py                 # GPUPoolRunner — SSH dispatch to remote H100s
├── prompts.py                 # System prompt + instance prompt builder
├── reward.py                  # Passthrough reward function for VERL
├── run_sdpo.py                # Entry point — patches SDPO trainer for env metrics logging
│
├── configs/
│   ├── autoresearch_sdpo.yaml # Hydra config for SDPO training (Qwen3-14B)
│   ├── smoke_test.yaml        # 1-step smoke test config (Qwen3-4B)
│   ├── bash_tool_config.yaml  # VERL tool config for bash tool
│   └── agent_loops.yaml       # Agent loop registry + behavioral_feedback flag
├── data/
│   └── prepare_autoresearch.py # Convert rollouts.jsonl → parquet for SDPO
├── scripts/
│   ├── run_training.sh        # Launch SDPO training (patches YaRN, copies configs)
│   ├── run_smoke_test.sh      # 1-step end-to-end verification
│   ├── setup_gpu_box.sh       # Bootstrap a remote GPU box
│   └── smoke_test.sh          # Verify GPU box connectivity
│
├── autoresearch/              # Upstream submodule (train.py, prepare.py) — read-only
├── SDPO/                      # SDPO/VERL fork — submodule
│
├── test_e2e.py                # End-to-end fleet verification
├── test_runner.py             # Runner unit tests
├── INTEGRATION.md             # Technical integration guide
└── README.md
```

## Usage

### 1. Prepare training data

Converts rollouts to parquet with deduplication and 80/20 train/test split.

```bash
python data/prepare_autoresearch.py
```

### 2. Run smoke test

Runs 1 batch / 1 step end-to-end with Qwen3-4B to verify the full pipeline.

```bash
bash scripts/run_smoke_test.sh
```

### 3. Run SDPO training

Launches VERL with our custom agent loop. The model uses a bash tool to edit `train.py` across multiple turns during training, experiments run on 5 remote GPUs, and rewards flow back into the policy update.

```bash
bash scripts/run_training.sh [experiment_name]
```

### Key config (`configs/autoresearch_sdpo.yaml`)

- **Model**: Qwen/Qwen3-14B with YaRN rope_scaling (factor=2.0 for 64k context)
- **Training GPUs**: 2 (colocated vLLM + FSDP2 with CPU offloading)
- **Experiment GPUs**: 5 (remote, via SSH)
- **Batch size**: 16 rollouts
- **Max prompt length**: 16384 tokens
- **Max response length**: 49152 tokens
- **Learning rate**: 1e-5
- **Epochs**: 60
- **Chain of thought**: enabled (`enable_thinking: true`)
- **Self-distillation**: Environment feedback included in teacher prompt
- **Behavioral feedback**: Configurable notes to teacher about noop commands, malformed args, no-change submissions

## Observability

### Wandb metrics

- **`env/val_bpb/mean`** — Environment metrics from experiment runs
- **`self_distillation/reprompt_samples`** — Decoded teacher reprompt prefixes (first 3 per step)
- **`feedback_available_fraction`** — Should be ~1.0
- **`actor/entropy`** — Watch for collapse toward 0
- **`pg_loss`**, **`self_distillation_loss`** — Should be > 0

### Rollout dumps

Set `trainer.rollout_data_dir` in config (default: `/data/rollout_dumps`). Each training step dumps a JSONL with inputs, outputs, scores, and feedback.

### Storage paths

On the training server:
- `/data/checkpoints/{experiment_name}` — model checkpoints
- `/data/rollout_dumps` — rollout data per training step
- `/data/exp{N}` — archived artifacts from previous runs

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- H100 GPUs with CUDA 12+
- SSH access between machines (configured in `~/.ssh/config`)

## Relationship to Upstream

- **autoresearch/** — submodule tracking [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Contains `train.py` (the file the agent modifies), `prepare.py` (data/eval), and `pyproject.toml`.
- **SDPO/** — submodule tracking a fork of [SDPO](https://github.com/lasgroup/SDPO). Provides the VERL trainer, agent loop system, and SDPO loss.

## License

MIT
