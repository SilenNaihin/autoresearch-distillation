# autoresearch-distillation

Self-improving ML research agent. An open-source model proposes modifications to a training script, runs real experiments, and learns from the outcomes via online self-distillation (SDPO).

Built on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and [ByteDance's SDPO](https://github.com/bytedance/SDPO).

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                     SDPO TRAINING LOOP                           │
│                                                                  │
│   1. Qwen3-4B generates a unified diff for train.py             │
│   2. Diff dispatched to a remote H100 via SSH                   │
│   3. uv run train.py executes (5 min fixed budget)              │
│   4. val_bpb parsed → reward signal computed                    │
│   5. SDPO updates model weights from the rollout                │
│   6. GOTO 1 — model improves at proposing experiments           │
│                                                                  │
│   No separate reward model. No offline data collection.          │
│   The agent trains on live experiment outcomes.                  │
└──────────────────────────────────────────────────────────────────┘
```

The model sees `train.py` + experiment history, outputs reasoning + a diff, and gets a scalar reward based on whether `val_bpb` improved. SDPO (Self-Distillation Policy Optimization) uses the model's own successful rollouts as the teacher signal — no separate reward model or critic needed.

## Architecture

```
box3 (h100-dev-box-3)           — vLLM inference + FSDP2 training (2 GPUs)
box1, box2, box4, box5          — experiment execution (6 H100s total)
```

**VERL** (via SDPO fork) handles the RL training loop: rollout generation, advantage estimation, policy updates. Our custom `AutoresearchAgentLoop` plugs into VERL's agent loop system — it generates a response, then calls `ExperimentEnvironment.step()` which dispatches the experiment to a free GPU and returns the reward.

**ExperimentEnvironment** handles prompt construction, diff extraction, patch application, experiment dispatch (via `GPUPoolRunner`), metric parsing, and reward computation.

**GPUPoolRunner** manages 6 remote H100s via SSH. Thread-safe — multiple experiments run in parallel automatically.

## GPU Fleet

| Slot | Host | GPU | Role |
|------|------|-----|------|
| box1-gpu0 | h100_azure | 0 | Experiment |
| box2-gpu0 | h100-dev-box-2 | 0 | Experiment |
| box3 | h100-dev-box-3 | 0,1 | vLLM + FSDP2 (VERL) |
| box4-gpu0 | h100-dev-box-4 | 0 | Experiment |
| box4-gpu1 | h100-dev-box-4 | 1 | Experiment |
| box5-gpu0 | h100-dev-box-5 | 0 | Experiment |
| box5-gpu1 | h100-dev-box-5 | 1 | Experiment |

## Reward Structure

| Status | Reward | When |
|--------|--------|------|
| improvement | `+delta * 100` | val_bpb beat the best (0.01 → reward 1.0) |
| no_improvement | `-delta * 50` | val_bpb equal or worse |
| crash | `-1.0` | script crashed, OOM, timeout |
| patch_error | `-1.0` | diff couldn't be applied |
| parse_error | `-1.0` | no diff found in model response |

## Project Structure

```
autoresearch-distillation/
├── agent_loop.py              # VERL agent loop — generates response, runs experiment, returns reward
├── environment.py             # ExperimentEnvironment — prompt, diff, patch, reward logic
├── runners.py                 # GPUPoolRunner — SSH dispatch to 6 remote H100s
├── loop.py                    # Standalone data collection loop (Qwen3-32B via vLLM API)
│
├── configs/
│   └── autoresearch_sdpo.yaml # Hydra config for SDPO training
├── data/
│   └── prepare_autoresearch.py # Convert rollouts.jsonl → parquet for SDPO
├── scripts/
│   ├── run_training.sh        # Launch SDPO training
│   ├── setup_gpu_box.sh       # Bootstrap a GPU box
│   └── smoke_test.sh          # Verify GPU box setup
│
├── autoresearch/              # Upstream submodule (train.py, prepare.py) — read-only
├── SDPO/                      # SDPO/VERL fork — submodule
├── rollouts/                  # Collected rollouts (rollouts.jsonl)
│
├── INTEGRATION.md             # Environment/runner API docs
├── test_e2e.py                # End-to-end fleet verification
└── test_runner.py             # Runner unit tests
```

## Usage

### 1. Collect rollouts (standalone loop)

Uses Qwen3-32B via vLLM to generate diffs and run experiments. Produces `rollouts/rollouts.jsonl`.

```bash
# Start vLLM on GPU 0, experiments on GPU 1
python loop.py
```

### 2. Prepare training data

Converts rollouts to parquet with deduplication and 80/20 train/test split.

```bash
python data/prepare_autoresearch.py
```

### 3. Run SDPO training

Launches VERL with our custom agent loop. The model generates diffs during training, experiments run on 6 remote GPUs, and rewards flow back into the policy update.

```bash
bash scripts/run_training.sh
```

### Key config (`configs/autoresearch_sdpo.yaml`)

- **Model**: Qwen/Qwen3-4B
- **Training GPUs**: 2 (colocated vLLM + FSDP2)
- **Experiment GPUs**: 6 (remote, via SSH)
- **Rollouts per batch**: 8
- **Max response length**: 32768 tokens
- **Learning rate**: 1e-5
- **Epochs**: 30

### Using the environment directly

```python
from environment import ExperimentEnvironment
from runners import GPUPoolRunner

env = ExperimentEnvironment(GPUPoolRunner())

system, user = env.get_prompt()       # prompt for the policy
result = env.step(model_response)     # ~5 min, auto-dispatches to free GPU
result.reward                         # scalar for RL
result.feedback                       # text feedback
```

See [INTEGRATION.md](INTEGRATION.md) for the full API.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- H100 GPUs with CUDA 12+
- SSH access between machines (configured in `~/.ssh/config`)

## Relationship to Upstream

- **autoresearch/** — submodule tracking [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Contains `train.py` (the file the agent modifies), `prepare.py` (data/eval), and `pyproject.toml`.
- **SDPO/** — submodule tracking a fork of [ByteDance's SDPO](https://github.com/bytedance/SDPO). Provides the VERL trainer, agent loop system, and SDPO loss.

## License

MIT
