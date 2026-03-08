# autoresearch-distillation

Self-improving ML research agent. An open-source model proposes modifications to a training script, runs real experiments, and learns from the outcomes via online self-distillation (SDPO).

Built on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and [ByteDance's SDPO](https://github.com/bytedance/SDPO).

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                     SDPO TRAINING LOOP                           │
│                                                                  │
│   1. Model receives prompt (train.py + experiment history)       │
│   2. Model edits train.py via bash tool (multi-turn)             │
│   3. Model submits when satisfied with changes                   │
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

The model sees `train.py` + experiment history, uses bash commands to read and edit the file in an isolated workdir, and submits when satisfied. The modified script is dispatched to a GPU for evaluation, and the model gets a scalar reward based on whether `val_bpb` improved. SDPO (Self-Distillation Policy Optimization) uses the model's own successful rollouts as the teacher signal — no separate reward model or critic needed.

The multi-turn editing approach (via [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)) lets the model verify its changes before submission, recovering from malformed edits that would fail silently in a single-turn diff-based approach.

## Architecture

```
box3 (h100-dev-box-3)           — vLLM inference + FSDP2 training (2 GPUs)
box1, box2, box4, box5          — experiment execution (6 H100s total)
```

**VERL** (via SDPO fork) handles the RL training loop: rollout generation, advantage estimation, policy updates. Our custom `AutoresearchAgentLoop` extends VERL's `ToolAgentLoop` — the model uses a bash tool to edit `train.py` across multiple turns, then the modified script is dispatched to a GPU and the experiment reward flows back into training. Tool response tokens are masked (`response_mask=0`), so we only train on the model's decisions.

**ExperimentEnvironment** handles prompt construction, diff extraction, patch application, experiment dispatch (via `GPUPoolRunner`), metric parsing, and reward computation. (Used by the legacy single-turn `loop.py`; the multi-turn `loop_swe.py` calls these components directly.)

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
├── agent_loop.py              # VERL agent loop — extends ToolAgentLoop for multi-turn bash editing
├── bash_tool.py               # Isolated workdir, mini-swe-agent runner, VERL BashTool
├── environment.py             # ExperimentEnvironment — prompt, diff, patch, reward logic
├── runners.py                 # GPUPoolRunner — SSH dispatch to 6 remote H100s
├── prompts.py                 # Shared prompt templates for bash-tool editing
├── loop_swe.py                # Multi-turn data collection loop (mini-swe-agent + vLLM)
├── loop.py                    # Legacy single-turn data collection loop (diff-based)
│
├── configs/
│   ├── autoresearch_sdpo.yaml # Hydra config for SDPO training (multi-turn enabled)
│   └── bash_tool_config.yaml  # VERL tool config for bash tool
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

### 1. Collect rollouts (multi-turn loop)

Uses Qwen3-32B via vLLM + mini-swe-agent for multi-turn bash editing. The model reads `train.py`, makes edits, verifies them, and submits. Produces `rollouts/rollouts.jsonl`.

```bash
# Start vLLM on GPU 0, experiments dispatched to GPU fleet
python loop_swe.py
```

A legacy single-turn loop (`loop.py`) is also available — it uses SEARCH/REPLACE blocks instead of bash commands.

### 2. Prepare training data

Converts rollouts to parquet with deduplication and 80/20 train/test split.

```bash
python data/prepare_autoresearch.py
```

### 3. Run SDPO training

Launches VERL with our custom agent loop. The model uses a bash tool to edit `train.py` across multiple turns during training, experiments run on 6 remote GPUs, and rewards flow back into the policy update.

```bash
bash scripts/run_training.sh
```

### Key config (`configs/autoresearch_sdpo.yaml`)

- **Model**: Qwen/Qwen3-4B
- **Training GPUs**: 2 (colocated vLLM + FSDP2)
- **Experiment GPUs**: 6 (remote, via SSH)
- **Rollouts per batch**: 8
- **Max response length**: 8192 tokens
- **Learning rate**: 1e-5
- **Epochs**: 30
- **Self-distillation**: Environment feedback always included in teacher prompt (not just when no solution exists)

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

## Observability

### Wandb metrics to watch

- **`feedback_available_fraction`** — Fraction of rollouts with environment feedback. Should be ~1.0. If 0.0, feedback isn't flowing from `agent_loop.py` through the VERL pipeline.
- **`success_sample_fraction`** — Fraction of rollouts that beat `success_reward_threshold`. High values (>0.8) mean most rollouts succeed — the model is learning.
- **`self_distillation_loss`** — KL loss toward the teacher distribution. Should be > 0 when self-distillation is active.
- **`pg_loss`** — Policy gradient loss. Should be > 0 during training.
- **`critic/score/mean`** — Average reward. Should be bounded (not inf/nan). If inf, check `compute_reward()` edge cases.

### Feedback pipeline

Environment feedback from experiment runs flows through:

```
SSHRunner (box1/2/4/5) → RunOutput(stdout, stderr, returncode)
  → _dispatch_experiment() → (reward, feedback_string)
    → output.extra_fields["reward_extra_info"]["feedback"]
      → VERL _postprocess() → non_tensor_batch["feedback"]
        → reward_extra_keys → reward_extra_infos_dict
          → _collect_feedback() → _build_teacher_message()
            → SDPO teacher prompt includes feedback
```

Every failure mode produces a feedback string:
- **SSH failure**: connection error or timeout message
- **Experiment crash** (exit != 0): SSH stderr + last 1000 chars of stdout
- **OOM/segfault**: signal info + partial training output
- **No val_bpb** (exit 0 but missing metric): last 20 lines of stdout
- **Success**: `compute_reward()` feedback with val_bpb delta

### Rollout dumps

Set `trainer.rollout_data_dir` in config. Each training step dumps a JSONL with inputs, outputs, scores, and feedback — useful for debugging what the model is generating.

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
