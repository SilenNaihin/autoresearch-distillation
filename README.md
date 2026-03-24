# Self-Distillation for Automated ML Research

We train **Qwen3-14B** to propose modifications to a GPT pretraining script to reduce validation loss using [Self-Distillation Policy Optimization (SDPO)](https://self-distillation.github.io/SDPO.html). Modified training scripts are run on H100 GPUs and natural language describing experimental outcomes is used as hindsight context for a teacher model which provides a reward signal to a student model.

On [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) benchmark, SDPO achieves **val_bpb = 1.028** during training — and **1.023** when the trained checkpoint is evaluated with ICL, a 3.1% improvement that surpasses the original 2.8% improvement from the Karpathy agent.

**[Project Page](https://silennaihin.github.io/autoresearch-distillation/)** | **[W&B (SDPO)](https://wandb.ai/silennai-endflow/autoresearch-sdpo)** | **[W&B (Baselines)](https://wandb.ai/silennai-endflow/autoresearch-baseline)**

## Results

| Method | Model | Experiments | Best | Avg |
|--------|-------|-------------|------|-----|
| **SDPO ckpt + ICL** | Qwen3-14B-SDPO | 50 turns (with feedback) | **1.023 (−3.1%)** | 1.071 |
| Karpathy Agent | Claude? | 126 | 0.970 (−2.8%) | — |
| SDPO (training) | Qwen3-14B | 960 rollouts (60 steps × 16) | 1.028 (−2.6%) | 1.073 |
| SDPO ckpt + single | Qwen3-14B-SDPO | 50 turns (no feedback) | 1.028 (−2.6%) | 1.060 |
| Single-turn | Qwen3-14B | 50 turns (no feedback) | 1.032 (−2.3%) | 1.122 |
| ICL baseline | Qwen3-14B | 50 turns (with feedback) | 1.038 (−1.7%) | 1.066 |

Absolute baselines differ (Karpathy: 0.998, ours: 1.056) due to platform/setup differences. Relative improvements are compared.

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                     SDPO TRAINING LOOP                           │
│                                                                  │
│   1. Model receives prompt (train.py + system prompt)            │
│   2. Model thinks step-by-step, then edits train.py via bash     │
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

SDPO does not need a successful demonstration for an advantage signal — it only needs the correct solution to be more likely under the teacher model compared to the student model, providing dense supervision even when all rollouts fail. Weight updates encode exploration patterns that compound across training — increasing effective memory and feedback adherence compared to context-bounded approaches.

## Three Approaches

**SDPO** — The model generates rollouts editing `train.py` via bash tool calls. Experiments execute on remote H100s. A reward signal is provided via hindsight context from a teacher model, flowing to the model's weights via policy gradient updates.

**ICL Baseline** — Same model, same tools, no weight updates. The model accumulates experiment feedback in its context window across 50 sequential turns. All learning happens through context conditioning. Best: **val_bpb = 1.038**.

**Single-turn Baseline** — N independent single-shot calls with no history and no feedback. Measures the entropy of the sampling distribution. At 50 turns: **val_bpb = 1.032** — actually beating ICL, suggesting the feedback loop adds consistency but not fundamentally better outcomes.

## Architecture

```
box3 (2x H100 NVL)    — vLLM inference + FSDP2 training
box2, box4, box5       — experiment execution (5 H100 slots)
```

[VERL](https://github.com/volcengine/verl) (via SDPO fork) handles the RL training loop. Our custom `AutoresearchAgentLoop` extends VERL's `ToolAgentLoop` — the model uses a bash tool to edit `train.py` across multiple turns, then the modified script is dispatched to a GPU and the experiment reward flows back into training.

`GPUPoolRunner` manages 5 remote H100s via SSH with thread-safe file-based locking.

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
├── loop_baseline.py           # ICL + single-turn baseline loop
│
├── configs/
│   ├── autoresearch_sdpo.yaml # Hydra config for SDPO training (Qwen3-14B)
│   ├── bash_tool_config.yaml  # VERL tool config for bash tool
│   └── agent_loops.yaml       # Agent loop registry
├── docs/
│   └── index.html             # Project page (GitHub Pages)
│
├── autoresearch/              # Upstream submodule (train.py, prepare.py) — read-only
└── SDPO/                      # SDPO/VERL fork — submodule
```

## Usage

### 1. Run SDPO training

```bash
bash scripts/run_training.sh [experiment_name]
```

### 2. Run baselines

```bash
# ICL (multi-turn with feedback)
python loop_baseline.py --max-turns 50 --mode agent --run-name qwen3-14b-icl

# Single-turn (no feedback)
python loop_baseline.py --max-turns 50 --mode agent --single-turn --run-name qwen3-14b-single
```

### Key config (`configs/autoresearch_sdpo.yaml`)

- **Model**: Qwen/Qwen3-14B with YaRN rope_scaling (factor=2.0 for 64k context)
- **Training**: 2 GPUs, colocated vLLM + FSDP2 with CPU offloading
- **Batch size**: 16 rollouts per step
- **Context**: 16k prompt + 49k response = 65k total
- **Chain of thought**: enabled (`enable_thinking: true`)

## Built On

- **[Autoresearch](https://github.com/karpathy/autoresearch)** (Karpathy) — Single-file GPT pretraining script. The benchmark: modify `train.py` to minimize `val_bpb` within a 5-minute budget on a single H100.
- **[SDPO](https://self-distillation.github.io/SDPO.html)** (Hubotter et al., 2026) — Self-Distillation Policy Optimization. Converts tokenized environment feedback into a dense learning signal via self-teacher distillation.

## Citation

```bibtex
@misc{naihin2026sdpoautoresearch,
  title   = {Self-Distillation for Automated ML Research},
  author  = {Naihin, Silen and Fallah, Kion},
  year    = {2026},
  url     = {https://github.com/SilenNaihin/autoresearch-distillation}
}
```

## License

MIT
