# autoresearch-distillation

An open-source repo for applying continual learning to [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) benchmark. Built on a [fork of VERL](https://github.com/SilenNaihin/SDPO) that supports [Self-Distillation Policy Optimization (SDPO)](https://self-distillation.github.io/SDPO.html), GRPO, and other RL algorithms for training LLM agents on live experiment outcomes.

The agent proposes modifications to a GPT pretraining script, runs real experiments on H100 GPUs, and learns from the results — updating its weights to become a better ML researcher over time.

**[Project Page](https://silennaihin.github.io/autoresearch-distillation/)** | **[W&B (SDPO)](https://wandb.ai/silennai-endflow/autoresearch-sdpo)** | **[W&B (Baselines)](https://wandb.ai/silennai-endflow/autoresearch-baseline)**

## Current Results (Qwen3-14B + SDPO)

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
│                     TRAINING LOOP                                │
│                                                                  │
│   1. Model receives prompt (train.py + system prompt)            │
│   2. Model thinks step-by-step, then edits train.py via bash     │
│   3. Model submits: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT   │
│   4. Modified train.py dispatched to a remote H100 via SSH       │
│   5. uv run train.py executes (5 min fixed budget)               │
│   6. val_bpb parsed → reward signal computed                     │
│   7. RL algorithm updates model weights from the rollout         │
│   8. GOTO 1 — model improves at proposing experiments            │
│                                                                  │
│   No separate reward model. No offline data collection.          │
│   The agent trains on live experiment outcomes.                   │
└──────────────────────────────────────────────────────────────────┘
```

The VERL fork supports multiple RL algorithms. SDPO provides dense supervision by using the model conditioned on experiment feedback as a self-teacher — it doesn't need a successful demonstration, only that the correct solution is more likely under the teacher than the student. GRPO and other algorithms supported by VERL work out of the box.

## Architecture

The repo is designed around a GPU pool pattern: one machine runs inference + training, while a fleet of remote GPUs execute experiments in parallel.

```
Training node (2x H100)   — vLLM inference + FSDP2 training (VERL)
Experiment fleet (N GPUs)  — experiment execution via SSH
```

`AutoresearchAgentLoop` extends VERL's `ToolAgentLoop` — the model uses a bash tool to edit `train.py` across multiple turns, then the modified script is dispatched to a GPU and the experiment reward flows back into training.

`GPUPoolRunner` manages remote GPUs via SSH with thread-safe file-based locking for concurrent dispatch.

## Project Structure

```
autoresearch-distillation/
├── agent_loop.py              # VERL agent loop — multi-turn bash editing + experiment dispatch
├── bash_tool.py               # VERL BashTool — isolated workdir + bash execution
├── environment.py             # RunOutput, parse_metrics(), compute_reward()
├── runners.py                 # GPUPoolRunner — SSH dispatch to remote H100s
├── prompts.py                 # System prompt + instance prompt builder
├── reward.py                  # Passthrough reward function for VERL
├── run_sdpo.py                # Entry point — patches trainer for env metrics logging
├── loop_baseline.py           # ICL + single-turn baseline loop
│
├── configs/
│   ├── autoresearch_sdpo.yaml # SDPO config (Qwen3-14B)
│   ├── bash_tool_config.yaml  # VERL tool config for bash tool
│   └── agent_loops.yaml       # Agent loop registry
├── docs/
│   └── index.html             # Project page (GitHub Pages)
│
├── autoresearch/              # Upstream submodule (train.py, prepare.py) — read-only
└── SDPO/                      # VERL fork with SDPO — submodule
```

## Usage

### Training

```bash
bash scripts/run_training.sh [experiment_name]
```

### Baselines

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
- **[VERL](https://github.com/volcengine/verl)** — RL training framework for LLMs. Our fork adds SDPO, agentic tool use, and multi-turn rollouts.

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
