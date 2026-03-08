# autoresearch-distillation

Self-improving autonomous ML research agents. Fork and extend [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) with open-source models and online self-distillation.

The original autoresearch lets an AI agent iterate on a small LLM training setup autonomously — modifying code, training for 5 minutes, keeping what works, discarding what doesn't. But it assumes a frontier closed-source model (Claude, GPT-4) as the agent. This project replaces that with an open-source model that **improves itself from its own experimental outcomes**.

## The Idea

```
┌─────────────────────────────────────────────────────────────┐
│                    THE DISTILLATION LOOP                     │
│                                                             │
│   1. Agent (Qwen3 70B) proposes experiment                  │
│   2. train.py runs on training node (5 min)                 │
│   3. Agent evaluates result: keep / discard / crash         │
│   4. (context, diff, outcome) logged as training example    │
│   5. After N successful experiments, LoRA fine-tune agent   │
│   6. Merge adapter, redeploy — agent gets smarter           │
│   7. GOTO 1                                                 │
│                                                             │
│   The agent that designs experiments learns from the        │
│   experiments it designed.                                  │
└─────────────────────────────────────────────────────────────┘
```

The bet: an open-source 70B model, fine-tuned on its own successful ML experiments, will eventually outperform a static frontier model at this narrow task — because it accumulates domain-specific knowledge about what works in this exact training setup.

## Architecture

Designed for a multi-node H100 cluster. Separates concerns across machines:

```
TRAINING NODES (1x H100 each)
├── h100-node-1  →  runs train.py experiments
└── h100-node-2  →  runs train.py experiments (parallel)
    Each experiment: 5 min fixed budget, ~45 GB VRAM

AGENT NODES (2x H100 each)
├── h100-node-3  →  vLLM inference (Qwen3 70B, tensor parallel)
├── h100-node-4  →  LoRA fine-tuning (QLoRA, ~40-50 GB)
└── h100-node-5  →  eval / staging / hot-swap
```

**Training nodes** run the autoresearch experiment loop — they receive a modified `train.py` from the agent, execute it, and return results. Two nodes means two experiments in parallel.

**Agent nodes** handle inference, fine-tuning, and model management. The inference node serves the current best model via vLLM. The fine-tuning node runs QLoRA updates on accumulated experiment data. The staging node lets you A/B test base vs. fine-tuned before promoting.

## Throughput

| Step | Time | Notes |
|------|------|-------|
| Agent generates code edit | 30-90s | 70B w/ tensor parallel on 2xH100 |
| Dispatch to training node | ~5s | SSH + launch |
| Training run | 300s | Fixed 5-min budget |
| Agent reads + evaluates | 10-20s | Short inference call |
| Log + git commit/revert | ~5s | |
| **Total per experiment** | **~6-7 min** | |

With 2 training nodes in parallel: **~16-18 experiments/hour**, **~150+ overnight**.

Fine-tuning pass (LoRA on ~50 examples): **~10-15 min**, triggered every ~5-8 hours.

## Distillation Strategy

### Training Signal

Each experiment produces a structured training example:

```json
{
  "context": {
    "current_train_py": "...",
    "results_tsv": "...",
    "recent_experiments": ["..."]
  },
  "agent_output": {
    "reasoning": "I'll try increasing the model depth...",
    "diff": "--- a/train.py\n+++ b/train.py\n@@ ...",
    "commit_msg": "increase depth from 8 to 12"
  },
  "outcome": {
    "status": "keep",
    "val_bpb": 0.9891,
    "val_bpb_delta": -0.0088,
    "peak_vram_mb": 52340.1
  }
}
```

### What We Train On

- **Positive examples**: experiments with status `keep` (val_bpb improved), weighted by improvement magnitude
- **Negative examples** (optional, DPO-style): pairs of (keep, discard) on similar ideas to teach preference
- **Crash avoidance**: crash examples as negative signal to reduce broken code generation

### Fine-tuning Config

- **Method**: QLoRA (4-bit base + LoRA adapters, rank 8-16)
- **Data**: accumulated (context → successful_modification) pairs
- **Schedule**: trigger after every ~50 new successful experiments
- **Safeguards**:
  - Conservative learning rate (1e-5 to 2e-5)
  - Low LoRA rank to limit capacity for overfitting
  - Hold out 20% of examples for validation
  - Replay buffer of general code data to prevent catastrophic forgetting
  - Always compare against base model on held-out eval before promoting

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA H100 GPU(s) with CUDA 12+
- SSH access between nodes

### Installation

```bash
# Clone
git clone https://github.com/yourusername/autoresearch-distillation.git
cd autoresearch-distillation

# Install dependencies
uv sync

# Download data and train tokenizer (run on training nodes)
uv run prepare.py
```

### Serving the Agent Model

On an agent node (2x H100):

```bash
# Serve Qwen3 70B with vLLM (tensor parallel across 2 GPUs)
vllm serve Qwen/Qwen3-70B \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --port 8000
```

### Running Experiments

```bash
# Configure your cluster in cluster.yaml
cp cluster.example.yaml cluster.yaml
# Edit with your node IPs and SSH keys

# Launch the orchestrator
uv run orchestrator.py --config cluster.yaml
```

### Triggering Distillation

```bash
# Manual trigger (or let the orchestrator auto-trigger)
uv run distill.py \
  --base-model Qwen/Qwen3-70B \
  --examples experiments/training_examples.jsonl \
  --output adapters/run-001/ \
  --lora-rank 16 \
  --lr 1.5e-5
```

## Project Structure

```
autoresearch-distillation/
├── README.md
├── pyproject.toml
├── cluster.example.yaml       # cluster configuration template
├── prepare.py                 # data prep + eval (from upstream, read-only)
├── train.py                   # the file agents modify (from upstream)
├── program.md                 # agent instructions (from upstream)
│
├── orchestrator.py            # multi-node experiment dispatcher
├── agent.py                   # agent interface (vLLM client + tool use)
├── distill.py                 # LoRA fine-tuning pipeline
├── collector.py               # experiment outcome → training example
├── evaluator.py               # A/B eval: base vs fine-tuned model
│
├── experiments/               # accumulated experiment logs
│   ├── training_examples.jsonl
│   └── results.tsv
│
└── adapters/                  # saved LoRA checkpoints
    ├── gen-000/               # base model (no adapter)
    ├── gen-001/               # first distillation pass
    └── gen-002/               # second distillation pass
```

## Known Risks and Open Questions

**Data scarcity** — overnight you get ~150 experiments, maybe ~50 keeps. That's very few examples for fine-tuning a 70B model. Early distillation passes may not help, or may hurt. This is the central gamble.

**Catastrophic forgetting** — aggressive fine-tuning on narrow ML-experiment data could degrade general coding ability. Mitigation: low LoRA rank, conservative LR, replay buffer.

**Reward hacking** — the agent could learn to make tiny safe changes that reliably produce small improvements, rather than exploring boldly. Mitigation: track diversity of experiments, penalize repetition.

**Evaluation cost** — properly evaluating whether a distilled model is better requires burning experiment cycles on A/B testing rather than research. Mitigation: batch evaluations, use staging node.

**When to give up** — if after 3 distillation generations the model hasn't improved, the signal may be too sparse. Fallback: use the accumulated experiment data to fine-tune a smaller model (8B) that's cheaper to iterate on.

## Relationship to Upstream

This project extends [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The core files (`prepare.py`, `train.py`, `program.md`) are kept in sync with upstream. Everything else — the orchestrator, agent interface, distillation pipeline — is new.

## License

MIT
