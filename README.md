# autoresearch-distillation

A framework for training LLM agents via RL (SDPO, GRPO) on **any task improvement loop**. Define a task as a YAML config — what file to edit, how to run experiments, how to score results — and the framework handles rollout generation, experiment dispatch, reward computation, and policy updates.

Built on a [fork of VERL](https://github.com/SilenNaihin/SDPO) that supports [Self-Distillation Policy Optimization (SDPO)](https://self-distillation.github.io/SDPO.html), GRPO, and other RL algorithms.

**[Project Page](https://silennaihin.github.io/autoresearch-distillation/)** | **[W&B (SDPO)](https://wandb.ai/silennai-endflow/autoresearch-sdpo)** | **[W&B (Baselines)](https://wandb.ai/silennai-endflow/autoresearch-baseline)**

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                                │
│                                                                  │
│   1. Model receives prompt (target file + system prompt)         │
│   2. Model thinks step-by-step, then edits file via bash         │
│   3. Model submits: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT   │
│   4. Modified file dispatched to a remote GPU/CPU via SSH        │
│   5. Run command executes (configurable timeout)                 │
│   6. Metrics parsed → reward signal computed                     │
│   7. RL algorithm updates model weights from the rollout         │
│   8. GOTO 1 — model improves at proposing experiments            │
│                                                                  │
│   No separate reward model. No offline data collection.          │
│   The agent trains on live experiment outcomes.                   │
└──────────────────────────────────────────────────────────────────┘
```

## Add a New Task

Create a YAML config in `tasks/` and a source directory with the target file:

```yaml
task:
  name: my-task

  workspace:
    source_dir: my_source        # local dir with target file
    target_file: model.py        # file the agent modifies
    remote_dir: ~/my-task        # where to run on remote machines

  execution:
    run_command: "python3 train.py 2>&1"
    timeout: 300
    needs_gpu: true
    clear_torch_cache: false
    setup_commands:
      - "pip install -r requirements.txt"

  scoring:
    metric: val_loss             # primary metric to optimize
    direction: minimize          # or "maximize"
    baseline: 1.0                # starting value
    parse_mode: key_value        # parses "metric_name: value" from stdout
    metrics: [val_loss, train_time_s]
    display_metrics: [train_time_s]
    degradation_threshold: 0.05

  prompt:
    system: |-
      You are optimizing model.py to minimize val_loss...
    instance: |-
      ## Current {target_file}
      ```{code_lang}
      {file_content}
      ```
    code_lang: python
    file_marker: "## Current {target_file}"

  fleet:
    slots:
      - {host: gpu-box-1, gpu_id: "0", name: gpu1, remote_dir: ~/my-task}
```

Verify it works:

```bash
python test_task_config.py tasks/my_task.yaml
```

Point the training config at it:

```yaml
# configs/agent_loops.yaml
- task_config: tasks/my_task.yaml
```

## Example Tasks

The framework ships with working examples across different domains:

| Task | Metric | Direction | Target File | GPU | Domain |
|------|--------|-----------|-------------|-----|--------|
| [autoresearch](tasks/autoresearch.yaml) | val_bpb | minimize | train.py | yes | ML pretraining |
| [triton-kernel](tasks/examples/triton_kernel.yaml) | kernel_latency_us | minimize | kernel.py | yes | GPU kernel optimization |
| [baseball-pitch](tasks/examples/baseball_pitch.yaml) | rmse_mph | minimize | model.py | no | Tabular ML |
| [voice-agent](tasks/examples/voice_agent.yaml) | eval_score | maximize | system_prompt.txt | no | Prompt engineering |
| [liquid-speedup](tasks/examples/liquid_speedup.yaml) | combined_time_ms | minimize | template.py | no | Code optimization |

All examples have working source directories and have been tested end-to-end on an A100.

```bash
# Run smoke tests on all task configs
python test_task_config.py --all
```

## Flagship: Autoresearch (Qwen3-14B + SDPO)

The framework was developed on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) benchmark — modifying a GPT pretraining script to minimize `val_bpb` within a 5-minute budget on a single H100.

| Method | Model | Experiments | Best | Avg |
|--------|-------|-------------|------|-----|
| Claude Haiku single | Claude Haiku 4.5 | 50 turns (no feedback) | **1.009 (-4.4%)** | 1.070 |
| **SDPO ckpt + ICL** | Qwen3-14B-SDPO | 50 turns (with feedback) | **1.023 (-3.1%)** | 1.071 |
| SDPO (training) | Qwen3-14B | 7,920 rollouts (495 steps × 16) | 1.023 (-3.1%) | — |
| Claude Opus ICL | Claude Opus 4.6 | 50 turns (with feedback) | 1.027 (-2.8%) | — |
| Karpathy Agent | Claude? | 126 | 0.970 (-2.8%) | — |
| SDPO ckpt + single | Qwen3-14B-SDPO | 50 turns (no feedback) | 1.028 (-2.6%) | 1.060 |
| GRPO (training) | Qwen3-14B | 4,864 rollouts (38 steps × 128) | 1.037 (-1.8%) | — |
| Claude Opus single | Claude Opus 4.6 | 30 turns (no feedback) | 1.032 (-2.3%) | — |
| Single-turn | Qwen3-14B | 50 turns (no feedback) | 1.032 (-2.3%) | 1.122 |
| ICL baseline | Qwen3-14B | 50 turns (with feedback) | 1.038 (-1.7%) | 1.066 |

Absolute baselines differ (Karpathy: 0.998, ours: 1.056) due to platform/setup differences. Relative improvements are compared. Claude Haiku single achieves the best absolute score but with high variance (19/50 turns succeeded, avg 1.070); SDPO ckpt + ICL offers similar best-case from a 14B open-source model. GRPO generates 128 rollouts/step (batch=16, group_size=8) but achieves only -1.8% in ~94M tokens vs SDPO's -3.1% in ~137M tokens.

## Architecture

All task-specific configuration lives in `TaskConfig`, loaded from YAML. The agent loops, runners, and tools are fully generic.

```
Training node (2x H100)   — vLLM inference + FSDP2 training (VERL)
Experiment fleet (N GPUs)  — experiment execution via SSH
```

```
                    ┌──────────────────┐
                    │   TaskConfig     │
                    │  (from YAML)     │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼────┐ ┌───────▼──────┐
     │  Agent Loop     │ │Runners │ │  BashTool    │
     │  (SDPO/GRPO)    │ │(SSH)   │ │  (workdir)   │
     └────────┬────────┘ └───┬────┘ └───────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Remote Fleet    │
                    │  (GPU/CPU boxes) │
                    └──────────────────┘
```

- **TaskConfig** (`task_config.py`) — Central config dataclass. Handles prompt generation, metric parsing, reward computation, diff generation, feedback formatting.
- **Agent Loops** (`agent_loop.py`, `agent_loop_grpo.py`) — VERL agent loops for SDPO and GRPO. Multi-turn bash editing + experiment dispatch. Fully generic.
- **Runners** (`runners.py`) — `SSHRunner` dispatches to a single slot. `GPUPoolRunner` manages a fleet with thread-safe locking and dead-box detection.
- **BashTool** (`bash_tool.py`) — VERL tool. Creates isolated workdirs, executes bash commands, reads target files.
- **Reuse Buffer** (`reuse_buffer.py`) — PUCT-based exploration tree. Tracks file versions and metric values for GRPO exploration.
- **Experiment Cache** (`experiment_cache.py`) — Deduplicates experiments by file hash.

## Project Structure

```
autoresearch-distillation/
├── task_config.py              # Central config — loaded from YAML
├── test_task_config.py         # Smoke tests for any task config
│
├── agent_loop.py               # VERL agent loop (SDPO) — multi-turn editing + dispatch
├── agent_loop_grpo.py          # VERL agent loop (GRPO) — with PUCT reuse buffer
├── bash_tool.py                # VERL BashTool — isolated workdir + bash execution
├── runners.py                  # GPUPoolRunner — SSH dispatch to remote fleet
├── environment.py              # RunOutput dataclass
├── prompts.py                  # Thin wrapper around TaskConfig prompt methods
├── reuse_buffer.py             # PUCT exploration tree for GRPO
├── experiment_cache.py         # File-hash deduplication cache
├── reward.py                   # Passthrough reward function for VERL
├── run_sdpo.py                 # Entry point — patches trainer for env metrics logging
├── loop_baseline.py            # ICL + single-turn baseline loop
│
├── tasks/
│   ├── autoresearch.yaml       # Flagship task — GPT pretraining optimization
│   └── examples/
│       ├── baseball_pitch.yaml # Tabular ML — minimize RMSE
│       ├── liquid_speedup.yaml # Code optimization — minimize runtime
│       ├── triton_kernel.yaml  # GPU kernel — minimize latency
│       └── voice_agent.yaml    # Prompt engineering — maximize eval score
│
├── autoresearch/               # Source for autoresearch task (train.py, prepare.py)
├── pitch_model/                # Source for baseball pitch task
├── voice_agent/                # Source for voice agent task
├── liquid/                     # Source for liquid speedup task
├── autokernel/                 # Source for triton kernel task
│
├── configs/
│   ├── autoresearch_sdpo.yaml  # SDPO training config (Qwen3-14B)
│   ├── autoresearch_grpo.yaml  # GRPO training config
│   ├── agent_loops.yaml        # SDPO agent loop registry
│   ├── agent_loops_grpo.yaml   # GRPO agent loop registry
│   └── bash_tool_config.yaml   # VERL tool config
│
├── docs/
│   └── index.html              # Project page (GitHub Pages)
│
└── SDPO/                       # VERL fork with SDPO — submodule
```

## Usage

### Training (SDPO)

```bash
bash scripts/run_training.sh [experiment_name]
```

### Training (GRPO)

```bash
bash scripts/run_grpo.sh [experiment_name]
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
@misc{naihin2026distillloop,
  title   = {RL Training for LLM Agents on Live Task Improvement Loops},
  author  = {Naihin, Silen and Fallah, Kion},
  year    = {2026},
  url     = {https://github.com/SilenNaihin/autoresearch-distillation}
}
```

## License

MIT
