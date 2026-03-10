# Baseline: Multi-turn In-Context Learning

## What this measures

The core hypothesis of SDPO is that weight updates let the model internalize what improves val_bpb, rather than just accumulating feedback in context. The baseline isolates the in-context learning component: same model, same agent loop, same prompts, same tools — the only difference is no weight updates. If SDPO beats the baseline, the improvement comes from the weights, not from seeing more feedback.

## Architecture

```
box1 (h100_azure)                    a100-backup-1
┌─────────────────────┐              ┌──────────────────┐
│ vLLM (Qwen3-14B)    │              │ Experiment GPU    │
│ port 8000            │   SSH+SCP    │ A100 80GB PCIe   │
│                      │ ──────────> │                   │
│ loop_baseline.py     │             │ uv run train.py   │
│  └─ mini-swe-agent   │ <────────── │ (~5 min/run)     │
│     └─ bash tool     │  stdout      │                  │
└─────────────────────┘              └──────────────────┘
```

- **Inference**: vLLM serves Qwen3-14B on box1 with `--enable-auto-tool-choice --tool-call-parser hermes`
- **Experiments**: Dispatched to A100 only (H100 fleet reserved for SDPO)
- **No weight updates**: The model is frozen. Learning happens purely through feedback accumulation in the prompt.

## Turn anatomy

Each turn is a complete edit-experiment cycle (~5-6 min):

1. **Prompt construction** (~0s): Baseline train.py + accumulated feedback history from all prior turns
2. **Agent episode** (~30-50s): mini-swe-agent gets up to 20 bash tool calls to read/edit train.py in an isolated workdir. Submits via `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
3. **Experiment dispatch** (~5 min): Modified train.py SSH'd to A100, run with `uv run train.py`, output streamed back.
4. **Reward + feedback** (~0s): Parse val_bpb from stdout, compute `reward = max(0, 1.056 - val_bpb)`, generate feedback string, append to history.

The agent does NOT run experiments or see results in real-time. It edits blind, submits, and the outcome feeds into the next turn's prompt.

## Feedback history format

Each turn appends one entry to `feedback_history`. The full history is injected into the next turn's prompt under `## Results from previous attempts`.

**Successful run:**
```
### Attempt 3
Changes:
--- a/train.py
+++ b/train.py
@@ -45,7 +45,7 @@
-    batch_size = 64
+    batch_size = 48
... (truncated at 1000 chars)

val_bpb=1.048000 (baseline=1.056, improved by 0.008000) Memory: 62.3 GB.
```

**Crash:**
```
### Attempt 4
Changes:
<diff truncated to 1000 chars>

Experiment crashed (exit 1):
torch.OutOfMemoryError: CUDA out of memory...
```

Content is diff (max 1000 chars) + outcome (val_bpb or crash trace). No model reasoning is preserved — the agent's chain-of-thought from each episode is discarded. This is intentional: it matches what SDPO sees (reward signal + environment feedback, not the model's own reasoning).

## Why this design

**Same agent interface as SDPO.** Both use mini-swe-agent with identical system/instance prompts (`prompts.py`), identical bash tool mechanics (`bash_tool.py`), and identical experiment dispatch/reward computation (`environment.py`, `runners.py`). The baseline imports these directly. Any change to the shared code affects both equally.

**SDPO comparison (from `configs/autoresearch_sdpo.yaml`):**
| | Baseline | SDPO |
|---|---|---|
| Model | Qwen3-14B (frozen) | Qwen3-14B (training) |
| Agent | mini-swe-agent DefaultAgent | VERL ToolAgentLoop |
| Tool calls/turn | up to 20 (step_limit) | up to 5 (max_assistant_turns) |
| Temperature | 0.7 | 0.7 |
| Learning signal | Feedback in context | Weight updates via GRPO |
| Experiment GPU | A100 (1x) | H100 fleet (6x) |
| Parallelism | Sequential (1 experiment/turn) | Parallel (batch_size=16) |

**Why 20 step_limit vs SDPO's 5:** SDPO is constrained by max_response_length (49k tokens) — each tool call's output eats into the token budget. The baseline has no such constraint since each episode is a fresh API call to vLLM. Giving the baseline more steps is generous, which makes it a stronger comparison (if SDPO still wins with fewer steps, the weight updates are clearly doing something).

**30 turns.** ~2.5-3 hours wall-clock. The model will likely exhaust novel ideas by turn 15-20, and feedback history will approach context limits (32k tokens) around the same time. Each feedback entry is ~1-1.5KB, so 30 entries = ~30-45KB of text = ~10-15k tokens. Combined with the ~8k token base prompt (train.py + system prompt), this stays within the 32k context window but leaves progressively less room for the model's own generation.

**No feedback history cap.** All 30 turns accumulate. This is a deliberate choice: the baseline's advantage over SDPO is its ability to condition on full history. Truncating would handicap it. If context pressure becomes an issue, it'll show up as degraded output quality in later turns — which is itself a useful signal (context-based learning has a ceiling that weight-based learning doesn't).

## Running

```bash
# On box1 (h100_azure):

# 1. Start vLLM (if not already running)
vllm serve Qwen/Qwen3-14B --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.9 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes

# 2. Run baseline
cd ~/autoresearch-distillation
python loop_baseline.py --max-turns 30 --run-name qwen3-14b-baseline
```

Logs to wandb project `autoresearch-baseline`. Compare against `autoresearch-sdpo`.

Output JSONL saved to `outputs/baseline/<run-name>-<timestamp>.jsonl` with full metrics, diffs, and feedback per turn.

## Key files

| File | Role |
|---|---|
| `loop_baseline.py` | Main loop: turn management, feedback accumulation, wandb logging |
| `bash_tool.py` | `create_isolated_workdir()` + `run_agent_episode()` — shared with SDPO |
| `prompts.py` | `SYSTEM_PROMPT` + `build_instance_prompt()` — shared with SDPO |
| `environment.py` | `compute_reward()`, `parse_metrics()`, `BASELINE_VAL_BPB=1.056` — shared with SDPO |
| `runners.py` | `SSHRunner`, `GPUPoolRunner`, fleet definitions — shared with SDPO |
| `agent_loop.py` | SDPO's VERL agent loop (not used by baseline, but uses same bash_tool + prompts) |

## What to look for in results

- **Best val_bpb curve**: Does it plateau? When? That's the ceiling of in-context learning.
- **Crash rate over time**: Does the model learn to avoid OOM/errors from feedback? If crash rate stays flat, feedback isn't helping.
- **Comparison to SDPO**: SDPO trains on 16 parallel trajectories per epoch. The baseline runs 1 sequential trajectory. Even accounting for this, if SDPO finds lower val_bpb faster, weight updates are encoding something that context alone can't.
- **Late-turn degradation**: As feedback history grows, does output quality drop? This would show the context window becoming a bottleneck.
