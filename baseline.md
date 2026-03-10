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

Each turn appends a structured result dict to `turn_results`. At prompt time, `format_feedback_prompt()` builds a ranked feedback block:

1. **Top 3 results** (by val_bpb, ascending) — shown with **full untruncated diffs**. The #1 result is marked `*** BEST ***`.
2. **Other successful runs** — one-line summaries: `Attempt N: val_bpb=X (depth=Y, tokens=ZM)`.
3. **Failed attempts** — one-line summaries with classified crash reason: `Attempt N: crash — OOM (out of VRAM)`.

Crash reasons are auto-classified into: OOM, SyntaxError, assertion failure, FlashAttention error, import error, or the last error line.

**Example prompt feedback (after 10 turns):**
```
## Best results so far (full diffs)

### #1 — val_bpb=1.037900 (attempt 11) *** BEST ***
\```diff
--- a/train.py
+++ b/train.py
... (full diff, no truncation)
\```

### #2 — val_bpb=1.057543 (attempt 21)
\```diff
...
\```

## Other successful attempts
- Attempt 4: val_bpb=1.0661 (depth=8, tokens=728M)
- Attempt 5: val_bpb=1.0667 (depth=8, tokens=729M)

## Failed attempts (avoid repeating these)
- Attempt 1: crash — OOM (out of VRAM)
- Attempt 3: crash — SyntaxError in generated code
- Attempt 7: crash — FlashAttention error (likely unsupported head_dim)
```

No model reasoning is preserved — the agent's chain-of-thought from each episode is discarded. This is intentional: it matches what SDPO sees (reward signal + environment feedback, not the model's own reasoning).

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

**No diff truncation.** Diffs are stored and displayed in full. Earlier versions truncated to 1000 chars, which silently dropped critical changes (e.g., DEPTH=12 in the best-performing run). If the model can't see what made a result work, it can't build on it — that's a scaffolding bug, not an ICL limitation.

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

## Design decisions

### Why always start from the original train.py

Each turn sees the unmodified train.py and must propose changes from scratch. This matches SDPO exactly — SDPO also starts from the original each rollout. The difference is where "memory" lives: SDPO stores it in weight updates, the baseline stores it in the context feedback. This is the core comparison. If we let the baseline carry forward modified files, it would be a different (and unfair) experiment.

### Why structured top-K feedback instead of flat history

The first H100 baseline run (30 turns, 2024-03) revealed that flat chronological feedback was ineffective:
- The model found val_bpb=1.0379 at turn 10 (10 simultaneous changes) but never replicated it in 20 subsequent turns.
- Turn 13 attempted the same approach but missed DEPTH=12 and set EMBEDDING_LR=1.45 (2.4x too high) — the winning diff was truncated so the model couldn't see the full recipe.
- Only 1/20 successful runs beat baseline (5%).

The restructured format shows top-3 results with full diffs so the model can see exactly what worked, while compressing failures to one-line summaries. This gives ICL its best possible chance — making the SDPO comparison more rigorous.

### Hardware constraints in system prompt

The first run had a 33% crash rate. Breakdown:
- 60% of crashes: OOM (model didn't know VRAM limit)
- 20% of crashes: FlashAttention head_dim errors (model didn't know kernel constraints)
- 10%: assertion failures (batch size divisibility)
- 10%: syntax errors

Adding explicit hardware constraints to the system prompt is fair — both baseline and SDPO share the same `prompts.py`. This reduces wasted turns on crashes the model could have avoided with information it was never given.
