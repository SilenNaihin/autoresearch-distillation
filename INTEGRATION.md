# Integration Guide: Experiment Environment

## What's here

Two files you care about:

- **`environment.py`** — `ExperimentEnvironment` class. Handles prompts, diff extraction, patch application, reward computation, rollout logging.
- **`runners.py`** — `GPUPoolRunner` class. Manages 6 H100 GPUs across 4 machines. Thread-safe, auto-allocates experiments to free GPUs.

## Quick start

```python
from environment import ExperimentEnvironment
from runners import GPUPoolRunner

pool = GPUPoolRunner()          # 6 GPUs, blocks if all busy
env = ExperimentEnvironment(pool)

# 1. Get the prompt for the policy model
system_prompt, user_message = env.get_prompt()

# 2. Generate a response from the policy (your vLLM call)
response = your_model.generate(system_prompt, user_message)

# 3. Run the experiment — dispatches to a free GPU, ~5 min
result = env.step(response)

# 4. Use the result
result.reward      # float — scalar for RL
result.status      # "improvement" | "no_improvement" | "crash" | "patch_error" | "parse_error"
result.val_bpb     # float | None — the metric (lower is better)
result.feedback    # str — text feedback (goes back into conversation for multi-turn)
result.diff        # str | None — the extracted diff
result.metrics     # dict — all parsed metrics from train.py output
```

## What `env.step()` does internally

1. Extracts a unified diff from the model response (looks for ````diff` blocks)
2. Applies the patch to a local copy of `train.py`, snapshots the content, reverts immediately
3. Pipes the modified `train.py` to a remote GPU via SSH
4. Runs `uv run train.py` (~5 min, fixed budget)
5. Parses `val_bpb` and other metrics from stdout
6. Computes reward: improvement → positive (scaled by delta), no improvement → small negative, crash → -1.0
7. Logs to `results.tsv` and `rollouts/rollouts.jsonl`

## Concurrency

`GPUPoolRunner` is thread-safe. You can call `env.step()` from multiple threads — the pool queues requests and dispatches to the next free GPU.

```python
from concurrent.futures import ThreadPoolExecutor

# Run up to 6 experiments in parallel (one per GPU)
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(env.step, response) for response in batch]
    results = [f.result() for f in futures]
```

**Important**: `env.step()` modifies shared state (`best_val_bpb`, `iteration`, log files). If you run parallel experiments, wrap state access or use one environment per thread. The runner itself is safe — the environment state tracking isn't locked.

## VERL integration

For VERL's reward pipeline, wrap `env.step()` as a `compute_score` function:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    result = env.step(solution_str)
    return {
        "score": result.reward,
        "status": result.status,
        "val_bpb": result.val_bpb,
    }
```

## GPU fleet

6 slots across 4 machines (box3 reserved for vLLM inference):

| Slot | Host | GPU | Remote dir |
|------|------|-----|------------|
| box1-gpu0 | h100_azure | 0 | ~/autoresearch |
| box2-gpu0 | h100-dev-box-2 | 0 | ~/autoresearch |
| box4-gpu0 | h100-dev-box-4 | 0 | ~/autoresearch-gpu0 |
| box4-gpu1 | h100-dev-box-4 | 1 | ~/autoresearch-gpu1 |
| box5-gpu0 | h100-dev-box-5 | 0 | ~/autoresearch-gpu0 |
| box5-gpu1 | h100-dev-box-5 | 1 | ~/autoresearch-gpu1 |

All machines are set up and verified. `test_e2e.py` confirms 6/6 passing.

## Reward structure

| Status | Reward | When |
|--------|--------|------|
| improvement | `+delta * 100` | val_bpb beat the best (0.01 improvement → reward 1.0) |
| no_improvement | `-delta * 50` | val_bpb equal or worse |
| crash | `-1.0` | script crashed, OOM, segfault, timeout |
| patch_error | `-1.0` | diff couldn't be applied to train.py |
| parse_error | `-1.0` | no diff found in model response |

## What the model sees

**System prompt**: "You are an autonomous ML researcher optimizing a GPT pretraining script..." — instructs the model to output reasoning + a single unified diff.

**User message**: Contains the full `train.py` source + recent experiment history (last 20 results from `results.tsv`).

**Expected output**: Reasoning text followed by a ````diff` block with a unified patch modifying only `train.py`.

## Files written during the loop

- `results.tsv` — tab-separated log of all experiments (iteration, val_bpb, memory_gb, status, description)
- `rollouts/rollouts.jsonl` — full conversation + result metadata per experiment, for SFT/DPO training data
