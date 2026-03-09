# Integration Guide: Experiment Environment

## What's here

Key files:

- **`run_sdpo.py`** — Entry point for SDPO training. Monkey-patches `compute_data_metrics` to log 9 env metrics to wandb, and `_maybe_build_self_distillation_batch` to log decoded teacher reprompt samples as wandb tables.
- **`bash_tool.py`** — Multi-turn editing via mini-swe-agent. `create_isolated_workdir()` + `run_agent_episode()` for data collection, `BashTool` for VERL RL training.
- **`environment.py`** — `ExperimentEnvironment` class (legacy single-turn), `compute_reward()`, `parse_metrics()`.
- **`runners.py`** — `GPUPoolRunner` class. Manages 6 H100 GPUs across 4 machines. Thread-safe, auto-allocates experiments to free GPUs.
- **`prompts.py`** — Shared prompt templates for bash-tool editing.

## Quick start (multi-turn, recommended)

```python
from bash_tool import create_isolated_workdir, run_agent_episode
from environment import compute_reward, parse_metrics
from prompts import SYSTEM_PROMPT, build_instance_prompt
from runners import GPUPoolRunner

pool = GPUPoolRunner()
baseline = open("autoresearch/train.py").read()

# 1. Build prompts
system = SYSTEM_PROMPT
instance = build_instance_prompt(baseline, history_lines=[])

# 2. Run multi-turn editing session
workdir = create_isolated_workdir("autoresearch")
modified_train_py, trajectory = run_agent_episode(workdir, model, system, instance)

# 3. Dispatch to GPU fleet
output = pool.run(modified_train_py)

# 4. Parse metrics, compute reward
metrics = parse_metrics(output.stdout)
val_bpb = metrics.get("val_bpb")
reward, status, feedback = compute_reward(val_bpb, best_val_bpb)
```

## Quick start (legacy single-turn)

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

## Multi-turn editing flow

1. `create_isolated_workdir()` copies autoresearch/ to a temp dir (~750KB)
2. mini-swe-agent's `DefaultAgent` runs in the workdir with a bash tool
3. Model reads `train.py`, makes edits via sed/cat/python, verifies changes
4. Model submits: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
5. Modified `train.py` is read from the workdir
6. Dispatched to a remote GPU via `GPUPoolRunner`
7. `val_bpb` parsed → `compute_reward()` → scalar for RL

## Legacy single-turn flow (env.step)

1. Extracts a unified diff from the model response (looks for ````diff` blocks)
2. Applies the patch to a local copy of `train.py`, snapshots the content, reverts immediately
3. Pipes the modified `train.py` to a remote GPU via SSH
4. Runs `uv run train.py` (~5 min, fixed budget)
5. Parses `val_bpb` and other metrics from stdout
6. Computes reward: `max(0, 0.9979 - val_bpb)` for successes, `0.0` for failures
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

**Cross-process safety**: `GPUPoolRunner` uses `fcntl.flock` file locks (not in-process `Queue`) so multiple Ray workers can share the GPU pool without collisions. Lock files live in `/data/tmp/gpu_locks/`.

## VERL integration

`AutoresearchAgentLoop` in `agent_loop.py` extends VERL's `ToolAgentLoop` for multi-turn bash editing:

1. Model uses a `BashTool` to edit `train.py` across multiple turns
2. VERL's state machine handles: PENDING → GENERATING → PROCESSING_TOOLS → ... → TERMINATED
3. Tool response tokens are masked (`response_mask=0`) — only the model's bash commands are trained on
4. After submission, modified `train.py` is dispatched to `GPUPoolRunner`
5. `compute_reward()` returns `(reward, feedback)` — both flow into SDPO

### SDPO self-distillation feedback

The feedback pipeline connects experiment outcomes to the SDPO teacher prompt:

```python
# agent_loop.py — _dispatch_experiment returns (reward, feedback)
reward, feedback = await self._dispatch_experiment(bash_tool, instance_id)
output.reward_score = reward
# Packs feedback + 9 env metrics (val_bpb, peak_vram_mb, training_seconds, etc.)
env_metrics = {f"env_{k}": ... for k in ENV_KEYS}
output.extra_fields["reward_extra_info"] = {"feedback": feedback, **env_metrics}

# VERL _postprocess() extracts reward_extra_info keys into non_tensor_batch
# ray_trainer.py _collect_feedback() reads non_tensor_batch["feedback"]
# _build_teacher_message() includes feedback in the SDPO teacher prompt
```

Key config settings in `autoresearch_sdpo.yaml`:

```yaml
self_distillation:
  include_environment_feedback: True                # Enable feedback in teacher prompt
  environment_feedback_only_without_solution: False  # Always include (not just when no solution)
  success_reward_threshold: 0.0                      # Any improvement over baseline is "successful"
  dont_reprompt_on_self_success: True                # Don't reprompt already-successful rollouts
```

With `environment_feedback_only_without_solution: False`, the teacher prompt includes **both** a solution (from a sibling rollout that succeeded) **and** environment feedback (val_bpb result, crash info, etc.). This gives the model richer signal than either alone.

### Cross-process GPU locking

`GPUPoolRunner` uses `fcntl.flock` file-based locks in `/data/tmp/gpu_locks/` for cross-process safety. Each Ray worker (separate process) can safely acquire GPU slots without collisions. The lock file per slot is `{slot_name}.lock`.

### Environment metrics logging

`run_sdpo.py` monkey-patches `compute_data_metrics` to extract 9 environment metrics from `non_tensor_batch` and log them to wandb:

- `env/val_bpb/{mean,max,min}` — validation bits per byte
- `env/peak_vram_mb/{mean,max,min}` — peak GPU memory usage
- `env/training_seconds/{mean,max,min}` — training wall time
- `env/total_seconds/{mean,max,min}` — total experiment time
- `env/mfu_percent/{mean,max,min}` — model FLOPs utilization
- `env/total_tokens_M/{mean,max,min}` — tokens processed (millions)
- `env/num_steps/{mean,max,min}` — training steps completed
- `env/num_params_M/{mean,max,min}` — model parameter count (millions)
- `env/depth/{mean,max,min}` — model depth

These flow from `agent_loop.py` → `reward_extra_info` → `non_tensor_batch` → `compute_data_metrics` → wandb.

### Teacher reprompt logging

`run_sdpo.py` also patches `_maybe_build_self_distillation_batch` to decode and log the first 3 teacher reprompt prefixes as a wandb table (`self_distillation/reprompt_samples`). This lets you inspect what context the teacher model sees during self-distillation.

### Memory management (FSDP2 + vLLM colocated)

Running a 32B model on 2 GPUs requires careful memory management. The config uses:
- **FSDP2 with `offload_policy: true`**: Offloads model parameters to CPU via `CPUOffloadPolicy(pin_memory=True)`, freeing GPU memory for vLLM during rollout generation.
- **vLLM `gpu_memory_utilization: 0.70`**: vLLM claims 70% of available GPU memory for KV cache.
- **`free_cache_engine: true`** (VERL default): Releases vLLM's KV cache during training steps, freeing memory for FSDP2.

Config: `configs/autoresearch_sdpo.yaml` (multi-turn enabled with `configs/bash_tool_config.yaml`)

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

Reward is computed against a hardcoded baseline (`BASELINE_VAL_BPB = 0.9979`):

| Status | Reward | When |
|--------|--------|------|
| improvement | `max(0, 0.9979 - val_bpb)` | val_bpb below baseline |
| no_improvement | `0.0` | val_bpb at or above baseline |
| crash / patch_error / parse_error | `0.0` | any failure mode |

No negative rewards. `success_reward_threshold: 0.0` means any improvement over baseline triggers self-distillation.

## What the model sees

**System prompt** (`prompts.py`): Instructs the model to edit `train.py` using bash commands (cat, sed, etc.), verify changes, and submit when satisfied.

**Instance prompt**: Contains the full `train.py` source + recent experiment history (last 20 results from `results.tsv`).

**Tools available**: A single `bash` tool for executing commands in the isolated workdir.

**Expected behavior**: Model reads `train.py`, makes targeted edits, verifies them, then runs `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

## Files written during the loop

- `results.tsv` — tab-separated log of all experiments (iteration, val_bpb, memory_gb, status, description)
- `rollouts/rollouts.jsonl` — full conversation + result metadata per experiment, for SFT/DPO training data
