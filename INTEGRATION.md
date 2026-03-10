# Integration Guide

## Key Files

- **`agent_loop.py`** — `AutoresearchAgentLoop` extends VERL's `ToolAgentLoop`. Handles multi-turn bash editing, experiment dispatch, reward computation, and behavioral feedback.
- **`bash_tool.py`** — `BashTool(BaseTool)` for VERL's tool system. Creates isolated workdirs, executes bash commands, detects submission signal.
- **`environment.py`** — `RunOutput` dataclass, `parse_metrics()`, `compute_reward()`.
- **`runners.py`** — `GPUPoolRunner` manages 5 H100s via SSH with file-based locking.
- **`prompts.py`** — System prompt + `build_instance_prompt()` for constructing task prompts.
- **`run_sdpo.py`** — Entry point. Monkey-patches `compute_data_metrics` for env metric logging and `_maybe_build_self_distillation_batch` for teacher reprompt logging.

## Training Flow

1. VERL creates `AutoresearchAgentLoop` per rollout
2. Agent loop pre-creates a `BashTool` instance (isolated workdir)
3. Model generates `<think>` reasoning, then tool calls (`sed` edits to train.py)
4. Model submits: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
5. `_handle_processing_tools_state` detects submission → TERMINATED
6. `_dispatch_experiment()` reads modified train.py, dispatches to `GPUPoolRunner`
7. Metrics parsed, reward computed: `max(0, 1.0 - val_bpb)`
8. Feedback (diff + results + behavioral notes) flows into SDPO teacher prompt

## SDPO Self-Distillation Feedback

```python
# agent_loop.py — _dispatch_experiment returns (reward, feedback)
reward, feedback = await self._dispatch_experiment(bash_tool, instance_id)
output.reward_score = reward
output.extra_fields["reward_extra_info"] = {"feedback": feedback, **env_metrics}

# VERL pipeline:
# _postprocess() → non_tensor_batch["feedback"]
# _collect_feedback() → _build_teacher_message()
# Teacher sees: prompt + feedback (diff, val_bpb result, behavioral notes)
```

With `dont_reprompt_on_self_success: True` and `n: 1`, the teacher only sees the prompt + feedback — no correct solution is provided.

## Behavioral Feedback (configurable)

When `behavioral_feedback: true` in `configs/agent_loops.yaml`, the agent loop tracks per-trajectory stats and appends warnings to the teacher feedback:

- Commands that don't modify train.py (noop tool calls)
- Malformed tool call arguments (JSON parse errors)
- Submitting without making any changes

## GPU Fleet

5 slots across 3 machines (box3 reserved for training):

| Slot | Host | GPU | Remote dir |
|------|------|-----|------------|
| box2-gpu0 | h100-dev-box-2 | 0 | ~/autoresearch |
| box4-gpu0 | h100-dev-box-4 | 0 | ~/autoresearch-gpu0 |
| box4-gpu1 | h100-dev-box-4 | 1 | ~/autoresearch-gpu1 |
| box5-gpu0 | h100-dev-box-5 | 0 | ~/autoresearch-gpu0 |
| box5-gpu1 | h100-dev-box-5 | 1 | ~/autoresearch-gpu1 |

Cross-process safety via `fcntl.flock` file locks in `/data/tmp/gpu_locks/`. Dead-box detection with 5-minute cooldown on SSH failures.

## Environment Metrics (wandb)

`run_sdpo.py` logs 9 env metrics per step: `val_bpb`, `peak_vram_mb`, `training_seconds`, `total_seconds`, `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth`. Each as `env/{key}/{mean,max,min}`.

## YaRN Rope Scaling

Qwen3-14B has 32k native context, but we use 64k `max_model_len`. Launch scripts patch the HF cached `config.json` on disk to add YaRN (`factor: 2.0`), so both FSDP and vLLM pick it up natively. The `hf_overrides` in `engine_kwargs.vllm` provides a safety net for vLLM.
