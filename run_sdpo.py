"""Entry point that monkey-patches SDPO trainer to log env metrics and reprompt samples."""

import math
import functools

import hydra

from verl.trainer.main_ppo import main as _original_main, run_ppo
from verl.trainer.ppo import ray_trainer as _ray_trainer_module
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.metric_utils import compute_data_metrics as _orig_compute_data_metrics

# ── A. Monkey-patch compute_data_metrics to extract env_* keys ──

ENV_KEYS = (
    "env_val_bpb", "env_peak_vram_mb", "env_training_seconds", "env_total_seconds",
    "env_mfu_percent", "env_total_tokens_M", "env_num_steps", "env_num_params_M", "env_depth",
)


def _patched_compute_data_metrics(batch, use_critic=True):
    metrics = _orig_compute_data_metrics(batch, use_critic=use_critic)

    # Extract env_* from non_tensor_batch (populated via reward_extra_keys)
    ntb = getattr(batch, "non_tensor_batch", {})
    for key in ENV_KEYS:
        values = ntb.get(key)
        if values is None:
            continue
        # values is a list/array of per-sample floats; filter NaN
        valid = [float(v) for v in values if not math.isnan(float(v))]
        if not valid:
            continue
        short = key.replace("env_", "")
        metrics[f"env/{short}/mean"] = sum(valid) / len(valid)
        metrics[f"env/{short}/max"] = max(valid)
        metrics[f"env/{short}/min"] = min(valid)

    # Log feedback strings as a wandb table
    feedback = ntb.get("feedback")
    if feedback is not None:
        import wandb
        rows = [[i, str(f)] for i, f in enumerate(feedback)]
        table = wandb.Table(columns=["sample_idx", "feedback"], data=rows)
        wandb.log({"rollout/feedback": table}, commit=False)

    return metrics


# Patch in the module namespace where it's imported
_ray_trainer_module.compute_data_metrics = _patched_compute_data_metrics

# ── B. Monkey-patch _maybe_build_self_distillation_batch to log reprompt text ──

_orig_build_sd = RayPPOTrainer._maybe_build_self_distillation_batch


def _patched_build_sd(self, batch, reward_tensor, reward_extra_infos_dict=None):
    result = _orig_build_sd(self, batch, reward_tensor, reward_extra_infos_dict)
    if result is None:
        return None

    sd_batch, sd_metrics = result

    # Decode teacher reprompt prefixes and log as wandb table
    import wandb

    teacher_ids = sd_batch.batch["teacher_input_ids"]
    response_len = batch.batch["responses"].shape[1]
    reprompt_len = teacher_ids.shape[1] - response_len

    num_samples = min(3, teacher_ids.shape[0])
    rows = []
    for i in range(num_samples):
        prefix_ids = teacher_ids[i, :reprompt_len]
        prefix_ids = prefix_ids[prefix_ids != 0]
        text = self.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        rows.append([i, text])

    table = wandb.Table(columns=["sample_idx", "reprompt_text"], data=rows)
    wandb.log({"self_distillation/reprompt_samples": table}, step=self.global_steps)

    return result


RayPPOTrainer._maybe_build_self_distillation_batch = _patched_build_sd

# ── Entry point ──


@hydra.main(config_path="SDPO/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    from verl.utils.device import auto_set_device
    auto_set_device(config)
    run_ppo(config)


if __name__ == "__main__":
    main()
