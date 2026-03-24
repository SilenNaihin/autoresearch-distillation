"""Entry point for GRPO baseline training.

Simpler than run_sdpo.py — no self-distillation patches needed.
Still logs env metrics (val_bpb, VRAM, etc.) to wandb.
"""

import math

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo

ENV_KEYS = (
    "env_val_bpb", "env_peak_vram_mb", "env_training_seconds", "env_total_seconds",
    "env_mfu_percent", "env_total_tokens_M", "env_num_steps", "env_num_params_M", "env_depth",
    "env_novel",
)


def _apply_patches():
    """Patch compute_data_metrics to extract env_* keys from non_tensor_batch."""
    from verl.trainer.ppo import ray_trainer as _rt
    from verl.trainer.ppo.metric_utils import compute_data_metrics as _orig

    def _patched_compute_data_metrics(batch, use_critic=True):
        metrics = _orig(batch, use_critic=use_critic)

        ntb = getattr(batch, "non_tensor_batch", {})
        for key in ENV_KEYS:
            values = ntb.get(key)
            if values is None:
                continue
            valid = [float(v) for v in values if not math.isnan(float(v))]
            if not valid:
                continue
            short = key.replace("env_", "")
            metrics[f"env/{short}/mean"] = sum(valid) / len(valid)
            metrics[f"env/{short}/max"] = max(valid)
            metrics[f"env/{short}/min"] = min(valid)

        return metrics

    _rt.compute_data_metrics = _patched_compute_data_metrics


class PatchedTaskRunner(TaskRunner):
    def run(self, config):
        _apply_patches()
        return super().run(config)


@hydra.main(config_path="../SDPO/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    from verl.utils.device import auto_set_device
    auto_set_device(config)
    task_runner_class = ray.remote(num_cpus=1)(PatchedTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
