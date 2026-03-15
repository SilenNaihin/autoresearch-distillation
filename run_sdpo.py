"""Entry point that monkey-patches SDPO trainer to log env metrics and reprompt samples.

The training loop runs inside a Ray remote actor (TaskRunner), so patches applied
in the driver process are invisible to the trainer. We subclass TaskRunner and apply
patches inside run(), which executes in the Ray worker process.
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
    """Apply monkey-patches. Must be called inside the Ray worker process."""
    from verl.trainer.ppo import ray_trainer as _rt
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.metric_utils import compute_data_metrics as _orig_compute_data_metrics

    # ── A. Patch compute_data_metrics to extract env_* keys ──

    def _patched_compute_data_metrics(batch, use_critic=True):
        metrics = _orig_compute_data_metrics(batch, use_critic=use_critic)

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

        # Log feedback strings as a wandb table
        feedback = ntb.get("feedback")
        if feedback is not None:
            import wandb
            rows = [[i, str(f)] for i, f in enumerate(feedback)]
            table = wandb.Table(columns=["sample_idx", "feedback"], data=rows)
            wandb.log({"rollout/feedback": table}, commit=False)

        return metrics

    _rt.compute_data_metrics = _patched_compute_data_metrics

    # ── B. Patch _maybe_build_self_distillation_batch to log reprompt text ──

    _orig_build_sd = RayPPOTrainer._maybe_build_self_distillation_batch

    def _patched_build_sd(self, batch, reward_tensor, reward_extra_infos_dict=None):
        result = _orig_build_sd(self, batch, reward_tensor, reward_extra_infos_dict)
        if result is None:
            return None

        sd_batch, sd_metrics = result

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


class PatchedTaskRunner(TaskRunner):
    def run(self, config):
        _apply_patches()
        return super().run(config)


@hydra.main(config_path="SDPO/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    from verl.utils.device import auto_set_device
    auto_set_device(config)
    task_runner_class = ray.remote(num_cpus=1)(PatchedTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
