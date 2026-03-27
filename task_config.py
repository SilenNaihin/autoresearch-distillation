"""
Task configuration for generalized autoresearch distillation.

A TaskConfig defines everything needed to run the SDPO/GRPO training loop
on any autoresearch-style task: what file to edit, how to run experiments,
how to score results, and what prompts to show the agent.

Load from YAML:
    task = TaskConfig.from_yaml("tasks/autoresearch.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WorkspaceConfig:
    source_dir: str          # local dir copied to isolated workdir
    target_file: str         # the file the agent modifies
    remote_dir: str = "~"    # default remote path on experiment machines


@dataclass
class ExecutionConfig:
    run_command: str                                        # e.g. "uv run train.py 2>&1"
    timeout: int = 600                                     # seconds
    needs_gpu: bool = True                                 # CUDA_VISIBLE_DEVICES + nvidia-smi cleanup
    clear_torch_cache: bool = False                        # rm torchinductor cache before each run
    setup_commands: list[str] = field(default_factory=list) # one-time remote box setup


@dataclass
class ScoringConfig:
    metric: str                                                       # primary metric name
    direction: str                                                    # "minimize" or "maximize"
    baseline: float                                                   # starting value
    parse_mode: str = "key_value"                                     # how to parse metrics from stdout
    metrics: list[str] = field(default_factory=list)                  # all metrics to parse
    display_metrics: list[str] = field(default_factory=list)          # shown in feedback
    degradation_threshold: float = 0.05                               # warn if metric degrades by more
    cacheable_crash_patterns: list[str] = field(default_factory=list) # deterministic crashes to cache


@dataclass
class PromptConfig:
    system: str                                       # system prompt
    instance: str                                     # instance prompt template with {target_file}, {file_content}, {code_lang}
    code_lang: str = "python"                         # language for code fence
    file_marker: str = "## Current {target_file}"     # marker for content replacement


@dataclass
class FeedbackConfig:
    """Feedback templates for agent loop. Placeholders: {target_file}, {metric}, {value}, {error}, {degradation}."""
    no_change: str = ""
    sed_failed: str = ""
    success: str = ""
    failure: str = ""
    duplicate: str = ""
    crash: str = ""
    degradation: str = ""


@dataclass
class FleetSlot:
    host: str
    gpu_id: str = "0"
    name: str = ""
    remote_dir: str = ""  # defaults to workspace.remote_dir if empty


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    name: str
    workspace: WorkspaceConfig
    execution: ExecutionConfig
    scoring: ScoringConfig
    prompt: PromptConfig
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    fleet: list[FleetSlot] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        """Load a TaskConfig from a YAML file."""
        path = Path(path)
        if not path.is_absolute():
            # Resolve relative to repo root (same dir as this file)
            path = Path(__file__).resolve().parent / path
        with open(path) as f:
            raw = yaml.safe_load(f)

        t = raw.get("task", raw)  # support both top-level and nested

        workspace = WorkspaceConfig(**t["workspace"])
        execution = ExecutionConfig(**t["execution"])
        scoring = ScoringConfig(**t["scoring"])
        prompt = PromptConfig(**t["prompt"])

        feedback_raw = t.get("feedback", {})
        feedback = _build_feedback(feedback_raw, workspace, scoring)

        fleet = []
        for s in t.get("fleet", {}).get("slots", []):
            slot = FleetSlot(**s)
            if not slot.name:
                slot.name = f"{slot.host}-gpu{slot.gpu_id}"
            if not slot.remote_dir:
                slot.remote_dir = workspace.remote_dir
            fleet.append(slot)

        return cls(
            name=t["name"],
            workspace=workspace,
            execution=execution,
            scoring=scoring,
            prompt=prompt,
            feedback=feedback,
            fleet=fleet,
        )

    # -----------------------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------------------

    def parse_metrics(self, output: str) -> dict:
        """Parse experiment output for metrics using configured parse_mode."""
        if self.scoring.parse_mode == "key_value":
            return _parse_key_value(output, self.scoring.metrics)
        raise ValueError(f"Unknown parse_mode: {self.scoring.parse_mode}")

    def compute_reward(self, value: float | None) -> tuple[float, str, str]:
        """Compute reward from metric value. Returns (reward, status, feedback_str)."""
        s = self.scoring
        if value is None:
            return 0.0, "crash", f"No {s.metric} in output."

        if s.direction == "minimize":
            improvement = s.baseline - value
        else:
            improvement = value - s.baseline
        reward = max(0.0, improvement)

        delta = value - s.baseline
        if delta < 0:
            delta_str = f"{delta:.6f}"
        elif delta > 0:
            delta_str = f"+{delta:.6f}"
        else:
            delta_str = "0.000000"

        info = f"{s.metric}={value:.6f} (baseline={s.baseline}, delta={delta_str})"

        if reward > 0:
            return reward, "improvement", info
        return 0.0, "no_improvement", info

    def is_improvement(self, value: float) -> bool:
        """Check if a metric value is better than baseline."""
        if self.scoring.direction == "minimize":
            return value < self.scoring.baseline
        return value > self.scoring.baseline

    def check_degradation(self, value: float) -> str | None:
        """Return degradation warning string if metric is significantly worse, else None."""
        s = self.scoring
        if s.direction == "minimize":
            degradation = value - s.baseline
        else:
            degradation = s.baseline - value
        if degradation > s.degradation_threshold:
            return self.feedback.degradation.format(
                degradation=f"{degradation:.6f}",
                metric=s.metric,
                target_file=self.workspace.target_file,
            )
        return None

    def is_cacheable_crash(self, error_text: str) -> bool:
        """Check if a crash should be cached (deterministic)."""
        return any(p in error_text for p in self.scoring.cacheable_crash_patterns)

    # -----------------------------------------------------------------------
    # Prompt helpers
    # -----------------------------------------------------------------------

    def build_instance_prompt(self, file_content: str) -> str:
        """Build instance prompt from template."""
        return self.prompt.instance.format(
            target_file=self.workspace.target_file,
            file_content=file_content,
            code_lang=self.prompt.code_lang,
        )

    def replace_file_in_prompt(self, user_content: str, new_content: str) -> str:
        """Replace the target file code block in a prompt with new content."""
        marker = self.prompt.file_marker.format(target_file=self.workspace.target_file)
        full_marker = f"{marker}\n```{self.prompt.code_lang}\n"
        start = user_content.find(full_marker)
        if start < 0:
            return user_content
        code_start = start + len(full_marker)
        end = user_content.find("\n```", code_start)
        if end < 0:
            return user_content
        return user_content[:code_start] + new_content + user_content[end:]

    # -----------------------------------------------------------------------
    # Diff helper
    # -----------------------------------------------------------------------

    def make_diff(self, baseline: str, modified: str) -> str:
        """Generate unified diff with correct file labels."""
        from difflib import unified_diff
        tf = self.workspace.target_file
        if baseline == modified:
            return f"No changes were made to {tf}."
        return "".join(unified_diff(
            baseline.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{tf}", tofile=f"b/{tf}",
        ))

    # -----------------------------------------------------------------------
    # Feedback helpers
    # -----------------------------------------------------------------------

    def fmt_feedback(self, template_name: str, **kwargs) -> str:
        """Format a feedback template by name with kwargs."""
        template = getattr(self.feedback, template_name, "")
        if not template:
            return ""
        return template.format(
            target_file=self.workspace.target_file,
            metric=self.scoring.metric,
            **kwargs,
        )

    # -----------------------------------------------------------------------
    # Fleet helpers
    # -----------------------------------------------------------------------

    def get_fleet_slots(self):
        """Return fleet slots as GPUSlot objects for runners.py."""
        from runners import GPUSlot
        return [
            GPUSlot(
                host=s.host,
                gpu_id=s.gpu_id,
                name=s.name,
                remote_dir=s.remote_dir or self.workspace.remote_dir,
            )
            for s in self.fleet
        ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_key_value(output: str, metric_names: list[str]) -> dict:
    """Parse stdout lines like 'metric_name: value'."""
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        for key in metric_names:
            if line.startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return metrics


def _build_feedback(raw: dict, workspace: WorkspaceConfig, scoring: ScoringConfig) -> FeedbackConfig:
    """Build FeedbackConfig with defaults based on task config."""
    tf = workspace.target_file
    m = scoring.metric
    if scoring.direction == "minimize":
        improve_verb = "reduced"
        metric_desc = m
    else:
        improve_verb = "improved"
        metric_desc = m

    defaults = {
        "no_change": "No changes were made to " + tf + ". You must make new creative edits to improve " + metric_desc + ".",
        "sed_failed": "FAILURE: your sed command failed: {error}\nYou must specify the target file and ensure the pattern matches exactly.",
        "success": "SUCCESS: These changes " + improve_verb + " " + metric_desc + ". This is a good result. Repeat this approach and combine it with additional modifications to further improve " + metric_desc + ".",
        "failure": "These changes did not improve " + metric_desc + ". Continue exploring new and creative modifications to improve " + metric_desc + ".",
        "duplicate": "This code has already been evaluated: {metric}={value}.\nDo not re-attempt this change.",
        "crash": "Do not re-attempt this change.",
        "degradation": "These changes made " + metric_desc + " significantly worse (degraded by {degradation}).",
    }
    return FeedbackConfig(**{k: raw.get(k, defaults[k]) for k in defaults})
