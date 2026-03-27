"""
Smoke tests for the generalized task config system.

Verifies that any task config YAML can be loaded and that all core operations
(prompt generation, metric parsing, reward computation, diff generation,
workdir creation) work correctly.

Run:
    python test_task_config.py
    python test_task_config.py tasks/autoresearch.yaml
    python test_task_config.py --all
"""

import argparse
import sys
import tempfile
import os
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(__file__))

from task_config import TaskConfig


def test_load(path: str) -> TaskConfig:
    """Test that a task config loads without errors."""
    task = TaskConfig.from_yaml(path)
    assert task.name, f"Task name is empty in {path}"
    assert task.workspace.source_dir, f"source_dir is empty in {path}"
    assert task.workspace.target_file, f"target_file is empty in {path}"
    assert task.execution.run_command, f"run_command is empty in {path}"
    assert task.scoring.metric, f"metric is empty in {path}"
    assert task.scoring.direction in ("minimize", "maximize"), f"Invalid direction: {task.scoring.direction}"
    assert task.prompt.system, f"system prompt is empty in {path}"
    assert task.prompt.instance, f"instance template is empty in {path}"
    print(f"  [load] OK — name={task.name}, metric={task.scoring.metric}, direction={task.scoring.direction}")
    return task


def test_prompt_generation(task: TaskConfig):
    """Test prompt generation with synthetic file content."""
    sample_content = f"# Sample {task.workspace.target_file}\nprint('hello')\n"
    prompt = task.build_instance_prompt(sample_content)
    assert task.workspace.target_file in prompt, "target_file not in generated prompt"
    assert sample_content in prompt, "file_content not in generated prompt"
    assert task.prompt.code_lang in prompt, "code_lang not in generated prompt"
    print(f"  [prompt] OK — {len(prompt)} chars, contains target_file + content + code_lang")

    # Test replacement
    replaced = task.replace_file_in_prompt(prompt, "# REPLACED CONTENT")
    assert "# REPLACED CONTENT" in replaced, "Replacement failed"
    assert sample_content not in replaced, "Old content still in prompt after replacement"
    print(f"  [replace] OK — content replacement works")


def test_metric_parsing(task: TaskConfig):
    """Test metric parsing with synthetic output."""
    # Build synthetic output with all configured metrics
    lines = []
    for i, m in enumerate(task.scoring.metrics):
        lines.append(f"{m}: {1.0 + i * 0.1:.4f}")
    output = "\n".join(lines)

    metrics = task.parse_metrics(output)
    assert task.scoring.metric in metrics, f"Primary metric {task.scoring.metric} not parsed"
    assert len(metrics) == len(task.scoring.metrics), f"Expected {len(task.scoring.metrics)} metrics, got {len(metrics)}"
    print(f"  [parse] OK — parsed {len(metrics)} metrics from synthetic output")


def test_reward_computation(task: TaskConfig):
    """Test reward computation for improvement, no_improvement, and crash cases."""
    baseline = task.scoring.baseline

    # Crash case
    reward, status, info = task.compute_reward(None)
    assert reward == 0.0 and status == "crash", f"Crash case failed: {reward}, {status}"

    # Improvement case
    if task.scoring.direction == "minimize":
        improved_val = baseline - 0.01
        worse_val = baseline + 0.01
    else:
        improved_val = baseline + 0.01
        worse_val = baseline - 0.01

    reward, status, info = task.compute_reward(improved_val)
    assert reward > 0 and status == "improvement", f"Improvement case failed: {reward}, {status}"
    assert task.scoring.metric in info, f"Metric name not in info: {info}"
    print(f"  [reward] improvement: reward={reward:.6f}, info={info}")

    # No improvement case
    reward, status, info = task.compute_reward(worse_val)
    assert reward == 0.0 and status == "no_improvement", f"No improvement case failed: {reward}, {status}"
    print(f"  [reward] no_improvement: reward={reward:.6f}, info={info}")

    # Degradation check
    if task.scoring.direction == "minimize":
        bad_val = baseline + task.scoring.degradation_threshold + 0.01
    else:
        bad_val = baseline - task.scoring.degradation_threshold - 0.01
    deg = task.check_degradation(bad_val)
    assert deg is not None, "Degradation check should have triggered"
    print(f"  [reward] degradation detected: {deg[:80]}...")

    # is_improvement
    assert task.is_improvement(improved_val), "is_improvement returned False for improved value"
    assert not task.is_improvement(worse_val), "is_improvement returned True for worse value"
    print(f"  [reward] OK — all reward cases correct")


def test_diff_generation(task: TaskConfig):
    """Test diff generation with correct labels."""
    baseline = f"# Original {task.workspace.target_file}\nx = 1\n"
    modified = f"# Original {task.workspace.target_file}\nx = 2\n"

    diff = task.make_diff(baseline, modified)
    assert f"a/{task.workspace.target_file}" in diff, "Diff missing a/ label"
    assert f"b/{task.workspace.target_file}" in diff, "Diff missing b/ label"
    assert "-x = 1" in diff, "Diff missing removed line"
    assert "+x = 2" in diff, "Diff missing added line"
    print(f"  [diff] OK — correct labels and content")

    # No changes
    no_diff = task.make_diff(baseline, baseline)
    assert "No changes" in no_diff, f"No-change diff should say 'No changes': {no_diff}"
    print(f"  [diff] no-change case OK")


def test_feedback_templates(task: TaskConfig):
    """Test that all feedback templates are non-empty and formattable."""
    fb = task.feedback

    assert fb.no_change, "no_change feedback is empty"
    assert fb.success, "success feedback is empty"
    assert fb.failure, "failure feedback is empty"
    assert fb.crash, "crash feedback is empty"

    # Test formatting
    sed_msg = task.fmt_feedback("sed_failed", error="sed: expression error")
    assert "sed: expression error" in sed_msg, f"sed_failed formatting failed: {sed_msg}"

    dup_msg = task.fmt_feedback("duplicate", value="1.050000")
    assert "1.050000" in dup_msg, f"duplicate formatting failed: {dup_msg}"

    print(f"  [feedback] OK — all templates non-empty and formattable")


def test_fleet(task: TaskConfig):
    """Test fleet slot configuration."""
    if not task.fleet:
        print(f"  [fleet] SKIP — no fleet configured")
        return

    for slot in task.fleet:
        assert slot.host, "Fleet slot missing host"
        assert slot.name, "Fleet slot missing name"
        assert slot.remote_dir, "Fleet slot missing remote_dir"
    print(f"  [fleet] OK — {len(task.fleet)} slots configured")


def test_cacheable_crash(task: TaskConfig):
    """Test cacheable crash pattern matching."""
    for pattern in task.scoring.cacheable_crash_patterns:
        assert task.is_cacheable_crash(f"Error: {pattern} blah"), f"Pattern '{pattern}' should match"
    assert not task.is_cacheable_crash("normal error"), "Non-matching should return False"
    print(f"  [crash] OK — {len(task.scoring.cacheable_crash_patterns)} patterns tested")


def run_smoke_test(path: str) -> bool:
    """Run all smoke tests on a single task config."""
    print(f"\n{'='*60}")
    print(f"Smoke test: {path}")
    print(f"{'='*60}")

    try:
        task = test_load(path)
        test_prompt_generation(task)
        test_metric_parsing(task)
        test_reward_computation(task)
        test_diff_generation(task)
        test_feedback_templates(task)
        test_fleet(task)
        test_cacheable_crash(task)
        print(f"\n  ALL PASSED for {task.name}")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_all_task_configs() -> list[str]:
    """Find all task config YAML files in the tasks/ directory."""
    repo_root = Path(__file__).resolve().parent
    configs = []
    for p in sorted(repo_root.glob("tasks/**/*.yaml")):
        configs.append(str(p))
    return configs


def main():
    parser = argparse.ArgumentParser(description="Smoke test task configs")
    parser.add_argument("configs", nargs="*", help="Task config YAML files to test")
    parser.add_argument("--all", action="store_true", help="Test all task configs in tasks/")
    args = parser.parse_args()

    if args.all:
        configs = find_all_task_configs()
    elif args.configs:
        configs = args.configs
    else:
        configs = find_all_task_configs()

    if not configs:
        print("No task configs found. Create YAML files in tasks/")
        sys.exit(1)

    print(f"Testing {len(configs)} task config(s)...")

    results = {}
    for path in configs:
        results[path] = run_smoke_test(path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for path, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {Path(path).name}")
    print(f"\n{passed} passed, {failed} failed out of {len(results)} total")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
