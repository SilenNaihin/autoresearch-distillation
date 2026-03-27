"""
Prompt utilities for autoresearch distillation.

All task-specific prompt text lives in the task config YAML.
This module provides backward-compatible helpers that delegate to TaskConfig.
"""

from __future__ import annotations

from task_config import TaskConfig


def build_instance_prompt(task: TaskConfig, file_content: str) -> str:
    """Build the task prompt with current file content."""
    return task.build_instance_prompt(file_content)


def replace_file_in_prompt(task: TaskConfig, user_content: str, new_content: str) -> str:
    """Replace the target file code block in a prompt with new content."""
    return task.replace_file_in_prompt(user_content, new_content)
