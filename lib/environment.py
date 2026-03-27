"""
Shared utilities for experiment evaluation.

Generic — all task-specific config (metric names, baselines, directions)
comes from TaskConfig. This module just provides RunOutput.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunOutput:
    stdout: str
    stderr: str
    returncode: int
