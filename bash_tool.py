"""
Bash tool for multi-turn agent editing of train.py.

BashTool(BaseTool) is used by VERL's ToolAgentLoop for RL training.
Creates an isolated copy of autoresearch/, lets the agent edit train.py
via bash commands, then reads back the result.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Submission signal — matches mini-swe-agent's LocalEnvironment protocol
SUBMIT_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# ---------------------------------------------------------------------------
# Isolated workdir
# ---------------------------------------------------------------------------

def create_isolated_workdir(autoresearch_dir: str = "autoresearch") -> str:
    """Copy autoresearch/ to a temp dir for isolated editing.

    Copies train.py, prepare.py, pyproject.toml, uv.lock, .python-version, etc.
    Data cache lives in ~/.cache/ so each copy is ~750KB.

    Returns the temp dir path.
    """
    src = os.path.abspath(autoresearch_dir)
    tmpdir = tempfile.mkdtemp(prefix="autoresearch_workdir_", dir="/data/tmp")
    shutil.copytree(src, tmpdir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc", ".venv"))
    return tmpdir



# ---------------------------------------------------------------------------
# VERL BashTool — for RL training via ToolAgentLoop
# ---------------------------------------------------------------------------

# Add SDPO to path for verl imports (lazy — only needed for BashTool)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SDPO"))

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
    _HAS_VERL = True
except ImportError:
    BaseTool = object
    OpenAIFunctionToolSchema = None
    ToolResponse = None
    _HAS_VERL = False


class BashTool(BaseTool):
    """VERL-compatible bash tool that executes commands in an isolated workdir.

    Lifecycle per trajectory:
      1. create() — copies autoresearch/ to a temp dir
      2. execute() — runs bash commands in that dir (called multiple times)
      3. release() — cleans up the temp dir

    After the model submits (echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT),
    the agent loop reads modified train.py from the workdir and dispatches
    it to the GPU fleet for experiment execution.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._workdirs: dict[str, str] = {}  # instance_id -> workdir path
        self._submitted: dict[str, bool] = {}  # instance_id -> submitted flag
        self.autoresearch_dir = config.get("autoresearch_dir", "autoresearch")
        self.command_timeout = config.get("command_timeout", 30)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create an isolated workdir for this trajectory."""
        instance_id, _ = await super().create(instance_id, **kwargs)
        workdir = create_isolated_workdir(self.autoresearch_dir)
        self._workdirs[instance_id] = workdir
        self._submitted[instance_id] = False
        logger.info(f"BashTool.create({instance_id}): workdir={workdir}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute a bash command in the isolated workdir."""
        workdir = self._workdirs.get(instance_id)
        if workdir is None:
            return ToolResponse(text="Error: no workdir for this instance"), 0.0, {}

        command = parameters.get("command", "")
        if not command:
            return ToolResponse(text="Error: empty command"), 0.0, {}

        # Check for submission signal
        if SUBMIT_SIGNAL in command:
            self._submitted[instance_id] = True
            return ToolResponse(text="Submission received. Your changes will be evaluated."), 0.0, {}

        # Execute command
        env = os.environ.copy()
        env.update({"PAGER": "cat", "TQDM_DISABLE": "1"})

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=workdir,
                env=env,
                timeout=self.command_timeout,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if not output:
                output = f"(exit code {result.returncode})"

            # Check if output contains submission signal
            if SUBMIT_SIGNAL in output.split("\n")[0] and result.returncode == 0:
                self._submitted[instance_id] = True
                return ToolResponse(text="Submission received. Your changes will be evaluated."), 0.0, {}

            return ToolResponse(text=output), 0.0, {}

        except subprocess.TimeoutExpired:
            return ToolResponse(text=f"Command timed out after {self.command_timeout}s"), 0.0, {}
        except Exception as e:
            return ToolResponse(text=f"Error executing command: {e}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up the isolated workdir."""
        workdir = self._workdirs.pop(instance_id, None)
        self._submitted.pop(instance_id, None)
        if workdir and os.path.exists(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
            logger.info(f"BashTool.release({instance_id}): cleaned up {workdir}")

    def get_workdir(self, instance_id: str) -> str | None:
        """Get the workdir path for a given instance."""
        return self._workdirs.get(instance_id)

    def is_submitted(self, instance_id: str) -> bool:
        """Check if the agent has submitted for this instance."""
        return self._submitted.get(instance_id, False)

    def read_train_py(self, instance_id: str) -> str | None:
        """Read the modified train.py from the workdir after submission."""
        workdir = self._workdirs.get(instance_id)
        if workdir is None:
            return None
        train_py_path = os.path.join(workdir, "train.py")
        if os.path.exists(train_py_path):
            return Path(train_py_path).read_text()
        return None
