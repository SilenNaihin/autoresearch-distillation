"""
GPU fleet runners for dispatching experiments.

Generic — works with any task config. The run command, GPU flags,
and target file are all configurable.

Usage:
    from task_config import TaskConfig
    task = TaskConfig.from_yaml("tasks/autoresearch.yaml")
    pool = GPUPoolRunner(task=task)
    output = pool.run(modified_content)  # blocks until a slot is free

Thread-safe — multiple VERL workers call pool.run() concurrently.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from lib.environment import RunOutput


# ---------------------------------------------------------------------------
# GPU slot definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUSlot:
    host: str        # SSH host alias from ~/.ssh/config
    gpu_id: str      # CUDA device index ("0" or "1")
    name: str        # Human-readable label
    remote_dir: str = "~"


# ---------------------------------------------------------------------------
# SSH runner — single slot
# ---------------------------------------------------------------------------

class SSHRunner:
    """Runs an experiment on a specific remote slot via SSH."""

    def __init__(self, slot: GPUSlot, timeout: int = 600,
                 target_file: str = "train.py",
                 run_command: str = "uv run train.py 2>&1",
                 needs_gpu: bool = True,
                 clear_torch_cache: bool = False):
        self.slot = slot
        self.timeout = timeout
        self.target_file = target_file
        self.run_command = run_command
        self.needs_gpu = needs_gpu
        self.clear_torch_cache = clear_torch_cache

    def run(self, content: str) -> RunOutput:
        slot = self.slot

        # 1. Write modified target file to remote via SSH stdin
        try:
            write_cmd = f"cat > {slot.remote_dir}/{self.target_file}"
            sync = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", slot.host, write_cmd],
                input=content, capture_output=True, text=True, timeout=30,
            )
            if sync.returncode != 0:
                return RunOutput("", f"[{slot.name}] File sync failed: {sync.stderr.strip()}", sync.returncode)
        except subprocess.TimeoutExpired:
            return RunOutput("", f"[{slot.name}] SSH connect timeout during file sync", -1)
        except OSError as e:
            return RunOutput("", f"[{slot.name}] SSH error during file sync: {e}", -1)

        # 2. Build remote command
        parts = []

        # GPU cleanup: kill stale processes on this device
        if self.needs_gpu:
            parts.append(
                f"nvidia-smi --query-compute-apps=pid --id={slot.gpu_id} --format=csv,noheader "
                f"| xargs -r kill -9 2>/dev/null; sleep 2"
            )

        # Torch inductor cache cleanup
        if self.clear_torch_cache:
            parts.append(
                f"export TORCHINDUCTOR_DIR=/tmp/torchinductor_${{USER}}_gpu{slot.gpu_id}; "
                f"rm -rf $TORCHINDUCTOR_DIR 2>/dev/null"
            )

        # PATH + cd + run
        run_prefix = f"export PATH=$HOME/.local/bin:$PATH && cd {slot.remote_dir}"
        if self.needs_gpu:
            run_prefix += f" && export CUDA_VISIBLE_DEVICES={slot.gpu_id}"
        run_part = f"{run_prefix} && {self.run_command}"
        parts.append(run_part)

        cmd = "; ".join(parts)

        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
                 "-o", "ServerAliveCountMax=3", "-o", "BatchMode=yes", slot.host, cmd],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if r.returncode == 137:
                return RunOutput(r.stdout, f"[{slot.name}] OOM killed (SIGKILL/137)\n{r.stderr}", 137)
            if r.returncode == 139:
                return RunOutput(r.stdout, f"[{slot.name}] Segfault (139)\n{r.stderr}", 139)
            return RunOutput(r.stdout, r.stderr, r.returncode)

        except subprocess.TimeoutExpired:
            # Kill remote process tree
            if self.needs_gpu:
                try:
                    subprocess.run(
                        ["ssh", "-o", "ConnectTimeout=5", slot.host,
                         f"nvidia-smi --query-compute-apps=pid --id={slot.gpu_id} --format=csv,noheader "
                         f"| xargs -r kill -9 2>/dev/null"],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    pass
            return RunOutput("", f"[{slot.name}] TIMEOUT: exceeded {self.timeout}s", -1)

        except OSError as e:
            return RunOutput("", f"[{slot.name}] SSH error: {e}", -1)


# ---------------------------------------------------------------------------
# Pool runner — multi-slot with auto-allocation
# ---------------------------------------------------------------------------

class GPUPoolRunner:
    """Pool of slots with automatic allocation.

    Uses file-based locking (fcntl.flock) for cross-process safety.
    Multiple Ray workers can safely share slots without collisions.

    Dead-box detection: if a slot fails with SSH errors (exit 255),
    it's marked dead for DEAD_COOLDOWN seconds.
    """

    LOCK_DIR = "/data/tmp/gpu_locks"
    DEAD_COOLDOWN = 300  # 5 minutes before retrying a dead box

    _dead_until: dict[str, float] = {}
    _dead_lock = threading.Lock()

    def __init__(self, slots: list[GPUSlot] | None = None, timeout: int = 600,
                 task=None):
        """Initialize pool.

        Args:
            slots: Explicit list of GPUSlots. If None, loaded from task config.
            timeout: SSH command timeout.
            task: TaskConfig instance (used for run_command, target_file, GPU flags).
        """
        if slots is not None:
            self.slots = list(slots)
        elif task is not None:
            self.slots = task.get_fleet_slots()
        else:
            self.slots = []

        self.timeout = timeout
        self._task = task

        # Extract execution params from task config
        if task is not None:
            self._target_file = task.workspace.target_file
            self._run_command = task.execution.run_command
            self._needs_gpu = task.execution.needs_gpu
            self._clear_torch_cache = task.execution.clear_torch_cache
        else:
            self._target_file = "train.py"
            self._run_command = "uv run train.py 2>&1"
            self._needs_gpu = True
            self._clear_torch_cache = False

        os.makedirs(self.LOCK_DIR, exist_ok=True)

    def _is_dead(self, slot: GPUSlot) -> bool:
        import time
        with self._dead_lock:
            deadline = self._dead_until.get(slot.name, 0)
            if time.time() < deadline:
                return True
            self._dead_until.pop(slot.name, None)
            return False

    def _mark_dead(self, slot: GPUSlot):
        import time
        with self._dead_lock:
            self._dead_until[slot.name] = time.time() + self.DEAD_COOLDOWN
            print(f"[pool] {slot.name} marked dead for {self.DEAD_COOLDOWN}s")

    def _make_runner(self, slot: GPUSlot) -> SSHRunner:
        return SSHRunner(
            slot, timeout=self.timeout,
            target_file=self._target_file,
            run_command=self._run_command,
            needs_gpu=self._needs_gpu,
            clear_torch_cache=self._clear_torch_cache,
        )

    def run(self, content: str) -> RunOutput:
        slot, lock_fd = self._acquire_slot()
        try:
            print(f"[pool] {slot.name} <- dispatching")
            runner = self._make_runner(slot)
            result = runner.run(content)

            # SSH-level failure → mark dead, retry on another slot
            if result.returncode == 255 or "Connection timed out" in result.stderr:
                self._mark_dead(slot)
                self._release_slot(lock_fd)
                slot2, lock_fd2 = self._acquire_slot()
                try:
                    print(f"[pool] {slot2.name} <- retrying (after {slot.name} SSH fail)")
                    runner2 = self._make_runner(slot2)
                    result = runner2.run(content)
                    if result.returncode == 255 or "Connection timed out" in result.stderr:
                        self._mark_dead(slot2)
                    status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
                    print(f"[pool] {slot2.name} -> done ({status})")
                    return result
                finally:
                    self._release_slot(lock_fd2)

            status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
            print(f"[pool] {slot.name} -> done ({status})")
            return result
        finally:
            self._release_slot(lock_fd)

    def _acquire_slot(self) -> tuple[GPUSlot, int]:
        """Acquire an exclusive lock on an available slot, skipping dead boxes."""
        import fcntl
        import time
        while True:
            for slot in self.slots:
                if self._is_dead(slot):
                    continue
                lock_path = os.path.join(self.LOCK_DIR, f"{slot.name}.lock")
                lock_fd = open(lock_path, "w")
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return slot, lock_fd
                except BlockingIOError:
                    lock_fd.close()
            time.sleep(2)

    def _release_slot(self, lock_fd) -> None:
        """Release the slot lock."""
        import fcntl
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        except Exception:
            pass

    @property
    def total(self) -> int:
        return len(self.slots)


# ---------------------------------------------------------------------------
# Fleet setup — bootstrap remote machines
# ---------------------------------------------------------------------------

def setup_slot(slot: GPUSlot, source_dir: str = "autoresearch",
               setup_commands: list[str] | None = None) -> bool:
    """Set up a remote machine for running experiments."""
    host = slot.host
    remote = slot.remote_dir
    uv = "export PATH=$HOME/.local/bin:$PATH && "
    print(f"[setup] {slot.name}: setting up {host}:{remote}")

    steps = [
        ("Ensure uv installed", [
            "ssh", host, "export PATH=$HOME/.local/bin:$PATH && which uv || curl -LsSf https://astral.sh/uv/install.sh | sh",
        ]),
        (f"Create {remote}", ["ssh", host, f"mkdir -p {remote}"]),
        ("Sync repo files", [
            "rsync", "-az", "--exclude=.git", "--exclude=__pycache__",
            f"{source_dir}/", f"{host}:{remote}/",
        ]),
    ]

    # Add task-specific setup commands
    for cmd in (setup_commands or []):
        steps.append((cmd, ["ssh", host, f"{uv}cd {remote} && {cmd}"]))

    for desc, cmd in steps:
        print(f"  [{desc}]...", end=" ", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"FAILED\n    {r.stderr.strip()[:200]}")
            return False
        print("ok")

    print(f"[setup] {slot.name}: ready")
    return True


def setup_fleet(slots: list[GPUSlot], source_dir: str = "autoresearch",
                setup_commands: list[str] | None = None):
    """Set up all slots in the fleet. Runs once per unique (host, remote_dir) pair."""
    seen: set[tuple[str, str]] = set()
    results = {}
    for slot in slots:
        key = (slot.host, slot.remote_dir)
        if key in seen:
            continue
        seen.add(key)
        results[slot.name] = setup_slot(slot, source_dir, setup_commands)
    return results
