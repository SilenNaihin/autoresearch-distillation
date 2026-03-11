"""
GPU fleet runners for dispatching autoresearch experiments.

Maps your fleet of H100s into a pool of GPU slots. Each experiment gets
dispatched to a free slot via SSH, run for ~5 min, and results returned.

Usage:
    pool = GPUPoolRunner()
    output = pool.run(modified_train_py)  # blocks until a GPU is free

Thread-safe — multiple VERL workers call pool.run() concurrently.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from environment import RunOutput


# ---------------------------------------------------------------------------
# GPU slot definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUSlot:
    host: str        # SSH host alias from ~/.ssh/config
    gpu_id: str      # CUDA device index ("0" or "1")
    name: str        # Human-readable label
    remote_dir: str = "~/autoresearch"


# Default fleet — 5 experiment GPUs
# box3 reserved for vLLM inference / SDPO training, not available for experiments.
# Each GPU gets its own remote_dir to avoid file races on 2-GPU boxes.
FLEET = [
    # h100-dev-box: 1x H100 NVL — down as of 2026-03-09
    # GPUSlot("h100_azure",      "0", "box1-gpu0", "~/autoresearch"),
    # h100-dev-box-2: 1x H100 NVL
    GPUSlot("h100-dev-box-2",  "0", "box2-gpu0", "~/autoresearch"),
    # h100-dev-box-4: 2x H100 — separate dirs per GPU
    GPUSlot("h100-dev-box-4",  "0", "box4-gpu0", "~/autoresearch-gpu0"),
    GPUSlot("h100-dev-box-4",  "1", "box4-gpu1", "~/autoresearch-gpu1"),
    # h100-dev-box-5: 2x H100 — separate dirs per GPU
    GPUSlot("h100-dev-box-5",  "0", "box5-gpu0", "~/autoresearch-gpu0"),
    GPUSlot("h100-dev-box-5",  "1", "box5-gpu1", "~/autoresearch-gpu1"),
    # a100-backup-1: 1x A100 80GB PCIe — down as of 2026-03-10 (DNS unreachable)
    # GPUSlot("a100-backup-1",   "0", "a100-gpu0", "~/autoresearch"),
]


# ---------------------------------------------------------------------------
# SSH runner — single slot
# ---------------------------------------------------------------------------

class SSHRunner:
    """Runs an experiment on a specific remote GPU via SSH."""

    def __init__(self, slot: GPUSlot, timeout: int = 600):
        self.slot = slot
        self.timeout = timeout

    def run(self, train_py: str) -> RunOutput:
        slot = self.slot

        # 1. Write modified train.py to remote via SSH stdin
        try:
            write_cmd = f"cat > {slot.remote_dir}/train.py"
            sync = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", slot.host, write_cmd],
                input=train_py, capture_output=True, text=True, timeout=30,
            )
            if sync.returncode != 0:
                return RunOutput("", f"[{slot.name}] File sync failed: {sync.stderr.strip()}", sync.returncode)
        except subprocess.TimeoutExpired:
            return RunOutput("", f"[{slot.name}] SSH connect timeout during file sync", -1)
        except OSError as e:
            return RunOutput("", f"[{slot.name}] SSH error during file sync: {e}", -1)

        # 2. Run experiment on remote GPU
        #    - Kill any stale train.py on this GPU first
        #    - PATH includes ~/.local/bin for uv
        #    - Redirect stderr to stdout so we capture everything
        #    - Exit code signals: 137=OOM killed, 139=segfault, -9=SIGKILL
        cmd = (
            f"nvidia-smi --query-compute-apps=pid --id={slot.gpu_id} --format=csv,noheader "
            f"| xargs -r kill -9 2>/dev/null; sleep 2; "
            f"export TORCHINDUCTOR_DIR=/tmp/torchinductor_${{USER}}_gpu{slot.gpu_id}; "
            f"rm -rf $TORCHINDUCTOR_DIR 2>/dev/null; "
            f"export PATH=$HOME/.local/bin:$PATH && "
            f"cd {slot.remote_dir} && "
            f"CUDA_VISIBLE_DEVICES={slot.gpu_id} uv run train.py 2>&1"
        )

        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
                 "-o", "ServerAliveCountMax=3", "-o", "BatchMode=yes", slot.host, cmd],
                capture_output=True, text=True, timeout=self.timeout,
            )
            # Detect OOM / signal kills
            if r.returncode == 137:
                return RunOutput(r.stdout, f"[{slot.name}] OOM killed (SIGKILL/137)\n{r.stderr}", 137)
            if r.returncode == 139:
                return RunOutput(r.stdout, f"[{slot.name}] Segfault (139)\n{r.stderr}", 139)
            return RunOutput(r.stdout, r.stderr, r.returncode)

        except subprocess.TimeoutExpired:
            # Kill remote process tree
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
# Pool runner — multi-GPU with auto-allocation
# ---------------------------------------------------------------------------

class GPUPoolRunner:
    """Pool of GPU slots with automatic allocation.

    Uses file-based locking (fcntl.flock) for cross-process safety.
    Multiple Ray workers can safely share GPU slots without collisions.

    Dead-box detection: if a slot fails with SSH errors (exit 255),
    it's marked dead for DEAD_COOLDOWN seconds. The next run() call
    skips dead slots and retries on a different one.
    """

    LOCK_DIR = "/data/tmp/gpu_locks"
    DEAD_COOLDOWN = 300  # 5 minutes before retrying a dead box

    # Shared across all instances in the same process
    _dead_until: dict[str, float] = {}
    _dead_lock = threading.Lock()

    def __init__(self, slots: list[GPUSlot] | None = None, timeout: int = 600):
        self.slots = slots or list(FLEET)
        self.timeout = timeout
        os.makedirs(self.LOCK_DIR, exist_ok=True)

    def _is_dead(self, slot: GPUSlot) -> bool:
        import time
        with self._dead_lock:
            deadline = self._dead_until.get(slot.name, 0)
            if time.time() < deadline:
                return True
            # Expired — remove
            self._dead_until.pop(slot.name, None)
            return False

    def _mark_dead(self, slot: GPUSlot):
        import time
        with self._dead_lock:
            self._dead_until[slot.name] = time.time() + self.DEAD_COOLDOWN
            print(f"[pool] {slot.name} marked dead for {self.DEAD_COOLDOWN}s")

    def run(self, train_py: str) -> RunOutput:
        slot, lock_fd = self._acquire_slot()
        try:
            print(f"[pool] {slot.name} <- dispatching")
            runner = SSHRunner(slot, timeout=self.timeout)
            result = runner.run(train_py)

            # SSH-level failure → mark dead, retry on another slot
            if result.returncode == 255 or "Connection timed out" in result.stderr:
                self._mark_dead(slot)
                self._release_slot(lock_fd)
                # Retry once on a different slot
                slot2, lock_fd2 = self._acquire_slot()
                try:
                    print(f"[pool] {slot2.name} <- retrying (after {slot.name} SSH fail)")
                    runner2 = SSHRunner(slot2, timeout=self.timeout)
                    result = runner2.run(train_py)
                    if result.returncode == 255 or "Connection timed out" in result.stderr:
                        self._mark_dead(slot2)
                    status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
                    print(f"[pool] {slot2.name} -> done ({status})")
                    return result
                finally:
                    self._release_slot(lock_fd2)
                return result

            status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
            print(f"[pool] {slot.name} -> done ({status})")
            return result
        finally:
            self._release_slot(lock_fd)

    def _acquire_slot(self) -> tuple[GPUSlot, int]:
        """Acquire an exclusive lock on an available GPU slot, skipping dead boxes."""
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
        """Release the GPU slot lock."""
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

def setup_slot(slot: GPUSlot, local_repo: str = "autoresearch") -> bool:
    """Set up a remote machine for running experiments.

    Syncs the autoresearch repo and installs deps at slot.remote_dir.
    """
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
            f"{local_repo}/", f"{host}:{remote}/",
        ]),
        ("Install dependencies", ["ssh", host, f"{uv}cd {remote} && uv sync"]),
        ("Prepare data", ["ssh", host, f"{uv}cd {remote} && uv run prepare.py"]),
    ]

    for desc, cmd in steps:
        print(f"  [{desc}]...", end=" ", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"FAILED\n    {r.stderr.strip()[:200]}")
            return False
        print("ok")

    print(f"[setup] {slot.name}: ready")
    return True


def setup_fleet(slots: list[GPUSlot] | None = None, local_repo: str = "autoresearch"):
    """Set up all slots in the fleet. Runs once per unique (host, remote_dir) pair."""
    slots = slots or FLEET
    seen: set[tuple[str, str]] = set()
    results = {}
    for slot in slots:
        key = (slot.host, slot.remote_dir)
        if key in seen:
            continue
        seen.add(key)
        results[slot.name] = setup_slot(slot, local_repo)
    return results
