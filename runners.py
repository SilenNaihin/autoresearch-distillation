"""
GPU fleet runners for dispatching autoresearch experiments.

Maps your fleet of H100s into a pool of GPU slots. Each experiment gets
dispatched to a free slot via SSH, run for ~5 min, and results returned.

Usage:
    pool = GPUPoolRunner()              # uses all 8 GPUs
    env = ExperimentEnvironment(pool)   # plug into the environment
    result = env.step(model_response)   # auto-dispatches to a free GPU

For parallel experiments (from VERL), multiple threads can call env.step()
concurrently — the pool handles queuing.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from queue import Queue

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


# Default fleet — 6 experiment GPUs
# box3 reserved for vLLM inference, not available for experiments.
# Each GPU gets its own remote_dir to avoid file races on 2-GPU boxes.
FLEET = [
    # h100-dev-box: 1x H100 NVL
    GPUSlot("h100_azure",      "0", "box1-gpu0", "~/autoresearch"),
    # h100-dev-box-2: 1x H100 NVL
    GPUSlot("h100-dev-box-2",  "0", "box2-gpu0", "~/autoresearch"),
    # h100-dev-box-4: 2x H100 — separate dirs per GPU
    GPUSlot("h100-dev-box-4",  "0", "box4-gpu0", "~/autoresearch-gpu0"),
    GPUSlot("h100-dev-box-4",  "1", "box4-gpu1", "~/autoresearch-gpu1"),
    # h100-dev-box-5: 2x H100 — separate dirs per GPU
    GPUSlot("h100-dev-box-5",  "0", "box5-gpu0", "~/autoresearch-gpu0"),
    GPUSlot("h100-dev-box-5",  "1", "box5-gpu1", "~/autoresearch-gpu1"),
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
        #    - PATH includes ~/.local/bin for uv
        #    - Redirect stderr to stdout so we capture everything
        #    - Exit code signals: 137=OOM killed, 139=segfault, -9=SIGKILL
        cmd = (
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
                     f"pkill -9 -f 'CUDA_VISIBLE_DEVICES={slot.gpu_id}.*train.py'"],
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

    Thread-safe. If all GPUs are busy, callers block until one frees up.
    """

    def __init__(self, slots: list[GPUSlot] | None = None, timeout: int = 600):
        self.slots = slots or list(FLEET)
        self.timeout = timeout
        self._pool: Queue[GPUSlot] = Queue()
        for slot in self.slots:
            self._pool.put(slot)
        self._active: dict[str, GPUSlot] = {}
        self._lock = threading.Lock()

    def run(self, train_py: str) -> RunOutput:
        slot = self._pool.get()  # blocks if all busy
        tid = threading.current_thread().name
        with self._lock:
            self._active[tid] = slot
        try:
            print(f"[pool] {slot.name} <- dispatching ({self.available}/{self.total} free)")
            runner = SSHRunner(slot, timeout=self.timeout)
            result = runner.run(train_py)
            status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
            print(f"[pool] {slot.name} -> done ({status})")
            return result
        finally:
            with self._lock:
                self._active.pop(tid, None)
            self._pool.put(slot)

    @property
    def available(self) -> int:
        return self._pool.qsize()

    @property
    def total(self) -> int:
        return len(self.slots)

    def status(self) -> dict:
        with self._lock:
            return {
                "total": self.total,
                "available": self.available,
                "active": {k: v.name for k, v in self._active.items()},
            }


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
