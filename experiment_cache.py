"""
Persistent experiment result cache.

Stores results keyed by SHA256 of the diff text. Both the baseline loop
and SDPO agent_loop write to their own cache files but read from both,
so results are shared across methods.

Uses fcntl.flock for cross-process safety (multiple VERL workers may
read/write concurrently). Tracks the best val_bpb result seen so far.

Each entry stores a written_at timestamp. get() skips entries written
after the cache was instantiated, so sibling rollouts in the same step
don't short-circuit each other.

Cache files live in outputs/cache/{baseline,sdpo}.json.
"""

import fcntl
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

CACHE_DIR = Path("outputs/cache")
BASELINE_CACHE = CACHE_DIR / "baseline.json"
SDPO_CACHE = CACHE_DIR / "sdpo.json"

FLOCK_TIMEOUT = 60  # seconds


def _flock_with_timeout(f, lock_type, timeout=FLOCK_TIMEOUT):
    """Acquire flock with timeout. Returns True if acquired, False if timed out."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(f, lock_type | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.1)


class ExperimentCache:
    """Thread-safe, process-safe persistent experiment cache.

    Args:
        write_path: The JSON file this instance writes to.
        read_paths: All JSON files to read from (including write_path).
    """

    def __init__(self, write_path: Path, read_paths: list[Path] | None = None):
        self._write_path = write_path
        self._read_paths = read_paths or [BASELINE_CACHE, SDPO_CACHE]
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._best_val_bpb: float = 1e9
        self._best_diff: str = ""
        # Timestamp before loading — entries written after this are from
        # the current step and should not count as cache hits.
        self._created_at = time.time()
        self._load()

    def _load(self):
        """Load all cache files into memory."""
        for path in self._read_paths:
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    if not _flock_with_timeout(f, fcntl.LOCK_SH):
                        logger.warning(f"Timed out acquiring shared lock on {path}")
                        continue
                    data = json.loads(f.read())
                    # Handle both flat format and structured format
                    if "diffs" in data:
                        self._cache.update(data["diffs"])
                        if data.get("best_val_bpb", 1e9) < self._best_val_bpb:
                            self._best_val_bpb = data["best_val_bpb"]
                            self._best_diff = data.get("best_diff", "")
                    else:
                        self._cache.update(data)
            except (json.JSONDecodeError, OSError):
                logger.exception(f"Failed to load cache file {path}")

    def _save(self, val_bpb: float | None = None, diff_text: str | None = None):
        """Save cache to our write file with flock."""
        self._write_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if self._write_path.exists() else "w+"
        try:
            with open(self._write_path, mode) as f:
                if not _flock_with_timeout(f, fcntl.LOCK_EX):
                    logger.warning(f"Timed out acquiring exclusive lock on {self._write_path}, skipping cache write")
                    return
                # Re-read under lock to avoid clobbering concurrent writes
                f.seek(0)
                content = f.read()
                if content.strip():
                    try:
                        on_disk = json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupt cache file {self._write_path}, overwriting")
                        on_disk = {}
                else:
                    on_disk = {}
                # Handle structured format
                if "diffs" in on_disk:
                    on_disk["diffs"].update(self._cache)
                else:
                    on_disk = {"diffs": {**on_disk, **self._cache},
                               "best_val_bpb": self._best_val_bpb,
                               "best_diff": self._best_diff}
                # Update best if this is a new low
                if val_bpb is not None and diff_text is not None and val_bpb < on_disk.get("best_val_bpb", 1e9):
                    on_disk["best_val_bpb"] = val_bpb
                    on_disk["best_diff"] = f"{diff_text}\nval_bpb={val_bpb:.6f}"
                    self._best_val_bpb = val_bpb
                    self._best_diff = on_disk["best_diff"]
                f.seek(0)
                f.truncate()
                f.write(json.dumps(on_disk, ensure_ascii=True))
        except Exception:
            logger.exception(f"Failed to write cache file {self._write_path}")

    @staticmethod
    def diff_hash(diff_text: str) -> str:
        return hashlib.sha256(diff_text.encode()).hexdigest()

    def get(self, diff_text: str) -> dict | None:
        """Look up cached result for a diff. Returns None on miss.

        Skips entries written after this instance was created, so sibling
        rollouts in the same step don't short-circuit each other.
        """
        h = self.diff_hash(diff_text)
        with self._lock:
            entry = self._cache.get(h)
            if entry is None:
                return None
            if entry.get("written_at", 0) > self._created_at:
                return None
            return entry

    def put(self, diff_text: str, result: dict,
            val_bpb: float | None = None, diff_text_raw: str | None = None):
        """Store a result and persist to disk. Optionally update best."""
        h = self.diff_hash(diff_text)
        result["written_at"] = time.time()
        with self._lock:
            self._cache[h] = result
            self._save(val_bpb=val_bpb, diff_text=diff_text_raw)

    def get_best_diff(self) -> str:
        """Return the best diff seen so far, or empty string."""
        return self._best_diff

    def __len__(self):
        return len(self._cache)
