"""
Persistent experiment result cache.

Stores results keyed by SHA256 of the diff text. Both the baseline loop
and SDPO agent_loop write to their own cache files but read from both,
so results are shared across methods.

Cache files live in outputs/cache/{baseline,sdpo}.json.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

CACHE_DIR = Path("outputs/cache")
BASELINE_CACHE = CACHE_DIR / "baseline.json"
SDPO_CACHE = CACHE_DIR / "sdpo.json"


class ExperimentCache:
    """Thread-safe persistent experiment cache.

    Args:
        write_path: The JSON file this instance writes to.
        read_paths: All JSON files to read from (including write_path).
    """

    def __init__(self, write_path: Path, read_paths: list[Path] | None = None):
        self._write_path = write_path
        self._read_paths = read_paths or [BASELINE_CACHE, SDPO_CACHE]
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load all cache files into memory."""
        for path in self._read_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    self._cache.update(data)
                except (json.JSONDecodeError, OSError):
                    pass

    def _save(self):
        """Save only our entries to our write file."""
        self._write_path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing entries from our file first
        existing = {}
        if self._write_path.exists():
            try:
                existing = json.loads(self._write_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        # Merge our in-memory cache (we may have entries from other files too,
        # but only save the ones that are new to our file)
        existing.update(self._cache)
        self._write_path.write_text(json.dumps(existing, ensure_ascii=True, indent=2))

    @staticmethod
    def diff_hash(diff_text: str) -> str:
        return hashlib.sha256(diff_text.encode()).hexdigest()

    def get(self, diff_text: str) -> dict | None:
        """Look up cached result for a diff. Returns None on miss."""
        h = self.diff_hash(diff_text)
        with self._lock:
            return self._cache.get(h)

    def put(self, diff_text: str, result: dict):
        """Store a result and persist to disk."""
        h = self.diff_hash(diff_text)
        with self._lock:
            self._cache[h] = result
            self._save()

    def __len__(self):
        return len(self._cache)
