"""
PUCT state reuse buffer for SDPO/GRPO training.

Maintains a search tree of file versions (e.g. train.py, kernel.py, system_prompt.txt).
Each state is a (content, metric_value, reward) tuple with parent pointers. States are
selected for exploration using PUCT scores, biasing toward high-reward states that
haven't been explored much.

Disk-backed JSON with fcntl locking (same pattern as experiment_cache.py).
"""

import fcntl
import json
import logging
import math
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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


class ReuseBuffer:
    """PUCT-based state reuse buffer for file versions.

    Args:
        path: JSON file for persistent storage.
        c_puct: Exploration constant for PUCT formula.
        max_states: Maximum number of states before pruning.
        direction: "minimize" or "maximize" — controls best metric comparison.
    """

    def __init__(self, path: Path, c_puct: float = 1.0, max_states: int = 1000,
                 direction: str = "minimize"):
        self._path = Path(path)
        self._c_puct = c_puct
        self._max_states = max_states
        self._direction = direction
        self._lock = threading.Lock()
        self._data: dict = {"next_id": 0, "total_visits": 0, "states": {}}
        self._load()

    def _load(self):
        """Load buffer from disk."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r") as f:
                if not _flock_with_timeout(f, fcntl.LOCK_SH):
                    logger.warning(f"Timed out acquiring shared lock on {self._path}")
                    return
                content = f.read()
                if content.strip():
                    self._data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            logger.exception(f"Failed to load reuse buffer from {self._path}")

    def _save(self):
        """Save buffer to disk with exclusive lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if self._path.exists() else "w+"
        try:
            with open(self._path, mode) as f:
                if not _flock_with_timeout(f, fcntl.LOCK_EX):
                    logger.warning(f"Timed out acquiring exclusive lock on {self._path}, skipping write")
                    return
                # Re-read under lock to merge concurrent writes
                f.seek(0)
                content = f.read()
                if content.strip():
                    try:
                        on_disk = json.loads(content)
                        # Merge: take max next_id, union of states (disk wins on conflict)
                        self._data["next_id"] = max(self._data["next_id"], on_disk.get("next_id", 0))
                        for sid, state in on_disk.get("states", {}).items():
                            if sid not in self._data["states"]:
                                self._data["states"][sid] = state
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupt buffer file {self._path}, overwriting")
                f.seek(0)
                f.truncate()
                f.write(json.dumps(self._data, ensure_ascii=True))
        except Exception:
            logger.exception(f"Failed to write reuse buffer to {self._path}")

    def seed(self, content: str, metric_value: float) -> None:
        """Add seed state (id=0) if buffer is empty."""
        with self._lock:
            if self._data["states"]:
                return
            self._data["states"]["0"] = {
                "id": 0,
                "content": content,
                "metric_value": metric_value,
                "reward": 0.0,
                "parent_id": None,
                "n_visits": 0,
                "best_child_reward": 0.0,
            }
            self._data["next_id"] = 1
            self._save()

    def _ancestors(self, sid: str) -> list[str]:
        """Walk parent chain from sid (exclusive) to root. Returns list of ancestor sids."""
        states = self._data["states"]
        ancestors = []
        current = states.get(sid)
        while current is not None and current["parent_id"] is not None:
            pid = str(current["parent_id"])
            if pid not in states:
                break
            ancestors.append(pid)
            current = states[pid]
        return ancestors

    def add(self, content: str, metric_value: float, reward: float, parent_id: int) -> int:
        """Add a new state and propagate reward up ancestor chain. Returns new state id."""
        with self._lock:
            sid = self._data["next_id"]
            self._data["next_id"] = sid + 1
            self._data["states"][str(sid)] = {
                "id": sid,
                "content": content,
                "metric_value": metric_value,
                "reward": reward,
                "parent_id": parent_id,
                "n_visits": 0,
                "best_child_reward": 0.0,
            }
            # Propagate reward up to all ancestors (Q = max descendant reward)
            for anc_sid in self._ancestors(str(sid)):
                anc = self._data["states"][anc_sid]
                anc["best_child_reward"] = max(anc["best_child_reward"], reward)
            self._prune()
            self._save()
            return sid

    def _puct_scores(self) -> dict[str, float]:
        """Compute PUCT scores for all states."""
        states = self._data["states"]
        if not states:
            return {}
        total_visits = self._data["total_visits"]

        # Rank states by reward (descending) for harmonic prior
        sorted_sids = sorted(states.keys(), key=lambda s: states[s]["reward"], reverse=True)
        rank_map = {sid: rank + 1 for rank, sid in enumerate(sorted_sids)}

        # Harmonic weights: sum of 1/rank for normalization
        harmonic_sum = sum(1.0 / r for r in rank_map.values())

        scores = {}
        for sid, state in states.items():
            q = state["best_child_reward"]
            rank = rank_map[sid]
            p = (1.0 / rank) / harmonic_sum  # normalized harmonic prior
            n = state["n_visits"]
            score = q + self._c_puct * p * math.sqrt(1 + total_visits) / (1 + n)
            scores[sid] = score
        return scores

    def select(self, n: int) -> list[tuple[int, str]]:
        """Pick n states by PUCT score, increment visits. Returns [(id, content), ...]."""
        with self._lock:
            states = self._data["states"]
            if not states:
                return []

            scores = self._puct_scores()
            top_sids = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)[:n]

            results = []
            for sid in top_sids:
                state = states[sid]
                state["n_visits"] += 1
                self._data["total_visits"] += 1
                for anc_sid in self._ancestors(sid):
                    states[anc_sid]["n_visits"] += 1
                results.append((state["id"], state["content"]))

            self._save()
            return results

    def find_by_content(self, content: str) -> dict | None:
        """Look up a state by its exact content. Returns state dict or None."""
        with self._lock:
            for state in self._data["states"].values():
                if state.get("content") == content:
                    return dict(state)  # copy
                # Backward compat: old buffers use "code" key
                if state.get("code") == content:
                    return dict(state)
            return None

    def get_best_metric(self) -> float:
        """Return best metric_value across all states (direction-aware)."""
        with self._lock:
            if not self._data["states"]:
                if self._direction == "minimize":
                    return float("inf")
                return float("-inf")
            values = []
            for s in self._data["states"].values():
                v = s.get("metric_value", s.get("val_bpb"))  # backward compat
                if v is not None:
                    values.append(v)
            if not values:
                if self._direction == "minimize":
                    return float("inf")
                return float("-inf")
            if self._direction == "minimize":
                return min(values)
            return max(values)

    def _prune(self):
        """Remove lowest-reward leaf states when over max_states. Never removes seed."""
        states = self._data["states"]
        if len(states) <= self._max_states:
            return

        parent_ids = {str(s["parent_id"]) for s in states.values() if s["parent_id"] is not None}
        candidates = [
            sid for sid in states
            if sid != "0" and sid not in parent_ids
        ]
        candidates.sort(key=lambda s: states[s]["reward"])

        to_remove = len(states) - self._max_states
        for sid in candidates[:to_remove]:
            del states[sid]

    def __len__(self) -> int:
        return len(self._data["states"])
