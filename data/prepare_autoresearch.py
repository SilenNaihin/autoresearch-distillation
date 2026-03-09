"""
Convert rollouts/rollouts.jsonl -> parquet for SDPO training.

Output schema per row:
  - data_source: "autoresearch"
  - agent_name: "autoresearch_agent"  (matches @register name)
  - prompt: [system_msg, user_msg]    (first 2 turns only)
  - reward_model: {"style": "rule", "ground_truth": ""}
  - extra_info: {"index": str(i), "split": "train"|"test"}

Deduplicates prompts (many rollouts share same prompt). 80/20 train/test split.
Outputs: data/autoresearch/train.parquet, data/autoresearch/test.parquet
"""

import json
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from prompts import SYSTEM_PROMPT, build_instance_prompt

ROLLOUT_FILE = os.path.join(PROJECT_ROOT, "rollouts", "rollouts.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "autoresearch")


def load_rollouts(path: str) -> list[dict]:
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def extract_train_py(rollout: dict) -> str:
    """Extract train.py content from the user message in a rollout."""
    user_msg = rollout["conversations"][1]["content"]
    start = user_msg.find("```python\n") + len("```python\n")
    end = user_msg.find("\n```", start)
    return user_msg[start:end]


def extract_prompt(rollout: dict) -> list[dict]:
    """Build prompt from prompts.py using train.py content from rollout."""
    train_py = extract_train_py(rollout)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_instance_prompt(train_py, [])},
    ]


def prompt_key(prompt: list[dict]) -> str:
    """Create a dedup key from prompt content."""
    return json.dumps([m["content"][:200] for m in prompt], sort_keys=True)


def build_rows(rollouts: list[dict]) -> list[dict]:
    """Build deduplicated rows from rollouts."""
    seen = set()
    rows = []

    for rollout in rollouts:
        prompt = extract_prompt(rollout)
        key = prompt_key(prompt)

        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "data_source": "autoresearch",
            "agent_name": "autoresearch_agent",
            "prompt": prompt,
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": str(len(rows)), "split": ""},
        })

    # Need enough rows so train split (80%) >= train_batch_size (16) after split
    # ceil(16 / 0.8) = 20
    MIN_ROWS = 20
    if 0 < len(rows) < MIN_ROWS:
        import copy
        base = rows[:]
        while len(rows) < MIN_ROWS:
            rows.extend(copy.deepcopy(base))
        rows = rows[:MIN_ROWS]

    return rows


def split_and_save(rows: list[dict], output_dir: str, train_ratio: float = 0.8):
    """Split into train/test and save as parquet."""
    os.makedirs(output_dir, exist_ok=True)

    random.seed(42)
    random.shuffle(rows)

    split_idx = max(1, int(len(rows) * train_ratio))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    for i, row in enumerate(train_rows):
        row["extra_info"]["split"] = "train"
        row["extra_info"]["index"] = str(i)

    for i, row in enumerate(test_rows):
        row["extra_info"]["split"] = "test"
        row["extra_info"]["index"] = str(i)

    def rows_to_table(rows: list[dict]) -> pa.Table:
        return pa.table({
            "data_source": [r["data_source"] for r in rows],
            "agent_name": [r["agent_name"] for r in rows],
            "prompt": [r["prompt"] for r in rows],
            "reward_model": [r["reward_model"] for r in rows],
            "extra_info": [r["extra_info"] for r in rows],
        })

    if train_rows:
        train_table = rows_to_table(train_rows)
        train_path = os.path.join(output_dir, "train.parquet")
        pq.write_table(train_table, train_path)
        print(f"Wrote {len(train_rows)} train rows to {train_path}")
    else:
        print("WARNING: No train rows to write")

    if test_rows:
        test_table = rows_to_table(test_rows)
        test_path = os.path.join(output_dir, "test.parquet")
        pq.write_table(test_table, test_path)
        print(f"Wrote {len(test_rows)} test rows to {test_path}")
    else:
        print("WARNING: No test rows to write")


def main():
    if not os.path.exists(ROLLOUT_FILE):
        print(f"ERROR: Rollout file not found: {ROLLOUT_FILE}")
        print("Run the data collection loop first (loop.py) to generate rollouts.")
        sys.exit(1)

    rollouts = load_rollouts(ROLLOUT_FILE)
    print(f"Loaded {len(rollouts)} rollouts from {ROLLOUT_FILE}")

    rows = build_rows(rollouts)
    print(f"Built {len(rows)} unique prompts (deduplicated)")

    if not rows:
        print("ERROR: No valid rows to process")
        sys.exit(1)

    split_and_save(rows, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
