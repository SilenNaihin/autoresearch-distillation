"""Generate prompt parquet for sparse parity SDPO training.

The prompt is: system prompt + instance prompt with solve.py content.
Since the agent loop replaces file content via PUCT selection at runtime,
we only need a few template rows — the actual solve.py content varies per rollout.

Output: data/sparse_parity/train.parquet, data/sparse_parity/test.parquet
"""

import json
import os
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_config import TaskConfig

OUTPUT_DIR = PROJECT_ROOT / "data" / "sparse_parity"
TASK_CONFIG = PROJECT_ROOT / "tasks" / "sparse_parity.yaml"


def main():
    task = TaskConfig.from_yaml(TASK_CONFIG)

    # Read the default solve.py
    solve_py = (PROJECT_ROOT / task.workspace.source_dir / task.workspace.target_file).read_text()

    # Build prompt
    system_prompt = task.prompt.system
    instance_prompt = task.build_instance_prompt(solve_py)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_prompt},
    ]

    # Create enough rows for training (need >= train_batch_size)
    # Duplicate the same prompt — agent loop replaces content at runtime anyway
    MIN_ROWS = 20
    rows = []
    for i in range(MIN_ROWS):
        rows.append({
            "data_source": "sparse_parity",
            "agent_name": "sparse_parity_agent",
            "prompt": prompt,
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": str(i), "split": ""},
        })

    # 80/20 split
    random.seed(42)
    random.shuffle(rows)
    split_idx = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    for i, row in enumerate(train_rows):
        row["extra_info"]["split"] = "train"
        row["extra_info"]["index"] = str(i)
    for i, row in enumerate(test_rows):
        row["extra_info"]["split"] = "test"
        row["extra_info"]["index"] = str(i)

    def rows_to_table(rows):
        return pa.table({
            "data_source": [r["data_source"] for r in rows],
            "agent_name": [r["agent_name"] for r in rows],
            "prompt": [r["prompt"] for r in rows],
            "reward_model": [r["reward_model"] for r in rows],
            "extra_info": [r["extra_info"] for r in rows],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_table = rows_to_table(train_rows)
    train_path = OUTPUT_DIR / "train.parquet"
    pq.write_table(train_table, train_path)
    print(f"Wrote {len(train_rows)} train rows to {train_path}")

    test_table = rows_to_table(test_rows)
    test_path = OUTPUT_DIR / "test.parquet"
    pq.write_table(test_table, test_path)
    print(f"Wrote {len(test_rows)} test rows to {test_path}")


if __name__ == "__main__":
    main()
