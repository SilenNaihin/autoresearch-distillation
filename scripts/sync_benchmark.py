#!/usr/bin/env python3
"""Download GAIA2 benchmark from HuggingFace and save as viewer-compatible JSON.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/sync_benchmark.py
"""

import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    import pandas as pd
except ImportError:
    print("Install deps: pip install huggingface_hub pandas pyarrow")
    sys.exit(1)

OUTPUT = Path(__file__).resolve().parent.parent / "viewer" / "public" / "data" / "benchmark.json"
REPO = "meta-agents-research-environments/gaia2"
TOKEN = os.environ.get("HF_TOKEN")
CATEGORIES = ["search", "time", "execution", "adaptability", "ambiguity"]


def download_category(category: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=REPO,
        filename=f"{category}/validation-00000-of-00001.parquet",
        repo_type="dataset",
        token=TOKEN,
    )
    return pd.read_parquet(path)


def extract_scenario(row, category: str) -> dict:
    data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
    events = data.get("events", [])
    metadata = data.get("metadata", {})
    definition = metadata.get("definition", {})

    # Extract task prompt from USER event
    question = ""
    expected_answer = ""
    for event in events:
        if event.get("event_type") == "USER":
            args = event.get("action", {}).get("args", [])
            if args:
                question = args[0].get("value", "")
        if event.get("class_name") == "OracleEvent":
            args = event.get("action", {}).get("args", [])
            if args:
                expected_answer = args[0].get("value", "")

    # App names available in the scenario
    apps = [app["name"] for app in data.get("apps", [])]

    # Oracle events summary (tool calls the oracle makes)
    oracle_actions = []
    for event in events:
        if event.get("class_name") == "OracleEvent" and event["action"]["app"] != "AgentUserInterface":
            oracle_actions.append({
                "app": event["action"]["app"],
                "function": event["action"]["function"],
            })

    return {
        "task_id": str(row.get("id", "")),
        "scenario_id": str(row.get("scenario_id", "")),
        "category": category,
        "question": question,
        "expected_answer": expected_answer,
        "split": str(row.get("split", "validation")),
        "tags": definition.get("tags", []),
        "apps": apps,
        "oracle_actions": oracle_actions,
        "start_time": definition.get("start_time"),
        "hints": definition.get("hints", []),
    }


def main():
    if not TOKEN:
        print("Set HF_TOKEN env var first: export HF_TOKEN=hf_xxx")
        sys.exit(1)

    scenarios = []
    for category in CATEGORIES:
        print(f"Downloading {category}...")
        df = download_category(category)
        print(f"  Got {len(df)} rows")

        for _, row in df.iterrows():
            scenarios.append(extract_scenario(row, category))

    stats = {
        "total": len(scenarios),
        "by_category": {cat: sum(1 for s in scenarios if s["category"] == cat) for cat in CATEGORIES},
        "with_answer": sum(1 for s in scenarios if s["expected_answer"]),
        "with_oracle_actions": sum(1 for s in scenarios if s["oracle_actions"]),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"stats": stats, "scenarios": scenarios}, indent=2))
    print(f"\nWrote {len(scenarios)} scenarios to {OUTPUT}")
    print(f"  Categories: {stats['by_category']}")
    print(f"  With answers: {stats['with_answer']}")
    print(f"  With oracle actions: {stats['with_oracle_actions']}")


if __name__ == "__main__":
    main()
