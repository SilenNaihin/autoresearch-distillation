#!/usr/bin/env python3
"""Export ARE trace directories to viewer-compatible JSON data."""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_metrics import aggregate_results, parse_trace


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def export(
    are_traces: Path,
    model_name: str,
    output_dir: Path,
    training_step: int | None = None,
) -> None:
    # Discover trace files
    trace_files = sorted(are_traces.glob("*.jsonl"))
    if not trace_files:
        print(f"No .jsonl trace files found in {are_traces}")
        sys.exit(1)

    print(f"Found {len(trace_files)} trace files in {are_traces}")

    # Parse all traces
    scenarios = []
    for tf in trace_files:
        try:
            result = parse_trace(str(tf))
            scenarios.append(result)
        except Exception as e:
            print(f"  Warning: failed to parse {tf.name}: {e}")

    if not scenarios:
        print("No scenarios parsed successfully.")
        sys.exit(1)

    print(f"Parsed {len(scenarios)} scenarios")

    # Generate run_id
    if training_step is not None:
        run_id = f"{slugify(model_name)}-step{training_step}"
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        run_id = f"{slugify(model_name)}-{date_str}"

    # Compute aggregates
    aggregates = aggregate_results(scenarios)

    # Build scenario summaries (without heavy fields)
    scenario_summaries = []
    for s in scenarios:
        scenario_summaries.append({
            "scenario_id": s["scenario_id"],
            "category": s["category"],
            "passed": s["passed"],
            "tool_score": s["tool_score"],
            "failure_type": s["failure_type"],
            "n_tool_calls": s["n_tool_calls"],
            "n_messages": s["n_messages"],
            "duration_s": s["duration_s"],
            "n_format_errors": s["n_format_errors"],
            "n_think_leaks": s["n_think_leaks"],
            "tool_efficiency": s["tool_efficiency"],
            "prompt_tokens": s["prompt_tokens"],
            "completion_tokens": s["completion_tokens"],
        })

    # Write summary.json
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "model_name": model_name,
        "training_step": training_step,
        "timestamp": datetime.now().isoformat(),
        "aggregates": aggregates,
        "scenarios": scenario_summaries,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    # Write per-scenario detail files
    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    for s in scenarios:
        detail = {
            "scenario_id": s["scenario_id"],
            "category": s["category"],
            "task_prompt": s["task_prompt"],
            "expected_answer": s["expected_answer"],
            "model_answer": s["model_answer"],
            "oracle_events": s["oracle_events"],
            "validation_decision": s["validation_decision"],
            "validation_rationale": s["validation_rationale"],
            "per_tool": s["per_tool"],
            "conversation": s["conversation"],
            "metrics": {
                "tool_score": s["tool_score"],
                "total_tokens": s["prompt_tokens"] + s["completion_tokens"],
                "prompt_tokens": s["prompt_tokens"],
                "completion_tokens": s["completion_tokens"],
                "duration_s": s["duration_s"],
                "tool_efficiency": s["tool_efficiency"],
                "failure_type": s["failure_type"],
            },
        }
        detail_path = details_dir / f"{s['scenario_id']}.json"
        detail_path.write_text(json.dumps(detail, indent=2))

    print(f"Wrote {len(scenarios)} detail files to {details_dir}")

    # Update index.json
    index_path = output_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"runs": []}

    # Remove existing entry for this run_id if present
    index["runs"] = [r for r in index["runs"] if r.get("run_id") != run_id]

    index["runs"].append({
        "run_id": run_id,
        "model_name": model_name,
        "timestamp": summary["timestamp"],
        "training_step": training_step,
        "n_scenarios": len(scenarios),
        "overall_pass_rate": aggregates["overall"]["pass_rate"],
        "overall_avg_tool_score": aggregates["overall"]["avg_tool_score"],
    })

    # Sort by timestamp descending
    index["runs"].sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Updated {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Export ARE traces to viewer JSON")
    parser.add_argument(
        "--are-traces",
        type=Path,
        required=True,
        help="Directory containing .jsonl trace files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Human-readable model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for viewer data",
    )
    parser.add_argument(
        "--training-step",
        type=int,
        default=None,
        help="Training step number (optional)",
    )

    args = parser.parse_args()
    export(args.are_traces, args.model_name, args.output_dir, args.training_step)


if __name__ == "__main__":
    main()
