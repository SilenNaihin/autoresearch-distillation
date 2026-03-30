#!/usr/bin/env python3
"""Import ARE baseline results from GPU boxes into the viewer data directory.

Reads output.jsonl (scores) + lite/ traces (conversations) from the ARE benchmark
output directories on remote boxes, converts to viewer format, and writes locally.

Usage:
    python scripts/import_are_baselines.py
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VIEWER_DATA = Path(__file__).resolve().parent.parent / "viewer" / "public" / "data"

# Runs to import: (ssh_host, remote_base_dir, model_name)
RUNS = [
    (
        "h100-dev-box-4",
        "/data/gaia2_baselines/Qwen3-14B/20260329_200215_full",
        "Qwen3-14B",
    ),
    (
        "h100-dev-box-4",
        "/data/gaia2_baselines/claude-haiku-4.5/us.anthropic.claude-haiku-4-5-20251001-v1:0/20260329_205059_full",
        "Claude Haiku 4.5",
    ),
]

CATEGORIES = ["search", "time", "execution", "adaptability", "ambiguity"]


def ssh_read(host: str, path: str) -> str:
    result = subprocess.run(
        ["ssh", host, f"cat {path}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh cat {host}:{path} failed: {result.stderr[:200]}")
    return result.stdout


def ssh_ls(host: str, path: str) -> list[str]:
    result = subprocess.run(
        ["ssh", host, f"ls {path}"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        return []
    return [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def import_run(host: str, remote_base: str, model_name: str):
    run_id = slugify(model_name) + "-baseline"
    print(f"\n{'='*60}")
    print(f"Importing {model_name} from {host}:{remote_base}")
    print(f"Run ID: {run_id}")

    all_scenarios = []
    all_details = []

    for cat in CATEGORIES:
        output_jsonl_path = f"{remote_base}/{cat}/output.jsonl"
        lite_dir = f"{remote_base}/{cat}/lite"

        # Read output.jsonl (scores)
        try:
            raw = ssh_read(host, output_jsonl_path)
        except Exception as e:
            print(f"  {cat}: SKIP (no output.jsonl: {e})")
            continue

        results = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
        print(f"  {cat}: {len(results)} scenarios", end="")

        # Map scenario_id -> trace file
        trace_files = ssh_ls(host, lite_dir)
        trace_map = {}
        for tf in trace_files:
            # filename format: scenario_universe_21_1afh09_c1cfa689.json
            # scenario_id is everything before the last _hex.json
            sid = "_".join(tf.replace(".json", "").split("_")[:-1])
            trace_map[sid] = tf

        passed = 0
        for r in results:
            scenario_id = r.get("task_id", r.get("metadata", {}).get("scenario_id", ""))
            score = r.get("score", 0)
            meta = r.get("metadata", {})
            is_pass = score > 0 and meta.get("status") == "success"
            if is_pass:
                passed += 1

            # Build scenario summary
            summary_entry = {
                "scenario_id": scenario_id,
                "category": cat,
                "passed": is_pass,
                "tool_score": score,
                "failure_type": "success" if is_pass else (meta.get("status", "failed")),
                "n_tool_calls": 0,
                "n_messages": 0,
                "duration_s": 0,
                "n_format_errors": 0,
                "n_think_leaks": 0,
                "tool_efficiency": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

            # Try to load the lite trace for detail
            detail = {
                "scenario_id": scenario_id,
                "category": cat,
                "task_prompt": "",
                "expected_answer": "",
                "model_answer": "",
                "oracle_events": [],
                "validation_decision": "Valid" if is_pass else "Invalid",
                "validation_rationale": meta.get("rationale", "") or "",
                "per_tool": [],
                "conversation": [],
                "metrics": {
                    "tool_score": score,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "duration_s": 0,
                    "tool_efficiency": 0,
                    "failure_type": summary_entry["failure_type"],
                },
            }

            trace_filename = trace_map.get(scenario_id)
            if trace_filename:
                try:
                    trace_raw = ssh_read(host, f"{lite_dir}/{trace_filename}")
                    trace = json.loads(trace_raw)

                    detail["validation_decision"] = trace.get("validation_decision", detail["validation_decision"])
                    detail["validation_rationale"] = trace.get("validation_rationale", detail["validation_rationale"])
                    detail["metrics"]["duration_s"] = trace.get("run_duration", 0)
                    summary_entry["duration_s"] = trace.get("run_duration", 0)

                    # Extract conversation from interaction history
                    histories = trace.get("per_agent_interaction_histories", {})
                    for agent_id, messages in histories.items():
                        conversation = []
                        for msg in messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            conversation.append({"role": role, "content": content})

                            # Extract task prompt from user message
                            if role == "user" and not detail["task_prompt"]:
                                detail["task_prompt"] = content
                        detail["conversation"] = conversation
                        summary_entry["n_messages"] = len(conversation)
                        break  # just use first agent

                    # Extract token usage
                    usage = trace.get("per_agent_llm_usage_stats", {})
                    for agent_id, stats in usage.items():
                        if isinstance(stats, dict):
                            summary_entry["prompt_tokens"] = stats.get("prompt_tokens", 0)
                            summary_entry["completion_tokens"] = stats.get("completion_tokens", 0)
                            detail["metrics"]["prompt_tokens"] = summary_entry["prompt_tokens"]
                            detail["metrics"]["completion_tokens"] = summary_entry["completion_tokens"]
                            detail["metrics"]["total_tokens"] = summary_entry["prompt_tokens"] + summary_entry["completion_tokens"]
                        break

                except Exception as e:
                    pass  # keep the basic detail without trace data

            all_scenarios.append(summary_entry)
            all_details.append(detail)

        print(f", {passed} passed ({100*passed/max(len(results),1):.1f}%)")

    if not all_scenarios:
        print(f"  No data found for {model_name}")
        return

    # Compute aggregates
    total = len(all_scenarios)
    total_passed = sum(1 for s in all_scenarios if s["passed"])
    overall_pass_rate = total_passed / max(total, 1)
    scores = [s["tool_score"] for s in all_scenarios]
    avg_tool_score = sum(scores) / max(len(scores), 1)

    by_category = {}
    for cat in CATEGORIES:
        cat_scenarios = [s for s in all_scenarios if s["category"] == cat]
        if cat_scenarios:
            cat_passed = sum(1 for s in cat_scenarios if s["passed"])
            cat_scores = [s["tool_score"] for s in cat_scenarios]
            by_category[cat] = {
                "n": len(cat_scenarios),
                "pass_rate": cat_passed / len(cat_scenarios),
                "avg_tool_score": sum(cat_scores) / len(cat_scores),
            }

    failure_types = {}
    for s in all_scenarios:
        ft = s["failure_type"]
        failure_types[ft] = failure_types.get(ft, 0) + 1

    summary = {
        "run_id": run_id,
        "model_name": model_name,
        "training_step": None,
        "timestamp": "2026-03-29T20:00:00Z",
        "aggregates": {
            "overall": {
                "n": total,
                "pass_rate": overall_pass_rate,
                "avg_tool_score": avg_tool_score,
            },
            "by_category": by_category,
            "failure_types": failure_types,
            "top_missing_tools": [],
        },
        "scenarios": all_scenarios,
    }

    # Write files
    run_dir = VIEWER_DATA / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote summary: {run_dir / 'summary.json'}")

    for detail in all_details:
        detail_path = details_dir / f"{detail['scenario_id']}.json"
        detail_path.write_text(json.dumps(detail, indent=2))

    print(f"  Wrote {len(all_details)} detail files")

    # Update index
    index_path = VIEWER_DATA / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"runs": []}

    index["runs"] = [r for r in index["runs"] if r.get("run_id") != run_id]
    index["runs"].append({
        "run_id": run_id,
        "model_name": model_name,
        "timestamp": summary["timestamp"],
        "training_step": None,
        "n_scenarios": total,
        "overall_pass_rate": overall_pass_rate,
        "overall_avg_tool_score": avg_tool_score,
    })
    index["runs"].sort(key=lambda r: r.get("model_name", ""))
    index_path.write_text(json.dumps(index, indent=2))
    print(f"  Updated index: {index_path}")
    print(f"  Overall: {total_passed}/{total} passed ({overall_pass_rate:.1%}), avg score: {avg_tool_score:.3f}")


def main():
    VIEWER_DATA.mkdir(parents=True, exist_ok=True)
    for host, remote_base, model_name in RUNS:
        import_run(host, remote_base, model_name)
    print("\nDone! Restart the viewer to see results.")


if __name__ == "__main__":
    main()
