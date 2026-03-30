#!/usr/bin/env python3
"""Discover, sync, and serve GAIA2 viewer data from GPU boxes.

Single command to get the viewer running with all available baseline and
training run data. Discovers ARE benchmark output directories on remote
boxes, converts them to viewer format, and starts the Next.js dev server.

Usage:
    python scripts/viewer_sync.py --serve        # sync + start viewer
    python scripts/viewer_sync.py                # sync only
    python scripts/viewer_sync.py --full         # sync including all detail files
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Install pyyaml: pip install pyyaml")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VIEWER_DATA = REPO_ROOT / "viewer" / "public" / "data"
CATEGORIES = ["search", "time", "execution", "adaptability", "ambiguity"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ssh_run(host: str, cmd: str, timeout: int = 15) -> str | None:
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, Exception):
        return None


def ssh_read_json(host: str, path: str) -> dict | list | None:
    raw = ssh_run(host, f"cat {path}", timeout=30)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


def ssh_read_jsonl(host: str, path: str) -> list[dict]:
    raw = ssh_run(host, f"cat {path}", timeout=30)
    if not raw:
        return []
    results = []
    for line in raw.strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def scp_file(host: str, remote: str, local: str) -> bool:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["scp", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"{host}:{remote}", local],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------------------
# Discovery: find ARE benchmark runs on remote boxes
# ---------------------------------------------------------------------------

def discover_are_runs(config: dict) -> list[dict]:
    """Find ARE benchmark run directories by looking for output.jsonl files."""
    discovered = []

    for box in config.get("boxes", []):
        host = box["ssh_host"]
        data_paths = box.get("data_paths", [])
        print(f"  {host}...", end="", flush=True)

        # Check connectivity
        if ssh_run(host, "echo ok", timeout=5) is None:
            print(" unreachable")
            continue

        # Find output.jsonl files (ARE benchmark output marker)
        find_cmds = [f"find {dp} -name output.jsonl -maxdepth 6 2>/dev/null" for dp in data_paths]
        raw = ssh_run(host, " ; ".join(find_cmds), timeout=20)
        if not raw:
            print(" no runs found")
            continue

        # Group by run directory (parent of category dir)
        # Structure: .../ModelName/timestamp_full/category/output.jsonl
        run_dirs: dict[str, list[str]] = {}
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cat_dir = str(Path(line).parent)
            cat_name = Path(cat_dir).name
            if cat_name in CATEGORIES:
                run_dir = str(Path(cat_dir).parent)
                run_dirs.setdefault(run_dir, []).append(cat_name)

        for run_dir, cats in run_dirs.items():
            # Derive model name from path
            # e.g. /data/gaia2_baselines/Qwen3-14B/20260329_200215_full
            parts = Path(run_dir).parts
            # Model name is typically 2 levels up from the timestamp dir
            model_name = parts[-2] if len(parts) >= 2 else parts[-1]
            timestamp = parts[-1] if len(parts) >= 1 else ""

            discovered.append({
                "host": host,
                "remote_dir": run_dir,
                "model_name": model_name,
                "timestamp": timestamp,
                "categories": sorted(cats),
            })

        print(f" {len(run_dirs)} run(s)")

    return discovered


# ---------------------------------------------------------------------------
# Conversion: ARE output → viewer format
# ---------------------------------------------------------------------------

def convert_run(host: str, remote_dir: str, model_name: str, categories: list[str],
                run_id: str, fetch_details: bool = False) -> dict | None:
    """Convert an ARE benchmark run to viewer summary format.

    Pulls output.jsonl per category (small) to build the summary.
    Detail files (lite traces) are fetched only if fetch_details=True.
    """
    run_dir = VIEWER_DATA / "runs" / run_id
    summary_path = run_dir / "summary.json"

    # Skip if already converted and not requesting full detail sync
    if summary_path.exists() and not fetch_details:
        summary = json.loads(summary_path.read_text())
        print(f"    {run_id}: already synced ({len(summary.get('scenarios', []))} scenarios)")
        return summary

    print(f"    {run_id}: syncing from {host}...")

    all_scenarios = []
    all_details = []

    for cat in categories:
        results = ssh_read_jsonl(host, f"{remote_dir}/{cat}/output.jsonl")
        if not results:
            continue

        # Get lite trace file list for detail mapping
        trace_list_raw = ssh_run(host, f"ls {remote_dir}/{cat}/lite/ 2>/dev/null")
        trace_map = {}
        if trace_list_raw:
            for tf in trace_list_raw.strip().split("\n"):
                tf = tf.strip()
                if tf.endswith(".json"):
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

            all_scenarios.append({
                "scenario_id": scenario_id,
                "category": cat,
                "passed": is_pass,
                "tool_score": score,
                "failure_type": "success" if is_pass else meta.get("status", "failed"),
                "n_tool_calls": 0,
                "n_messages": 0,
                "duration_s": 0,
                "n_format_errors": 0,
                "n_think_leaks": 0,
                "tool_efficiency": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            })

            # Store enough info to lazily fetch detail later
            all_details.append({
                "scenario_id": scenario_id,
                "category": cat,
                "remote_trace": f"{remote_dir}/{cat}/lite/{trace_map.get(scenario_id, '')}",
                "has_trace": scenario_id in trace_map,
                "score": score,
                "is_pass": is_pass,
                "rationale": meta.get("rationale", "") or "",
            })

        print(f"      {cat}: {len(results)} scenarios, {passed} passed")

    if not all_scenarios:
        return None

    # Compute aggregates
    total = len(all_scenarios)
    total_passed = sum(1 for s in all_scenarios if s["passed"])
    scores = [s["tool_score"] for s in all_scenarios]

    by_category = {}
    for cat in CATEGORIES:
        cat_s = [s for s in all_scenarios if s["category"] == cat]
        if cat_s:
            cat_passed = sum(1 for s in cat_s if s["passed"])
            cat_scores = [s["tool_score"] for s in cat_s]
            by_category[cat] = {
                "n": len(cat_s),
                "pass_rate": cat_passed / len(cat_s),
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
                "pass_rate": total_passed / max(total, 1),
                "avg_tool_score": sum(scores) / max(len(scores), 1),
            },
            "by_category": by_category,
            "failure_types": failure_types,
            "top_missing_tools": [],
        },
        "scenarios": all_scenarios,
    }

    # Write summary
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    # Write a manifest so the detail API route can lazily fetch traces
    manifest = {d["scenario_id"]: {
        "host": host,
        "remote_trace": d["remote_trace"],
        "has_trace": d["has_trace"],
        "category": d["category"],
        "score": d["score"],
        "is_pass": d["is_pass"],
        "rationale": d["rationale"],
    } for d in all_details}
    (run_dir / ".manifest.json").write_text(json.dumps(manifest))

    # Optionally fetch all detail files
    if fetch_details:
        fetch_all_details(host, run_id, all_details)

    return summary


def fetch_all_details(host: str, run_id: str, details: list[dict]):
    """Pull all lite trace files for a run."""
    details_dir = VIEWER_DATA / "runs" / run_id / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    fetched = 0
    for d in details:
        if not d["has_trace"]:
            continue
        local_path = details_dir / f"{d['scenario_id']}.json"
        if local_path.exists():
            continue
        if scp_file(host, d["remote_trace"], str(local_path)):
            # Convert lite trace to viewer detail format
            try:
                convert_lite_trace(local_path, d)
            except Exception:
                pass
            fetched += 1

    print(f"      Fetched {fetched} detail files")


def fetch_single_detail(run_id: str, scenario_id: str) -> bool:
    """Lazily fetch a single detail file from the remote box."""
    run_dir = VIEWER_DATA / "runs" / run_id
    manifest_path = run_dir / ".manifest.json"
    if not manifest_path.exists():
        return False

    manifest = json.loads(manifest_path.read_text())
    info = manifest.get(scenario_id)
    if not info or not info.get("has_trace"):
        return False

    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    local_path = details_dir / f"{scenario_id}.json"

    # SCP the lite trace
    tmp_path = str(local_path) + ".tmp"
    if not scp_file(info["host"], info["remote_trace"], tmp_path):
        return False

    # Convert to viewer detail format
    try:
        convert_lite_trace(Path(tmp_path), info)
        Path(tmp_path).rename(local_path)
        return True
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        return False


def convert_lite_trace(path: Path, meta: dict):
    """Convert an ARE lite trace JSON to viewer detail format in-place."""
    trace = json.loads(path.read_text())

    # Extract conversation
    conversation = []
    task_prompt = ""
    histories = trace.get("per_agent_interaction_histories", {})
    for agent_id, messages in histories.items():
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            conversation.append({"role": role, "content": content})
            if role == "user" and not task_prompt:
                task_prompt = content
        break

    # Extract token usage
    prompt_tokens = 0
    completion_tokens = 0
    usage = trace.get("per_agent_llm_usage_stats", {})
    for agent_id, stats in usage.items():
        if isinstance(stats, dict):
            prompt_tokens = stats.get("prompt_tokens", 0)
            completion_tokens = stats.get("completion_tokens", 0)
        break

    detail = {
        "scenario_id": meta.get("scenario_id", trace.get("scenario_id", "")),
        "category": meta.get("category", ""),
        "task_prompt": task_prompt,
        "expected_answer": "",
        "model_answer": "",
        "oracle_events": [],
        "validation_decision": trace.get("validation_decision", "Valid" if meta.get("is_pass") else "Invalid"),
        "validation_rationale": trace.get("validation_rationale", meta.get("rationale", "")),
        "per_tool": [],
        "conversation": conversation,
        "metrics": {
            "tool_score": meta.get("score", 0),
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_s": trace.get("run_duration", 0),
            "tool_efficiency": 0,
            "failure_type": "success" if meta.get("is_pass") else "failed",
        },
    }

    path.write_text(json.dumps(detail, indent=2))


# ---------------------------------------------------------------------------
# Training run discovery
# ---------------------------------------------------------------------------

def discover_training_runs(config: dict) -> list[dict]:
    """Find active/recent training runs by checking wandb dirs and processes."""
    training_runs = []

    for box in config.get("boxes", []):
        host = box["ssh_host"]
        print(f"  {host}...", end="", flush=True)

        if ssh_run(host, "echo ok", timeout=5) is None:
            print(" unreachable")
            continue

        # Find wandb URLs from claas.log (active training)
        wandb_info = ssh_run(host, r"""
            grep -h 'View run at' /tmp/claas.log 2>/dev/null | tail -1;
            echo '---';
            grep -h 'Syncing run' /tmp/claas.log 2>/dev/null | tail -1;
            echo '---';
            pgrep -fa 'serve.main|verl|sdpo|wake_sleep|train' 2>/dev/null | grep -v grep | head -1;
            echo '---';
            ls -d /data/checkpoints/*/global_step_* 2>/dev/null;
            echo '---';
            nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
        """, timeout=15)

        if not wandb_info:
            print(" no data")
            continue

        parts = wandb_info.split("---")
        wandb_url_line = parts[0].strip() if len(parts) > 0 else ""
        run_name_line = parts[1].strip() if len(parts) > 1 else ""
        active_process = parts[2].strip() if len(parts) > 2 else ""
        checkpoint_dirs = parts[3].strip() if len(parts) > 3 else ""
        gpu_name = parts[4].strip() if len(parts) > 4 else ""

        # Parse wandb URL
        wandb_url = ""
        if "https://" in wandb_url_line:
            wandb_url = wandb_url_line.split("https://")[-1]
            wandb_url = "https://" + wandb_url.strip().rstrip(")")

        # Parse run name
        run_name = ""
        if "Syncing run" in run_name_line:
            run_name = run_name_line.split("Syncing run")[-1].strip()

        if not wandb_url and not active_process:
            print(" no training runs")
            continue

        # Parse checkpoints — paths like /data/checkpoints/model-name/global_step_100
        checkpoints = []
        checkpoint_model = ""
        if checkpoint_dirs:
            for line in checkpoint_dirs.split("\n"):
                line = line.strip().rstrip("/")
                name = Path(line).name
                if name.startswith("global_step_"):
                    try:
                        checkpoints.append(int(name.replace("global_step_", "")))
                    except ValueError:
                        pass
                    if not checkpoint_model:
                        checkpoint_model = Path(line).parent.name
        checkpoints.sort()

        # Determine model name from run name, checkpoint path, or fallback
        model_name = run_name or checkpoint_model or "Unknown"

        is_active = bool(active_process)

        # Extract wandb project from URL
        wandb_project = ""
        if wandb_url:
            # https://wandb.ai/team/project/runs/id
            url_parts = wandb_url.rstrip("/").split("/")
            if len(url_parts) >= 5:
                wandb_project = url_parts[-3] if "runs" in url_parts else ""

        training_runs.append({
            "run_id": f"training-{host}-{slugify(model_name)}",
            "run_name": run_name,
            "model_name": model_name,
            "box": host,
            "wandb_url": wandb_url,
            "wandb_project": wandb_project,
            "started_at": "",
            "status": "active" if is_active else "complete",
            "checkpoints": checkpoints,
            "gpu": gpu_name,
        })

        status = "active" if is_active else "complete"
        print(f" {run_name} ({status}, {len(checkpoints)} checkpoints)")

    return training_runs


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------

def generate_index(training_runs: list[dict] | None = None):
    """Generate index.json from all synced runs + training runs."""
    runs_dir = VIEWER_DATA / "runs"
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    index_runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        summary_file = run_dir / "summary.json"
        if not summary_file.exists():
            continue
        try:
            s = json.loads(summary_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        overall = s.get("aggregates", {}).get("overall", {})
        index_runs.append({
            "run_id": s.get("run_id", run_dir.name),
            "model_name": s.get("model_name", run_dir.name),
            "timestamp": s.get("timestamp", ""),
            "training_step": s.get("training_step"),
            "n_scenarios": overall.get("n", len(s.get("scenarios", []))),
            "overall_pass_rate": overall.get("pass_rate", 0),
            "overall_avg_tool_score": overall.get("avg_tool_score", 0),
        })

    index_runs.sort(key=lambda r: r.get("model_name", ""))

    index = {"runs": index_runs}
    if training_runs:
        index["training_runs"] = training_runs

    index_path = VIEWER_DATA / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2))
    print(f"  Index: {len(index_runs)} eval runs, {len(training_runs or [])} training runs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sync and serve GAIA2 viewer data")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "viewer_config.yaml"))
    parser.add_argument("--serve", action="store_true", help="Start Next.js dev server after sync")
    parser.add_argument("--full", action="store_true", help="Also pull all detail files (slow)")
    parser.add_argument("--fetch-detail", nargs=2, metavar=("RUN_ID", "SCENARIO_ID"),
                        help="Lazily fetch a single detail file (used by Next.js API route)")
    args = parser.parse_args()

    # Single detail fetch mode (called by Next.js API route)
    if args.fetch_detail:
        run_id, scenario_id = args.fetch_detail
        ok = fetch_single_detail(run_id, scenario_id)
        sys.exit(0 if ok else 1)

    config = load_config(args.config)

    # Discover eval runs
    print("Discovering eval runs on GPU boxes...")
    runs = discover_are_runs(config)

    if not runs:
        print("  No eval runs found.")
    else:
        print(f"\nSyncing {len(runs)} eval run(s)...")
        for run in runs:
            run_id = slugify(run["model_name"]) + "-baseline"
            convert_run(
                host=run["host"],
                remote_dir=run["remote_dir"],
                model_name=run["model_name"],
                categories=run["categories"],
                run_id=run_id,
                fetch_details=args.full,
            )

    # Discover training runs
    print("\nDiscovering training runs...")
    training_runs = discover_training_runs(config)

    # Generate index
    print("\nUpdating index...")
    generate_index(training_runs)

    if args.serve:
        port = config.get("viewer_port", 9000)
        print(f"\nStarting viewer at http://localhost:{port}")
        os.chdir(REPO_ROOT / "viewer")
        os.execlp("npx", "npx", "next", "dev", "-p", str(port))
    else:
        print("\nDone. Run with --serve to start the viewer, or: cd viewer && npx next dev -p 9000")


if __name__ == "__main__":
    main()
