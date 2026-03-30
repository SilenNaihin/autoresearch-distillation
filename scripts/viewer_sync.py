#!/usr/bin/env python3
"""SSH discovery, sync, and local serve for GAIA2 viewer data.

Usage:
    python scripts/viewer_sync.py [--serve] [--config scripts/viewer_config.yaml]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VIEWER_DIR = REPO_ROOT / "viewer"
VIEWER_DATA_DIR = VIEWER_DIR / "data"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def ssh_run(host: str, cmd: str, timeout: int = 15) -> str | None:
    """Run a command over SSH, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def scp_file(host: str, remote_path: str, local_path: str) -> bool:
    """Copy a file from remote to local via scp."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["scp", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"{host}:{remote_path}", local_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def remote_mtime(host: str, path: str) -> float:
    """Get mtime of a remote file as epoch seconds."""
    out = ssh_run(host, f"stat -c %Y {path} 2>/dev/null || stat -f %m {path} 2>/dev/null")
    if out:
        try:
            return float(out.strip().splitlines()[-1])
        except ValueError:
            pass
    return 0.0


def discover_runs(config: dict) -> list[dict]:
    """SSH to each box to find summary.json files and run metadata."""
    discovered = []

    for box in config.get("boxes", []):
        host = box["ssh_host"]
        data_paths = box.get("data_paths", [])
        print(f"Discovering on {host}...")

        # Find summary.json files
        find_parts = []
        for dp in data_paths:
            find_parts.append(f"find {dp} -name summary.json -maxdepth 4 2>/dev/null")
        find_cmd = " ; ".join(find_parts)
        output = ssh_run(host, find_cmd, timeout=20)

        summaries = []
        if output:
            summaries = [line.strip() for line in output.splitlines() if line.strip()]

        # Check for active scaffold processes
        active_out = ssh_run(host, "pgrep -f gaia2_scaffold -c 2>/dev/null || echo 0")
        is_active = False
        if active_out:
            try:
                is_active = int(active_out.strip()) > 0
            except ValueError:
                pass

        for summary_path in summaries:
            # Derive run_id from path: .../runs/<run_id>/summary.json
            parts = Path(summary_path).parts
            run_id = None
            for i, part in enumerate(parts):
                if part == "runs" and i + 1 < len(parts):
                    run_id = parts[i + 1]
                    break

            if not run_id:
                # Fallback: use parent directory name
                run_id = Path(summary_path).parent.name

            # Check for .wandb_url file
            wandb_url = None
            wandb_path = str(Path(summary_path).parent / ".wandb_url")
            wandb_out = ssh_run(host, f"cat {wandb_path} 2>/dev/null")
            if wandb_out:
                wandb_url = wandb_out.strip()

            discovered.append({
                "run_id": run_id,
                "host": host,
                "remote_summary": summary_path,
                "wandb_url": wandb_url,
                "is_active": is_active,
            })

    print(f"Discovered {len(discovered)} runs across {len(config.get('boxes', []))} boxes")
    return discovered


def sync_runs(discovered: list[dict]) -> list[dict]:
    """Sync summary.json files from remote to local, only if remote is newer."""
    synced = []

    for run in discovered:
        run_id = run["run_id"]
        host = run["host"]
        remote_path = run["remote_summary"]

        local_dir = VIEWER_DATA_DIR / "runs" / run_id
        local_summary = local_dir / "summary.json"

        # Check if we need to sync
        needs_sync = True
        if local_summary.exists():
            local_mt = local_summary.stat().st_mtime
            remote_mt = remote_mtime(host, remote_path)
            if remote_mt > 0 and remote_mt <= local_mt:
                needs_sync = False

        if needs_sync:
            print(f"  Syncing {run_id} from {host}...")
            ok = scp_file(host, remote_path, str(local_summary))
            if ok:
                synced.append(run)
                print(f"    OK")
            else:
                print(f"    FAILED to scp {remote_path}")
        else:
            synced.append(run)

    return synced


def generate_index(discovered: list[dict]) -> None:
    """Create viewer/data/index.json from all synced summaries."""
    runs_dir = VIEWER_DATA_DIR / "runs"
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    # Build a lookup from run_id to discovery metadata
    meta_by_run = {}
    for d in discovered:
        meta_by_run[d["run_id"]] = d

    index_runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        summary_file = run_dir / "summary.json"
        if not summary_file.exists():
            continue

        try:
            summary = json.loads(summary_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        run_id = summary.get("run_id", run_dir.name)
        meta = meta_by_run.get(run_id, {})
        aggregates = summary.get("aggregates", {})
        overall = aggregates.get("overall", {})

        index_runs.append({
            "run_id": run_id,
            "model_name": summary.get("model_name", run_id),
            "timestamp": summary.get("timestamp", ""),
            "training_step": summary.get("training_step"),
            "n_scenarios": overall.get("n", len(summary.get("scenarios", []))),
            "overall_pass_rate": overall.get("pass_rate", 0.0),
            "overall_avg_tool_score": overall.get("avg_tool_score", 0.0),
            "box": meta.get("host", "local"),
            "wandb_url": meta.get("wandb_url"),
            "status": "active" if meta.get("is_active") else "complete",
        })

    index_runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    index_path = VIEWER_DATA_DIR / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"runs": index_runs}, indent=2))
    print(f"Generated index with {len(index_runs)} runs at {index_path}")


class ViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves viewer files and on-demand detail fetching."""

    def __init__(self, *args, config=None, discovered=None, **kwargs):
        self._config = config or {}
        self._discovered = discovered or []
        super().__init__(*args, **kwargs)

    def do_GET(self):
        # Handle /api/detail/<run_id>/<scenario_id>
        if self.path.startswith("/api/detail/"):
            parts = self.path.split("/")
            # /api/detail/<run_id>/<scenario_id>
            if len(parts) >= 5:
                run_id = parts[3]
                scenario_id = parts[4].split("?")[0]  # strip query params
                self._serve_detail(run_id, scenario_id)
                return

        # Serve static files from viewer directory
        super().do_GET()

    def _serve_detail(self, run_id: str, scenario_id: str):
        local_path = VIEWER_DATA_DIR / "runs" / run_id / "details" / f"{scenario_id}.json"

        # If not cached locally, fetch on demand
        if not local_path.exists():
            host = self._find_host(run_id)
            if host:
                # Look for detail on remote
                remote_detail = None
                for d in self._discovered:
                    if d["run_id"] == run_id:
                        remote_run_dir = str(Path(d["remote_summary"]).parent)
                        remote_detail = f"{remote_run_dir}/details/{scenario_id}.json"
                        break

                if remote_detail:
                    print(f"  On-demand fetch: {run_id}/{scenario_id} from {host}")
                    scp_file(host, remote_detail, str(local_path))

        if local_path.exists():
            data = local_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "detail not found"}).encode())

    def _find_host(self, run_id: str) -> str | None:
        for d in self._discovered:
            if d["run_id"] == run_id:
                return d["host"]
        return None

    def log_message(self, format, *args):
        # Quieter logging
        if "/api/" in (args[0] if args else ""):
            super().log_message(format, *args)


def serve(config: dict, discovered: list[dict], open_browser: bool = False):
    """Start local HTTP server."""
    port = config.get("viewer_port", 9000)

    # Change to viewer directory so static files are served correctly
    os.chdir(VIEWER_DIR)

    handler = partial(ViewerHandler, config=config, discovered=discovered)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)

    url = f"http://localhost:{port}"
    print(f"Serving viewer at {url}")
    print("Press Ctrl+C to stop")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Sync and serve GAIA2 viewer data")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "viewer_config.yaml"),
        help="Path to viewer config YAML",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start local HTTP server and open browser",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync data, don't start server",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Discovery and sync
    print("=== Discovery ===")
    discovered = discover_runs(config)

    print("\n=== Sync ===")
    synced = sync_runs(discovered)

    print("\n=== Index ===")
    generate_index(discovered)

    if args.sync_only:
        return

    if args.serve:
        print("\n=== Server ===")
        serve(config, discovered, open_browser=True)
    else:
        print("\nDone. Use --serve to start the local viewer server.")


if __name__ == "__main__":
    main()
