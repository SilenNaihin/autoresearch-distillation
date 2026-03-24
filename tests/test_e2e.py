"""
End-to-end smoke test for all GPU slots in the fleet.

Tests the full SSHRunner pipeline on each GPU:
  1. SSH connect + file sync
  2. uv run on remote with correct CUDA_VISIBLE_DEVICES
  3. torch.cuda works on the assigned GPU
  4. Output parsing (fake metrics in train.py format)

Does NOT run a 5-minute experiment — uses a tiny stub script (~2s).

Usage:
    python test_e2e.py              # test all 6 GPUs
    python test_e2e.py --setup      # setup machines first, then test
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from lib.runners import FLEET, GPUSlot, SSHRunner, setup_fleet
from lib.environment import RunOutput, parse_metrics

# A tiny train.py that verifies GPU access and prints metrics in the expected format.
SMOKE_TRAIN_PY = '''\
import torch
import sys

if not torch.cuda.is_available():
    print("ERROR: CUDA not available", file=sys.stderr)
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU: {gpu_name} ({mem_gb:.0f} GB)")
print("---")
print(f"val_bpb:          0.998000")
print(f"training_seconds: 0.1")
print(f"total_seconds:    0.2")
print(f"peak_vram_mb:     {torch.cuda.mem_get_info()[1] / 1024 / 1024:.1f}")
print(f"mfu_percent:      0.0")
print(f"total_tokens_M:   0.0")
print(f"num_steps:        0")
print(f"num_params_M:     0.0")
print(f"depth:            0")
'''


def test_slot(slot: GPUSlot) -> dict:
    """Run full SSHRunner pipeline on a slot. Returns result dict."""
    result = {"slot": slot.name, "host": slot.host, "gpu_id": slot.gpu_id}

    runner = SSHRunner(slot, timeout=60)
    t0 = time.time()
    output = runner.run(SMOKE_TRAIN_PY)
    elapsed = time.time() - t0

    result["elapsed_s"] = round(elapsed, 1)
    result["returncode"] = output.returncode

    if output.returncode != 0:
        result["status"] = "FAIL"
        result["error"] = (output.stderr or output.stdout or "no output").strip()[-300:]
        return result

    metrics = parse_metrics(output.stdout)
    if "val_bpb" not in metrics:
        result["status"] = "FAIL"
        result["error"] = f"No val_bpb in output: {output.stdout.strip()[-200:]}"
        return result

    result["status"] = "OK"
    # Extract the GPU info line from stdout
    for line in output.stdout.splitlines():
        if line.startswith("GPU:"):
            result["gpu_info"] = line
            break
    result["val_bpb"] = metrics["val_bpb"]
    result["peak_vram_mb"] = metrics.get("peak_vram_mb", 0)

    return result


def main():
    do_setup = "--setup" in sys.argv

    if do_setup:
        print("Setting up fleet machines...\n")
        results = setup_fleet(local_repo="autoresearch")
        for name, ok in results.items():
            if not ok:
                print(f"  WARNING: setup failed for {name}")
        print()

    print(f"Testing {len(FLEET)} GPU slots end-to-end:\n")
    print(f"  {'Slot':<15s}  {'Status':<6s}  {'Time':>5s}  {'GPU':>40s}  {'VRAM':>10s}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*5}  {'-'*40}  {'-'*10}")

    results = []
    for slot in FLEET:
        r = test_slot(slot)
        results.append(r)

        if r["status"] == "OK":
            gpu = r.get("gpu_info", "?")
            vram = f"{r.get('peak_vram_mb', 0):.0f} MB"
            print(f"  {r['slot']:<15s}  {'OK':<6s}  {r['elapsed_s']:>4.1f}s  {gpu:>40s}  {vram:>10s}")
        else:
            err = r.get("error", "unknown error")
            # Show first line of error
            err_short = err.splitlines()[-1][:60] if err else "?"
            print(f"  {r['slot']:<15s}  {'FAIL':<6s}  {r.get('elapsed_s', 0):>4.1f}s  {err_short}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n  {ok_count}/{len(FLEET)} slots passed.")

    if ok_count < len(FLEET):
        print("\n  Failed slots:")
        for r in results:
            if r["status"] != "OK":
                print(f"    {r['slot']}: {r.get('error', '?')[:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
