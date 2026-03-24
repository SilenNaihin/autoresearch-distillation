"""
Quick smoke test for the SSH runner infrastructure.

Tests the full pipeline WITHOUT running a 5-minute experiment:
  1. SSH connectivity to each slot
  2. File sync (write train.py content to remote)
  3. Python/uv availability
  4. CUDA visibility on the correct GPU
  5. torch import + GPU detection

Usage:
    python test_runner.py                    # test all fleet slots
    python test_runner.py box1-gpu0          # test specific slot
    python test_runner.py --setup            # setup + test all
"""

from __future__ import annotations

import subprocess
import sys
import time

from lib.runners import FLEET, GPUSlot, SSHRunner, setup_slot


def test_slot(slot: GPUSlot) -> dict:
    """Run a quick diagnostic on a single GPU slot. Returns a result dict."""
    result = {"slot": slot.name, "host": slot.host, "gpu": slot.gpu_id}

    # 1. SSH connectivity
    t0 = time.time()
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", slot.host, "echo ok"],
            capture_output=True, text=True, timeout=10,
        )
        result["ssh"] = r.returncode == 0
        result["ssh_latency_ms"] = int((time.time() - t0) * 1000)
        if not result["ssh"]:
            result["ssh_error"] = r.stderr.strip()[:100]
            return result
    except Exception as e:
        result["ssh"] = False
        result["ssh_error"] = str(e)[:100]
        return result

    # 2. Remote dir exists
    r = subprocess.run(
        ["ssh", slot.host, f"test -d {slot.remote_dir} && echo exists || echo missing"],
        capture_output=True, text=True, timeout=10,
    )
    result["remote_dir"] = r.stdout.strip()

    # 3. File sync via SSH stdin (same mechanism as SSHRunner)
    test_content = "# smoke test\nimport sys; print('SMOKE_OK'); sys.exit(0)\n"
    r = subprocess.run(
        ["ssh", slot.host, f"cat > /tmp/_smoke_test.py"],
        input=test_content, capture_output=True, text=True, timeout=10,
    )
    result["file_sync"] = r.returncode == 0

    # 4. uv available
    r = subprocess.run(
        ["ssh", slot.host, "export PATH=$HOME/.local/bin:$PATH && uv --version"],
        capture_output=True, text=True, timeout=10,
    )
    result["uv"] = r.stdout.strip() if r.returncode == 0 else f"MISSING ({r.stderr.strip()[:50]})"

    # 5. CUDA visibility on correct GPU
    cuda_check = (
        f"export PATH=$HOME/.local/bin:$PATH && "
        f"CUDA_VISIBLE_DEVICES={slot.gpu_id} python3 -c \""
        f"import subprocess; "
        f"r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader', '-i', '{slot.gpu_id}'], "
        f"capture_output=True, text=True); "
        f"print(r.stdout.strip() if r.returncode == 0 else 'GPU_ERROR: ' + r.stderr.strip()[:100])\""
    )
    r = subprocess.run(
        ["ssh", slot.host, cuda_check],
        capture_output=True, text=True, timeout=15,
    )
    result["gpu_info"] = r.stdout.strip() if r.returncode == 0 else f"ERROR: {r.stderr.strip()[:100]}"

    # 6. torch + CUDA check
    torch_check = (
        f"export PATH=$HOME/.local/bin:$PATH && "
        f"cd {slot.remote_dir} && "
        f"CUDA_VISIBLE_DEVICES={slot.gpu_id} uv run python -c \""
        f"import torch; "
        f"ok = torch.cuda.is_available(); "
        f"name = torch.cuda.get_device_name(0) if ok else 'N/A'; "
        f"mem = torch.cuda.get_device_properties(0).total_memory // (1024**3) if ok else 0; "
        f"print(f'cuda={{ok}} gpu={{name}} mem={{mem}}GB')\""
    )
    r = subprocess.run(
        ["ssh", slot.host, torch_check],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0:
        result["torch"] = r.stdout.strip()
    else:
        result["torch"] = f"FAILED: {(r.stderr or r.stdout).strip()[-200:]}"

    # 7. Data cache check
    r = subprocess.run(
        ["ssh", slot.host, "ls ~/.cache/autoresearch/tokenizer/tokenizer.pkl 2>/dev/null && echo data_ok || echo data_missing"],
        capture_output=True, text=True, timeout=10,
    )
    result["data"] = r.stdout.strip().splitlines()[-1]

    return result


def print_result(result: dict):
    name = result["slot"]
    ssh_ok = result.get("ssh", False)

    if not ssh_ok:
        print(f"  {name:15s}  SSH FAILED  {result.get('ssh_error', '')}")
        return

    latency = result.get("ssh_latency_ms", "?")
    remote = result.get("remote_dir", "?")
    uv = result.get("uv", "?")
    gpu = result.get("gpu_info", "?")
    torch_info = result.get("torch", "?")
    data = result.get("data", "?")

    status = "OK" if "cuda=True" in torch_info and data == "data_ok" else "NEEDS SETUP"

    print(f"  {name:15s}  {status:12s}  ssh={latency}ms  dir={remote}  uv={uv}")
    print(f"  {'':15s}  gpu: {gpu}")
    print(f"  {'':15s}  torch: {torch_info}")
    print(f"  {'':15s}  data: {data}")


def main():
    do_setup = "--setup" in sys.argv
    target = None
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            target = arg

    slots = FLEET
    if target:
        slots = [s for s in FLEET if s.name == target]
        if not slots:
            print(f"Unknown slot: {target}. Available: {[s.name for s in FLEET]}")
            sys.exit(1)

    if do_setup:
        print("Setting up fleet...\n")
        seen = set()
        for slot in slots:
            if slot.host not in seen:
                seen.add(slot.host)
                setup_slot(slot, local_repo="autoresearch")
        print()

    print(f"Testing {len(slots)} GPU slot(s):\n")
    for slot in slots:
        result = test_slot(slot)
        print_result(result)
        print()


if __name__ == "__main__":
    main()
