#!/bin/bash
set -euo pipefail

# ============================================================================
# Smoke Test — verifies GPU box is correctly set up
# Checks: Python, CUDA, torch, vLLM, flash-attn, Ray, GPU memory, NVMe
#
# Usage: bash scripts/smoke_test.sh
# ============================================================================

PASS=0
FAIL=0
WARN=0

pass() { echo "  PASS: $1"; ((PASS++)); }
fail() { echo "  FAIL: $1"; ((FAIL++)); }
warn() { echo "  WARN: $1"; ((WARN++)); }

echo "============================================"
echo "  Smoke Test"
echo "============================================"

# ---- GPU checks ----
echo ""
echo "[GPUs]"

if nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    pass "$GPU_COUNT x $GPU_NAME"
else
    fail "nvidia-smi not found"
fi

for i in $(seq 0 $((GPU_COUNT - 1))); do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" | xargs)
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$i" | xargs)
    if [ "$MEM_USED" -lt 1000 ]; then
        pass "GPU $i: ${MEM_USED}/${MEM_TOTAL} MiB (clean)"
    else
        warn "GPU $i: ${MEM_USED}/${MEM_TOTAL} MiB (in use)"
    fi
done

# ---- Python + venv ----
echo ""
echo "[Python]"

PYTHON_VER=$(python --version 2>&1)
if echo "$PYTHON_VER" | grep -q "3.12"; then
    pass "$PYTHON_VER"
else
    fail "Expected Python 3.12, got: $PYTHON_VER"
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
    pass "venv active: $VIRTUAL_ENV"
else
    warn "No virtual environment active"
fi

# ---- Package imports ----
echo ""
echo "[Packages]"

python -c "
import sys

checks = []

try:
    import torch
    ver = torch.__version__
    cuda = torch.version.cuda
    gpus = torch.cuda.device_count()
    if gpus > 0:
        checks.append(('PASS', f'torch={ver}, cuda={cuda}, gpus={gpus}'))
    else:
        checks.append(('FAIL', f'torch={ver} but no CUDA GPUs detected'))
except Exception as e:
    checks.append(('FAIL', f'torch import failed: {e}'))

try:
    import vllm
    checks.append(('PASS', f'vllm={vllm.__version__}'))
except Exception as e:
    checks.append(('FAIL', f'vllm import failed: {e}'))

try:
    import flash_attn
    checks.append(('PASS', f'flash_attn={flash_attn.__version__}'))
except Exception as e:
    checks.append(('FAIL', f'flash_attn import failed: {e}'))

try:
    import ray
    checks.append(('PASS', f'ray={ray.__version__}'))
except Exception as e:
    checks.append(('FAIL', f'ray import failed: {e}'))

try:
    import transformers
    checks.append(('PASS', f'transformers={transformers.__version__}'))
except Exception as e:
    checks.append(('FAIL', f'transformers import failed: {e}'))

try:
    import peft
    checks.append(('PASS', f'peft={peft.__version__}'))
except Exception as e:
    checks.append(('WARN', f'peft not installed (needed for QLoRA): {e}'))

for status, msg in checks:
    print(f'{status}:{msg}')
"  | while IFS=: read -r status msg; do
    case "$status" in
        PASS) pass "$msg" ;;
        FAIL) fail "$msg" ;;
        WARN) warn "$msg" ;;
    esac
done

# ---- CUDA functionality ----
echo ""
echo "[CUDA]"

python -c "
import torch
# Basic CUDA op
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c = a @ b
assert c.shape == (1000, 1000)
print('PASS:Basic CUDA matmul')

# Multi-GPU check
if torch.cuda.device_count() > 1:
    for i in range(torch.cuda.device_count()):
        x = torch.randn(100, device=f'cuda:{i}')
        print(f'PASS:GPU {i} tensor allocation')

# Flash attention
try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v)
    print('PASS:Flash attention forward pass')
except Exception as e:
    print(f'FAIL:Flash attention: {e}')

# BF16 support
if torch.cuda.is_bf16_supported():
    print('PASS:BF16 supported')
else:
    print('WARN:BF16 not supported')
" | while IFS=: read -r status msg; do
    case "$status" in
        PASS) pass "$msg" ;;
        FAIL) fail "$msg" ;;
        WARN) warn "$msg" ;;
    esac
done

# ---- Disk ----
echo ""
echo "[Storage]"

if mount | grep -q "/scratch"; then
    AVAIL=$(df -h /scratch | tail -1 | awk '{print $4}')
    pass "NVMe scratch: $AVAIL available"
else
    warn "No /scratch mount — checkpoints will use home directory"
fi

ROOT_AVAIL=$(df -h / | tail -1 | awk '{print $4}')
ROOT_PCT=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$ROOT_PCT" -lt 90 ]; then
    pass "Root disk: $ROOT_AVAIL available ($ROOT_PCT% used)"
else
    warn "Root disk low: $ROOT_AVAIL available ($ROOT_PCT% used)"
fi

# ---- Summary ----
echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed, $WARN warnings"
if [ "$FAIL" -eq 0 ]; then
    echo "  Status: READY"
else
    echo "  Status: NOT READY — fix failures above"
fi
echo "============================================"

exit "$FAIL"
