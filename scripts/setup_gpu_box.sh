#!/bin/bash
set -euo pipefail

# ============================================================================
# GPU Box Setup for autoresearch-distillation
# Tested on: Ubuntu 22.04, H100 80GB, Python 3.12, CUDA 12.x
#
# Sets up: vLLM serving, QLoRA fine-tuning deps, experiment orchestration
#
# Usage: bash scripts/setup_gpu_box.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$HOME/autoresearch-env"

echo "============================================"
echo "  GPU Box Setup for autoresearch-distillation"
echo "  Project: $PROJECT_ROOT"
echo "============================================"

# ---- 1. Check prerequisites ----
echo ""
echo "[1/6] Checking prerequisites..."

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers required."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  GPU detected: $GPU_NAME"
echo "  GPU count: $GPU_COUNT"

# ---- 2. Install Python 3.12 if needed ----
echo ""
echo "[2/6] Checking Python 3.12..."

if ! command -v python3.12 &>/dev/null; then
    echo "  Installing Python 3.12 from deadsnakes PPA..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
else
    echo "  Python 3.12 already installed: $(python3.12 --version)"
fi

# ---- 3. Create virtual environment ----
echo ""
echo "[3/6] Setting up virtual environment at $VENV_DIR..."

if [ ! -d "$VENV_DIR" ]; then
    python3.12 -m venv "$VENV_DIR"
    echo "  Created new venv"
else
    echo "  Venv already exists"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# ---- 4. Install pinned dependencies ----
echo ""
echo "[4/6] Installing pinned dependencies (this takes a few minutes)..."

cd "$PROJECT_ROOT"

# Install core stack first (order matters)
pip install 'torch==2.9.0' 'torchvision==0.24.0' 'torchaudio==2.9.0' 'triton==3.5.0'

# Install vllm
pip install 'vllm==0.12.0'

# Install flash-attn prebuilt wheel (must match torch version + CXX11 ABI)
pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' --no-deps

# Upgrade opentelemetry (Ray 2.53.0 needs newer versions)
pip install \
    'opentelemetry-api==1.40.0' \
    'opentelemetry-sdk==1.40.0' \
    'opentelemetry-exporter-otlp==1.40.0' \
    'opentelemetry-exporter-otlp-proto-grpc==1.40.0' \
    'opentelemetry-exporter-otlp-proto-http==1.40.0' \
    'opentelemetry-exporter-otlp-proto-common==1.40.0' \
    'opentelemetry-exporter-prometheus==0.61b0' \
    'opentelemetry-proto==1.40.0' \
    'opentelemetry-semantic-conventions==0.61b0'

# Install remaining deps from requirements file
pip install -r requirements.txt 2>/dev/null || true

# ---- 5. Mount NVMe scratch (if available) ----
echo ""
echo "[5/6] Setting up scratch storage..."

NVME_DEV=""
for dev in /dev/nvme0n1 /dev/nvme1n1; do
    if [ -b "$dev" ] && ! mount | grep -q "$dev"; then
        NVME_DEV="$dev"
        break
    fi
done

if [ -n "$NVME_DEV" ]; then
    if ! mount | grep -q "/scratch"; then
        echo "  Formatting $NVME_DEV as scratch storage..."
        sudo mkfs.ext4 -L nvme-scratch "$NVME_DEV"
        sudo mkdir -p /scratch
        sudo mount "$NVME_DEV" /scratch
        sudo chown "$(whoami):$(id -gn)" /scratch
    fi
    mkdir -p /scratch/checkpoints /scratch/output /scratch/cache
    echo "  Scratch mounted at /scratch ($(df -h /scratch | tail -1 | awk '{print $4}') free)"
else
    echo "  No unmounted NVMe found, using home directory for storage"
    mkdir -p "$HOME/checkpoints" "$HOME/output"
fi

# ---- 6. Verify imports ----
echo ""
echo "[6/6] Verifying key imports..."

python -c "
import torch
print(f'  torch={torch.__version__}, cuda={torch.version.cuda}, gpus={torch.cuda.device_count()}')
import vllm
print(f'  vllm={vllm.__version__}')
import flash_attn
print(f'  flash_attn={flash_attn.__version__}')
import ray
print(f'  ray={ray.__version__}')
print('  All imports successful!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To verify everything works:"
echo "    bash $PROJECT_ROOT/scripts/smoke_test.sh"
echo ""
echo "  Activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo "============================================"
