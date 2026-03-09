#!/bin/bash
set -euo pipefail

# ============================================================================
# Smoke Test — runs 1 batch / 1 training step end-to-end
# Verifies the full pipeline: rollout → GPU experiment → reward → SDPO update
# Usage: bash scripts/run_smoke_test.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SDPO_ROOT="$PROJECT_ROOT/SDPO"
PYTHON="${PYTHON:-/data/envs/vllm/bin/python}"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export RAY_TMPDIR=/data/tmp
export TMPDIR=/data/tmp
CONFIG_NAME="smoke_test"
DATA_DIR="$PROJECT_ROOT/data/autoresearch"

# Ensure data is prepared
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "Data not found. Preparing autoresearch dataset..."
    $PYTHON "$PROJECT_ROOT/data/prepare_autoresearch.py"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-autoresearch-sdpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EXPERIMENT="smoke-test"
export TASK="data/autoresearch"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0,1

unset VLLM_ATTENTION_BACKEND
ulimit -c 0

# Copy configs into SDPO's config directory so Hydra can find them
cp "$PROJECT_ROOT/configs/autoresearch_sdpo.yaml" \
   "$SDPO_ROOT/verl/trainer/config/autoresearch_sdpo.yaml"
cp "$PROJECT_ROOT/configs/smoke_test.yaml" \
   "$SDPO_ROOT/verl/trainer/config/smoke_test.yaml"

export PYTHONPATH="$SDPO_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"

cd "$SDPO_ROOT"

echo "============================================"
echo "  Smoke Test: 1 batch, 1 step"
echo "  Config: $CONFIG_NAME"
echo "  Data: $DATA_DIR"
echo "============================================"

$PYTHON "$PROJECT_ROOT/run_sdpo.py" \
    --config-name "$CONFIG_NAME" \
    vars.dir="$PROJECT_ROOT" \
    vars.ckpt_dir="/data/checkpoints" \
    "$@"
