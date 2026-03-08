#!/bin/bash
set -euo pipefail

# ============================================================================
# SDPO Training Launcher for Autoresearch
# 2 GPUs for verl (colocated vLLM + FSDP2), 8 GPUs for experiments
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SDPO_ROOT="$PROJECT_ROOT/SDPO"
PYTHON="${PYTHON:-/data/envs/vllm/bin/python}"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# Defaults
EXPERIMENT_NAME="${1:-qwen3-4b-sdpo}"

export RAY_TMPDIR=/data/tmp
export TMPDIR=/data/tmp
CONFIG_NAME="autoresearch_sdpo"
DATA_DIR="$PROJECT_ROOT/data/autoresearch"

# Ensure data is prepared
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "Data not found. Preparing autoresearch dataset..."
    $PYTHON "$PROJECT_ROOT/data/prepare_autoresearch.py"
fi

# Export required variables
export WANDB_PROJECT="${WANDB_PROJECT:-autoresearch-sdpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EXPERIMENT="$EXPERIMENT_NAME"
export TASK="data/autoresearch"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

# Disable VLLM_ATTENTION_BACKEND to let vLLM auto-select
unset VLLM_ATTENTION_BACKEND

# Set ulimit
ulimit -c 0

# Copy our config into SDPO's config directory so Hydra can find it
cp "$PROJECT_ROOT/configs/autoresearch_sdpo.yaml" \
   "$SDPO_ROOT/verl/trainer/config/autoresearch_sdpo.yaml"

# SDPO first on PYTHONPATH so `import verl` resolves to SDPO/verl
# Project root second so `import agent_loop` finds our custom loop
export PYTHONPATH="$SDPO_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

# Ensure ~/.local/bin is on PATH for uv
export PATH="$HOME/.local/bin:$PATH"

cd "$SDPO_ROOT"

echo "============================================"
echo "  SDPO Training: Autoresearch"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Config: $CONFIG_NAME"
echo "  Data: $DATA_DIR"
echo "============================================"

# Shift past the experiment name if provided
shift 2>/dev/null || true

$PYTHON -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" \
    vars.dir="$PROJECT_ROOT" \
    vars.ckpt_dir="$PROJECT_ROOT/checkpoints" \
    "$@"
