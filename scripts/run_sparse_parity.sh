#!/bin/bash
set -euo pipefail

# ============================================================================
# SDPO Training Launcher for Sparse Parity Challenge
# 1x A100 80GB (a100-backup-1), colocated vLLM + FSDP2
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
EXPERIMENT_NAME="${1:-qwen3-14b-sdpo-shared}"
CONFIG_NAME="${CONFIG_NAME:-sparse_parity_sdpo}"

export RAY_TMPDIR=/data/tmp
export TMPDIR=/data/tmp

DATA_DIR="$PROJECT_ROOT/data/sparse_parity"

# Ensure data is prepared
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "Data not found. Preparing sparse parity dataset..."
    $PYTHON "$PROJECT_ROOT/data/prepare_sparse_parity.py"
fi

# Export required variables
export WANDB_PROJECT="${WANDB_PROJECT:-sparse-parity-sdpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EXPERIMENT="$EXPERIMENT_NAME"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0

# Disable VLLM_ATTENTION_BACKEND to let vLLM auto-select
unset VLLM_ATTENTION_BACKEND

# Set ulimit
ulimit -c 0

# Patch HF cached config.json for YaRN (32k context for sparse parity)
$PYTHON -c "
import json, glob, os
target_max_pos = 32768
target_orig = 32768
target_factor = 1.0
for p in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/config.json')):
    real = os.path.realpath(p)
    with open(real) as f: c = json.load(f)
    changed = False
    if c.get('max_position_embeddings') != target_max_pos:
        c['max_position_embeddings'] = target_max_pos
        changed = True
    rs = c.get('rope_scaling')
    if rs and rs.get('factor') != target_factor:
        c['rope_scaling'] = {'rope_type': 'yarn', 'factor': target_factor, 'original_max_position_embeddings': target_orig}
        changed = True
    if changed:
        with open(real, 'w') as f: json.dump(c, f, indent=2)
        print(f'Patched context config in {real}')
    else:
        print(f'Config already set in {real}')
"

# Copy our config into SDPO's config directory so Hydra can find it
cp "$PROJECT_ROOT/configs/sparse_parity_sdpo.yaml" \
   "$SDPO_ROOT/verl/trainer/config/sparse_parity_sdpo.yaml"
cp "$PROJECT_ROOT/configs/sparse_parity_icl.yaml" \
   "$SDPO_ROOT/verl/trainer/config/sparse_parity_icl.yaml"

# SDPO first on PYTHONPATH so `import verl` resolves to SDPO/verl
# Project root second so `import agent_loop` finds our custom loop
export PYTHONPATH="$SDPO_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

# Ensure ~/.local/bin is on PATH for uv
export PATH="$HOME/.local/bin:$PATH"

cd "$SDPO_ROOT"

echo "============================================"
echo "  SDPO Training: Sparse Parity Challenge"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Config: $CONFIG_NAME"
echo "  Data: $DATA_DIR"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Shift past the experiment name if provided
shift 2>/dev/null || true

$PYTHON "$PROJECT_ROOT/training/run_sdpo.py" \
    --config-name "$CONFIG_NAME" \
    vars.dir="$PROJECT_ROOT" \
    vars.ckpt_dir="/data/checkpoints" \
    "$@"
