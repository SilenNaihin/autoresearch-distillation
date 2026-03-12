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
EXPERIMENT_NAME="${1:-qwen3-32b-sdpo-lora}"

export RAY_TMPDIR=/data/tmp
export TMPDIR=/data/tmp
export RAY_memory_usage_threshold=0.98
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
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0,1

# Disable VLLM_ATTENTION_BACKEND to let vLLM auto-select
unset VLLM_ATTENTION_BACKEND

# Set ulimit
ulimit -c 0

# Patch HF cached config.json to enforce YaRN settings used by this run.
$PYTHON -c "
import json, glob, os
target_max_pos = int(os.environ.get('TARGET_MAX_POSITION_EMBEDDINGS', '65536'))
target_orig = 32768
target_factor = float(target_max_pos) / float(target_orig)
for p in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/config.json')):
    real = os.path.realpath(p)
    with open(real) as f: c = json.load(f)
    changed = False
    if c.get('max_position_embeddings') != target_max_pos:
        c['max_position_embeddings'] = target_max_pos
        changed = True
    if c.get('rope_scaling') != {'rope_type': 'yarn', 'factor': target_factor, 'original_max_position_embeddings': target_orig}:
        c['rope_scaling'] = {'rope_type': 'yarn', 'factor': target_factor, 'original_max_position_embeddings': target_orig}
        changed = True
    if changed:
        with open(real, 'w') as f: json.dump(c, f, indent=2)
        print(f'Patched context config in {real}: max_position_embeddings={target_max_pos}, rope_factor={target_factor}')
    else:
        print(f'context config already set in {real}')
"

# Copy our config into SDPO's config directory so Hydra can find it
cp "$PROJECT_ROOT/configs/autoresearch_sdpo.yaml" \
   "$SDPO_ROOT/verl/trainer/config/autoresearch_sdpo.yaml"

# SDPO first on PYTHONPATH so `import verl` resolves to SDPO/verl
# Project root second so `import agent_loop` finds our custom loop
export PYTHONPATH="$SDPO_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

# Ensure ~/.local/bin is on PATH for uv
export PATH="$HOME/.local/bin:$PATH"

cd "$SDPO_ROOT"

# Clean up stale processes and free memory before launch
echo "Cleaning up stale processes and freeing memory..."
ray stop --force 2>/dev/null || true
$PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches 2>/dev/null || true
$PYTHON -c "import gc; gc.collect()" 2>/dev/null || true

echo "============================================"
echo "  SDPO Training: Autoresearch"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Config: $CONFIG_NAME"
echo "  Data: $DATA_DIR"
echo "============================================"

# Shift past the experiment name if provided
shift 2>/dev/null || true

$PYTHON "$PROJECT_ROOT/run_sdpo.py" \
    --config-name "$CONFIG_NAME" \
    vars.dir="$PROJECT_ROOT" \
    vars.ckpt_dir="/data/checkpoints" \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.max_position_embeddings=${max_model_len}' \
    "$@"
