#!/bin/bash
set -euo pipefail

# ============================================================================
# GRPO Baseline Training Launcher for Autoresearch
# GPU 0 for training (vLLM rollout + FSDP2 actor), GPU 1 for experiments
# Run on box6 (h100-dev-box-6, 2xH100 NVL)
#
# If TP=1 OOMs, switch to TP=2 with:
#   CUDA_VISIBLE_DEVICES=0,1 and override:
#   trainer.n_gpus_per_node=2
#   actor_rollout_ref.rollout.tensor_model_parallel_size=2
#   actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
#   (and move experiments to remote fleet only)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SDPO_ROOT="$PROJECT_ROOT/SDPO"
PYTHON="${PYTHON:-/data/envs/vllm/bin/python}"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

EXPERIMENT_NAME="${1:-qwen3-14b-grpo}"

export RAY_TMPDIR=/data/tmp
export TMPDIR=/data/tmp
CONFIG_NAME="autoresearch_grpo"
DATA_DIR="$PROJECT_ROOT/data/autoresearch"

# Ensure data is prepared
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "Data not found. Preparing autoresearch dataset..."
    $PYTHON "$PROJECT_ROOT/data/prepare_autoresearch.py"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-autoresearch-grpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EXPERIMENT="$EXPERIMENT_NAME"
export TASK="data/autoresearch"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# Only GPU 0 for training; GPU 1 reserved for experiment dispatch
export CUDA_VISIBLE_DEVICES=0

# Let vLLM auto-select attention backend
unset VLLM_ATTENTION_BACKEND

# Redirect torch inductor cache to /data (root partition is small)
export TORCHINDUCTOR_DIR=/data/torchinductor
export TMPDIR=/data/tmp
export XDG_CACHE_HOME=/data/cache

ulimit -c 0

# Patch HF cached config.json to enforce YaRN settings
$PYTHON -c "
import json, glob, os
target_max_pos = int(os.environ.get('TARGET_MAX_POSITION_EMBEDDINGS', '65536'))
target_orig = 32768
target_factor = float(target_max_pos) / float(target_orig)
for p in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/config.json')):
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
        print(f'Patched context config in {real}')
    else:
        print(f'Context config already set in {real}')
"

# Copy configs into SDPO's config directory so Hydra can find them
cp "$PROJECT_ROOT/configs/autoresearch_grpo.yaml" \
   "$SDPO_ROOT/verl/trainer/config/autoresearch_grpo.yaml"

# SDPO first on PYTHONPATH so `import verl` resolves to SDPO/verl
# Project root second so `import agent_loop_grpo` finds our custom loop
export PYTHONPATH="$SDPO_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

# Ensure ~/.local/bin is on PATH for uv
export PATH="$HOME/.local/bin:$PATH"

cd "$SDPO_ROOT"

echo "============================================"
echo "  GRPO Baseline: Autoresearch"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Config: $CONFIG_NAME"
echo "  Training GPU: 0 (TP=1, FSDP2 offload)"
echo "  Experiment GPU: 1 (box6-gpu1)"
echo "  Rollouts: n=4, batch=4, 16/step"
echo "============================================"

# Shift past the experiment name if provided
shift 2>/dev/null || true

$PYTHON "$PROJECT_ROOT/run_grpo.py" \
    --config-name "$CONFIG_NAME" \
    vars.dir="$PROJECT_ROOT" \
    vars.ckpt_dir="/data/checkpoints" \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.max_position_embeddings=${max_model_len}' \
    "$@"
