#!/bin/bash
set -euo pipefail

# ============================================================================
# GRPO Baseline Training Launcher for Autoresearch
# Both GPUs on box6 for training (TP=2 vLLM rollout + FSDP2 actor)
# Experiments dispatched to box2 via SSH
# Run on box6 (h100-dev-box-6, 2xH100 NVL)
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
# Both GPUs for training (TP=2); experiments on remote box2
export CUDA_VISIBLE_DEVICES=0,1

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
for base in [os.path.expanduser('~/.cache'), os.environ.get('XDG_CACHE_HOME', '')]:
    if not base: continue
    for p in glob.glob(os.path.join(base, 'huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/config.json')):
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

# Seed /data/cache with repo-committed cache (merged from all prior runs)
mkdir -p /data/cache
if [ -f "$PROJECT_ROOT/cache/all.json" ]; then
    cp -n "$PROJECT_ROOT/cache/all.json" /data/cache/all.json 2>/dev/null || true
    echo "Seeded experiment cache from repo ($(python3 -c "import json; d=json.load(open('/data/cache/all.json')); print(len(d.get('diffs',d)))" 2>/dev/null || echo '?') entries)"
fi

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
echo "  GRPO+MCTS: Autoresearch"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Config: $CONFIG_NAME"
echo "  Training GPUs: 0,1 (TP=2, FSDP2 offload)"
echo "  Experiment GPU: box2-gpu0"
echo "  Rollouts: n=4, batch=4, 16/step"
echo "  KL loss: coef=0.001, PUCT c=1.0"
echo "============================================"

# Shift past the experiment name if provided
shift 2>/dev/null || true

$PYTHON "$PROJECT_ROOT/run_grpo.py" \
    --config-name "$CONFIG_NAME" \
    vars.dir="$PROJECT_ROOT" \
    vars.ckpt_dir="/data/checkpoints" \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.max_position_embeddings=${max_model_len}' \
    "$@"
