#!/usr/bin/env bash
# One command to set up and launch the GAIA2 viewer.
# Usage: bash scripts/setup_viewer.sh
set -e

cd "$(dirname "$0")/.."

echo "=== GAIA2 Viewer Setup ==="

# Python deps
echo "Installing Python dependencies..."
pip install -q pyyaml wandb 2>/dev/null || pip3 install -q pyyaml wandb

# Check wandb login
if ! python3 -c "import wandb; wandb.Api()" 2>/dev/null; then
  echo ""
  echo "You need to log in to wandb. Get your API key from https://wandb.ai/authorize"
  wandb login
fi

# Node deps
echo "Installing viewer dependencies..."
cd viewer && npm install --silent && cd ..

# Sync + serve
echo ""
echo "Syncing data and starting viewer..."
python3 scripts/viewer_sync.py --serve
