# GAIA2 Results Viewer

A local web app for browsing GAIA2 benchmark scenarios and comparing model evaluation results across machines.

## What it does

- **Browse the benchmark**: All 800 GAIA2 scenarios (5 categories: search, time, execution, adaptability, ambiguity) with task prompts, expected answers, and available apps
- **View model results**: Per-scenario pass/fail, tool scores, failure classification, and full conversation traces
- **Compare models**: Side-by-side comparison of how different models (e.g., Qwen3-14B vs Claude Haiku 4.5 vs trained checkpoints) handle the same scenario
- **Track training**: Line charts showing pass rate and tool score progression across training steps

## Quick start

```bash
bash scripts/setup_viewer.sh
# Installs deps, logs into wandb if needed, syncs data, opens http://localhost:9000
```

Or manually:

```bash
pip install pyyaml wandb && wandb login
cd viewer && npm install && cd ..
python scripts/viewer_sync.py --serve
```

That's it. The benchmark data (800 scenarios) is already committed in the repo. The sync script pulls eval results from GPU boxes over SSH, fetches training runs from wandb, and starts Next.js.

**Prerequisites**: You need SSH access to the dev boxes (`ssh h100-dev-box-3` should work without a password). Ask Silen if you need to be added. Conversation traces for individual scenarios are fetched lazily over SSH when you click into a scenario detail page.

## How it works

```
   GPU boxes (h100-dev-box-3, 4)              wandb.ai
   ┌─────────────────────────────┐          ┌──────────────────────┐
   │ ARE benchmark writes:       │          │ Training runs:       │
   │   output.jsonl (scores)     │          │   status, tags       │
   │   lite/*.json (traces)      │          │   config, start time │
   └──────────────┬──────────────┘          └──────────┬───────────┘
                  │ SSH/SCP                             │ wandb API
                  ▼                                     ▼
            ┌──────────────────────────────────────────────┐
            │  Your Mac                                    │
            │                                              │
            │  viewer_sync.py    discovers eval runs (SSH) │
            │                    fetches training (wandb)  │
            │                                              │
            │  Next.js viewer    serves on :9000           │
            │                    detail files lazy on click│
            └──────────────────────────────────────────────┘
```

**Benchmark data** (800 GAIA2 scenarios) is committed at `viewer/public/data/benchmark.json`. To refresh it from HuggingFace: `HF_TOKEN=hf_xxx python scripts/sync_benchmark.py`

**Eval runs** live on GPU boxes. `viewer_sync.py` SSHs into each box listed in `scripts/viewer_config.yaml`, finds ARE benchmark output directories, reads `output.jsonl` files (scores + pass/fail), and generates viewer-format summaries locally.

**Training runs** are fetched live from the wandb API every time you open the Training tab. No manual sync needed — if a run is deleted in wandb, it disappears on page refresh.

**Detail files** (full conversation traces, ~50KB each) are fetched lazily over SSH when you click into a specific scenario. Cached locally after the first fetch.

**Full sync**: Use `--full` to pull all detail files upfront (slow but good for offline use): `python scripts/viewer_sync.py --full --serve`

## Adding new boxes

Edit `scripts/viewer_config.yaml`:

```yaml
boxes:
  - ssh_host: h100-dev-box-4
    data_paths: [/data/gaia2_baselines]
wandb_project: silennai-endflow/claas-verl
```

## Project structure

```
scripts/
  viewer_sync.py          # Discover + sync + serve (the main entry point)
  viewer_config.yaml      # Which boxes to sync from
  sync_benchmark.py       # HuggingFace -> viewer/public/data/benchmark.json

viewer/
  src/app/                # Next.js pages (dashboard, benchmark, scenarios, compare, training)
  src/components/         # Shared components (nav, ui primitives, conversation renderer)
  src/lib/                # Types and data fetching
  public/data/
    benchmark.json        # Committed: 800 GAIA2 scenarios
    runs/                 # Gitignored: synced model run data
    index.json            # Gitignored: run manifest
```
