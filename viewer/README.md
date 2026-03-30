# GAIA2 Results Viewer

A local web app for browsing GAIA2 benchmark scenarios and comparing model evaluation results across machines.

## What it does

- **Browse the benchmark**: All 800 GAIA2 scenarios (5 categories: search, time, execution, adaptability, ambiguity) with task prompts, expected answers, and available apps
- **View model results**: Per-scenario pass/fail, tool scores, failure classification, and full conversation traces
- **Compare models**: Side-by-side comparison of how different models (e.g., Qwen3-14B vs Claude Haiku 4.5 vs trained checkpoints) handle the same scenario
- **Track training**: Line charts showing pass rate and tool score progression across training steps

## Quick start

```bash
# 1. Install viewer dependencies (one time)
cd viewer && npm install

# 2. Sync model results from GPU boxes and start the viewer
python scripts/viewer_sync.py --serve
# Opens at http://localhost:9000
```

That's it. The benchmark data (800 scenarios) is already committed in the repo. The sync script SSHs into the GPU boxes, discovers ARE baseline runs, converts them to viewer format, and starts Next.js.

Conversation traces for individual scenarios are fetched lazily -- they load over SSH when you click into a scenario detail page.

## How it works

```
   GPU boxes (h100-dev-box-3, 4)              Your Mac
   ┌─────────────────────────────┐          ┌──────────────────────┐
   │ ARE benchmark writes:       │          │                      │
   │   output.jsonl (scores)     │   SSH    │  viewer_sync.py      │
   │   lite/*.json (traces)      │ -------> │  discovers runs      │
   │   benchmark_stats.json      │   scp    │  pulls scores        │
   └─────────────────────────────┘          │  generates summaries │
                                            │                      │
                                            │  Next.js viewer      │
                                            │  serves on :9000     │
                                            │                      │
                                            │  Detail files pulled │
                                            │  lazily on click     │
                                            └──────────────────────┘
```

**Benchmark data** (800 GAIA2 scenarios) is committed at `viewer/public/data/benchmark.json`. To refresh it from HuggingFace: `HF_TOKEN=hf_xxx python scripts/sync_benchmark.py`

**Model run data** lives on GPU boxes. `viewer_sync.py` SSHs into each box listed in `scripts/viewer_config.yaml`, finds ARE benchmark output directories, reads the lightweight `output.jsonl` files (scores + pass/fail), and generates viewer-format summaries locally. No manual conversion needed.

**Detail files** (full conversation traces, ~50KB each) are fetched lazily over SSH when you click into a specific scenario. They're cached locally after the first fetch.

**Full sync**: Use `--full` to pull all detail files upfront (slow but good for offline use): `python scripts/viewer_sync.py --full --serve`

## Adding new boxes

Edit `scripts/viewer_config.yaml`:

```yaml
boxes:
  - ssh_host: h100-dev-box-4
    data_paths: [/data/gaia2_baselines]
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
