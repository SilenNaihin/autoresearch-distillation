# GAIA2 Results Viewer

A local web app for browsing GAIA2 benchmark scenarios and comparing model evaluation results across machines.

## What it does

- **Browse the benchmark**: All 800 GAIA2 scenarios (5 categories: search, time, execution, adaptability, ambiguity) with task prompts, expected answers, and available apps
- **View model results**: Per-scenario pass/fail, tool scores, failure classification, and full conversation traces
- **Compare models**: Side-by-side comparison of how different models (e.g., Qwen3-14B vs Claude Haiku 4.5 vs trained checkpoints) handle the same scenario
- **Track training**: Line charts showing pass rate and tool score progression across training steps

## Quick start

```bash
# 1. Install dependencies (one time)
cd viewer && npm install

# 2. Download benchmark data from HuggingFace (one time)
export HF_TOKEN=hf_xxx
python scripts/sync_benchmark.py

# 3. Start the viewer
cd viewer && npx next dev -p 9000
# Opens at http://localhost:9000
```

The benchmark browser works immediately. To see model run results, sync from your dev boxes:

```bash
# Pull results from all GPU boxes and start the viewer
python scripts/viewer_sync.py --serve
```

## How it works

```
   GPU boxes (h100-dev-box-1, 3, 4)          Your Mac
   ┌─────────────────────────────┐          ┌──────────────────────┐
   │ ARE scaffold runs eval      │          │                      │
   │ Writes traces to            │   SSH    │  viewer_sync.py      │
   │ /data/viewer_data/runs/     │ -------> │  discovers + pulls   │
   │                             │   scp    │  summary.json files  │
   └─────────────────────────────┘          │                      │
                                            │  Next.js viewer      │
   HuggingFace                              │  serves on :9000     │
   ┌─────────────────────────────┐          │                      │
   │ meta-agents-research-       │  HTTPS   │  sync_benchmark.py   │
   │ environments/gaia2          │ -------> │  downloads 800       │
   │ (800 scenarios, 5 cats)     │          │  scenarios to JSON   │
   └─────────────────────────────┘          └──────────────────────┘
```

**Benchmark data** is downloaded once from HuggingFace and saved as a static JSON file. This powers the Benchmark tab — no runs needed.

**Model run data** lives on GPU boxes. `viewer_sync.py` SSHs into each box listed in `scripts/viewer_config.yaml`, finds completed eval runs, and copies the lightweight summary files (~5KB per run) locally. Detail files (full conversations, ~50KB each) are fetched lazily — only when you click into a specific scenario.

**During training**, the scaffold auto-exports viewer data after each eval checkpoint. When you sync, new training steps appear in the Training tab as additional data points.

## Project structure

```
scripts/
  sync_benchmark.py      # HuggingFace -> viewer/public/data/benchmark.json
  viewer_sync.py          # SSH to GPU boxes -> viewer/public/data/runs/
  viewer_config.yaml      # Which boxes to sync from
  trace_metrics.py        # Shared metric computation (tool score, failure classification)
  export_viewer_data.py   # ARE traces -> viewer JSON (runs on GPU boxes)

viewer/
  src/app/                # Next.js pages (dashboard, benchmark, scenarios, compare, training)
  src/components/         # Shared components (nav, ui primitives, conversation renderer)
  src/lib/                # Types and data fetching utilities
  public/data/            # .gitignored — synced data lives here
```
