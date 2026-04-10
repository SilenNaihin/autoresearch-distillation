# Sparse Parity Challenge: SDPO Hill Climbing

**Branch:** `feat/sparse-parity-challenge`
**Task config:** [`tasks/sparse_parity.yaml`](../tasks/sparse_parity.yaml)

## Problem

The [sparse parity challenge](https://github.com/cybertronai/sparse-parity-challenge) asks: given
random {-1, +1} inputs and labels `y = product of k secret bits`, identify which k bits are secret.

Evaluation parameters (matching the challenge leaderboard):
- **n_bits = 20**, **k_sparse = 3**
- **n_train = 200** samples given to the solver
- **n_test = 200** held-out samples for accuracy
- **seeds = [42, 123, 456]** — accuracy averaged across all three

### What we're optimizing

| Metric | Requirement | Description |
|--------|-------------|-------------|
| **Accuracy** | >= 95% (hard gate) | Must correctly identify the secret bits |
| **DMC** | Lower is better (primary) | Data Movement Complexity: `sum(sqrt(stack_distance))` for every array read, measured via LRU stack tracking. Captures algorithmic cache/energy efficiency, not FLOPs. |
| **Time** | < 60s | Wall clock |

**DMC is the ranking metric.** It penalizes algorithms that thrash memory — reading data that hasn't been
touched recently costs `sqrt(depth)` per element. This rewards sequential access patterns, small working
sets, and algorithms that reuse recently-written data.

### Current leaderboard (as of 2026-04-07)

| Rank | Method | DMC | Time | Family |
|------|--------|-----|------|--------|
| 1 | Sequential Elimination | 19,153 | 0.009s | Iterative |
| 2 | GF(2) RREF Minimal (22 rows) | 45,904 | 0.0002s | Algebraic |
| 3 | GF(2) Ultra-Minimal (n+1, retry) | 286,011 | 0.004s | Algebraic |
| 4 | GF(2) Minimal (2n samples) | 601,157 | 0.009s | Algebraic |
| 5 | SMT Backtracking | 1,625,260 | 0.006s | Search |
| 6 | GF(2) Gaussian Elimination | 11,164,685 | 0.101s | Algebraic |
| 7 | GF(2) proposed candidate | 24,397,704 | 0.002s | Algebraic |

The gap between #1 (19K) and #6 (11M) is ~580x — algorithmic choice dominates. The best solutions
minimize sample usage (fewer rows = less data to move) and use cache-friendly access patterns.

### Why this is a good fit for SDPO

- **Eval is fast** (~seconds, CPU-only numpy) — tight feedback loop, many iterations per hour
- **Solution space is structured** — small edits to a numpy-only Python function
- **Reward signal is rich** — DMC is continuous, accuracy provides a hard gate
- **Self-distillation feedback is informative** — "your solution had DMC 50K but only 92% accuracy because..."
- **Multiple algorithmic families** to explore — PUCT can track which branches are promising
- **No GPU needed for eval** — all A100 compute goes to model serving/training

## Experimental Design

### Approach

Use the autoresearch-distillation SDPO pipeline to iteratively improve `solve.py`. The model
receives the current best solution, edits it via bash, and the evaluation harness measures accuracy
and DMC. Reward = `max(0, baseline_dmc - new_dmc)`, gated on accuracy >= 95%.

The PUCT reuse buffer is seeded with 7 different algorithmic approaches, giving the model diverse
starting points to explore from.

### Seed Approaches (7 families)

Each seed is a complete `solve.py` implementing a different algorithmic strategy. These are pre-evaluated
and loaded into the reuse buffer before training begins.

| # | Approach | Family | Description |
|---|----------|--------|-------------|
| 1 | Brute-force triple scan | Search | Check all C(n,k) combinations with minimal samples, early termination |
| 2 | GF(2) Gaussian elimination | Algebraic | Map to binary field, build parity matrix, RREF |
| 3 | Sequential GF(2) elimination | Iterative | Remove bits one at a time via GF(2) rank/consistency testing |
| 4 | Random k-subset sampling | Randomized | Sample random k-subsets, check Walsh coefficient, fast on average |
| 5 | Brute-force with early exit | Search | Like #1 but checks exact sample agreement, different access pattern |
| 6 | Build-up exact match | Constructive | Triple nested loop checking exact prod == y on all samples |
| 7 | Random restart hill climbing | Optimization | Start random, swap bits greedily until exact match, restart if stuck |

**Important note on sparse parity:** For k-sparse parity in {-1,+1} encoding, ALL lower-order
correlations are zero — E[x_i * y] = 0, E[x_i * x_j * y] = 0, etc. Only the exact k-th order
interaction E[prod(x_S) * y] is non-zero when S = secret set. This means statistical screening
approaches (pairwise interactions, mutual information) fundamentally cannot work. Every successful
approach must either: (a) check k-th order interactions directly, or (b) use algebraic methods
(GF(2) linear algebra). This is a core property of the problem.

The 7 families span search, algebraic, iterative, randomized, and optimization paradigms.
Each has fundamentally different DMC profiles (sample usage, access patterns, working set size).

### Runs

We run 4 classes of experiments, all on `a100-backup-1` (Azure, single A100 80GB):

#### Baselines (ICL — no weight updates)

ICL baselines use the full SDPO infrastructure (PUCT sampling, feedback, multi-turn agent loop)
but with **LR = 0** — model weights never change. This isolates the contribution of search
(PUCT + feedback) from learning (gradient updates).

| Run | Model | Buffer | Purpose |
|-----|-------|--------|---------|
| **B1** | Qwen3-14B (vLLM, local) | Shared (all 7 seeds) | ICL search baseline |
| **B2** | Opus 4.6 (Bedrock API) | Shared (all 7 seeds) | Frontier model ICL baseline |

B2 runs via a standalone script calling Bedrock — no A100 compute needed. Can run first/in parallel
with setup.

#### SDPO (with weight updates)

| Run | Model | Buffer | Purpose |
|-----|-------|--------|---------|
| **S1** | Qwen3-14B SDPO | Shared (all 7 seeds) | Cross-pollination across families |
| **S2** | Qwen3-14B SDPO | Isolated: correlation | |
| **S3** | Qwen3-14B SDPO | Isolated: GF(2) | |
| **S4** | Qwen3-14B SDPO | Isolated: sequential elimination | |
| **S5** | Qwen3-14B SDPO | Isolated: Walsh-Hadamard | |
| **S6** | Qwen3-14B SDPO | Isolated: brute-force | |
| **S7** | Qwen3-14B SDPO | Isolated: recursive bisection | |
| **S8** | Qwen3-14B SDPO | Isolated: coordinate descent | |

#### Key ablation: Shared vs Isolated buffers

**S1** (shared) vs **S2-S8** (isolated) tests whether cross-pollination across algorithmic families
helps or hurts. Two hypotheses:

- **Shared wins:** The model learns general DMC-reduction principles (minimize sample usage, sequential
  access, reuse recently written data) that transfer across families. Seeing GF(2) techniques might
  inspire improvements to the sequential elimination approach.
- **Isolated wins:** Focused exploration goes deeper into each family's local optima. The shared buffer
  dilutes exploration budget across too many branches, and techniques from one family create noise
  when applied to another.

We expect the answer depends on training budget. Short runs favor isolated (faster convergence in
each niche), long runs may favor shared (cross-family insight transfer).

### Metrics & Logging (wandb)

Each run logs to wandb under project `sparse-parity-sdpo`:

- `env_dmc` — DMC of each evaluated solution
- `env_accuracy` — accuracy across 3 seeds
- `env_time` — wall clock time
- `env_novel` — whether the solution was novel (not cached/duplicate)
- `reward` — max(0, baseline_dmc - dmc) if accuracy >= 0.95, else 0
- `best_dmc` — running best DMC in the reuse buffer
- `buffer_size` — number of states in reuse buffer
- `seed_family` — which algorithmic family the parent state belongs to

### Run order

Since we have one A100, runs are sequential:

1. **B2 (Opus ICL)** — runs immediately, no GPU needed, establishes frontier model baseline
2. **B1 (Qwen ICL)** — first GPU run, validates infrastructure, establishes ICL search ceiling
3. **S1 (SDPO shared)** — main experiment
4. **S2-S8 (SDPO isolated)** — ablation runs (can be prioritized based on S1 findings)

## Architecture

```
a100-backup-1 (Azure, Standard_NC24ads_A100_v4)
  |
  |-- vLLM (Qwen3-14B, fp16, ~28GB) -- GPU
  |-- SDPO training (FSDP2) ---------- GPU
  |
  |-- Eval dispatch (SSH to self) ----- CPU
  |     |-- evaluate.py (numpy + TrackedArray)
  |     |-- ~2-5 seconds per evaluation
  |
  |-- Reuse buffer (JSON, /data/) ----- disk
  |-- Experiment cache (JSON, /data/) - disk
```

Eval dispatches via SSH to the same box (CPU-only, no GPU contention). Training and inference
share the GPU in wake/sleep mode or split across tensor parallel.

## Files

```
sparse_parity/
  README.md           # this file
  solve.py            # target file the model edits (seeded with simplest approach)
  evaluate.py         # evaluation harness: runs solve(), measures accuracy + DMC
  seeds/              # 7 seed solutions, one per algorithmic family
    01_correlation.py
    02_gf2_gaussian.py
    03_sequential_elimination.py
    04_walsh_hadamard.py
    05_brute_force.py
    06_recursive_bisection.py
    07_coordinate_descent.py
  seed_buffer.py      # pre-evaluates seeds and populates reuse buffer

tasks/
  sparse_parity.yaml  # task config (workspace, scoring, prompts, fleet)

configs/
  sparse_parity_sdpo.yaml           # SDPO training config (Qwen3-14B, 1xA100)
  sparse_parity_icl.yaml            # ICL baseline (LR=0, same infrastructure)
  sparse_parity_agent_loops.yaml    # agent loop config pointing to task
  sparse_parity_bash_tool.yaml      # bash tool config pointing to task

baselines/
  opus_icl.py         # standalone Opus 4.6 ICL via Bedrock API
```

## Setup

```bash
# 1. SSH to a100-backup-1
ssh a100-backup-1

# 2. Clone and setup
git clone <repo> && cd autoresearch-distillation
git checkout feat/sparse-parity-challenge
pip install numpy  # only dependency for eval

# 3. Install sparse-parity-challenge tracker (for DMC measurement)
pip install git+https://github.com/cybertronai/sparse-parity-challenge.git

# 4. Verify evaluation works
cd sparse_parity && python evaluate.py  # should print accuracy + DMC

# 5. Pre-populate reuse buffer with seeds
python seed_buffer.py

# 6. Run Opus baseline (no GPU needed, can run from anywhere)
python baselines/opus_icl.py

# 7. Run Qwen ICL baseline
bash scripts/run_training.sh configs/sparse_parity_icl.yaml

# 8. Run SDPO
bash scripts/run_training.sh configs/sparse_parity_sdpo.yaml
```

## Open Questions

- **n_bits scaling:** The challenge uses n_bits=20. Should we also test n_bits=50 or n_bits=100 to
  see if SDPO discovers approaches that scale better?
- **Sample budget:** The challenge gives 200 training samples. Some seeds use far fewer (n+1 = 21).
  Should we reward using fewer samples as a secondary metric?
- **Submission:** If we beat the leaderboard, we can submit via GitHub issue to the challenge repo.
