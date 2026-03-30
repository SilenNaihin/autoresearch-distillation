# Forgetting Probes: Base vs GRPO vs SDPO

lm-eval-harness evaluation of general capabilities to measure catastrophic forgetting from fine-tuning, plus KL divergence analysis.

## Setup

- **Base**: Qwen3-14B (HuggingFace)
- **GRPO**: step 40, ~100M tokens. Converted from FSDP to HF via `fsdp_to_hf.py`
- **SDPO**: step 480, ~125M tokens. Converted from FSDP to HF via `fsdp_to_hf.py`
- **Eval**: lm-eval-harness v0.4.11, bfloat16, batch_size=auto
- **Hardware**: 2x NVIDIA H100 NVL 96GB

## Results: Standard Few-Shot MC

| Benchmark | Base | GRPO | SDPO | GRPO Delta | SDPO Delta |
|---|---|---|---|---|---|
| MMLU (5-shot) | 78.70% | 78.76% | 78.72% | +0.06% | +0.02% |
| HellaSwag (10-shot) | 61.10% | 61.09% | 60.59% | -0.01% | -0.52% |
| ARC-Challenge (25-shot) | 66.47% | 66.38% | 66.04% | -0.09% | -0.43% |
| Winogrande (5-shot) | 75.06% | 74.27% | 74.43% | -0.79% | -0.63% |
| **MC Average** | | | | **-0.21%** | **-0.39%** |

## Results: Generative (GSM8K 8-shot CoT)

| Benchmark | Base | GRPO | SDPO | GRPO Delta | SDPO Delta |
|---|---|---|---|---|---|
| GSM8K CoT (8-shot) | 89.08% | 89.31% | **90.30%** | +0.23% | **+1.21%** |

## Results: WikiText Perplexity

| Model | Word Perplexity | Delta |
|---|---|---|
| Base | 10.785 | — |
| GRPO | 10.782 | -0.003 |
| SDPO | 10.680 | **-0.105** |

## Results: KL Divergence (200 C4 samples)

| Metric | GRPO | SDPO | Ratio S/G |
|---|---|---|---|
| KL(base\|\|model) | 0.000738 | 0.001717 | 2.3x |
| KL(model\|\|base) | 0.000737 | 0.001742 | 2.4x |
| Model PPL on C4 | 18.36 | 18.23 | ~1.0x |

## Interpretation

**GRPO barely changes the model.** Near-zero forgetting (-0.21% MC avg), near-zero KL divergence (0.0007), near-zero GSM8K change (+0.23%). The KL penalty (1e-5) effectively constrains GRPO to stay very close to the base distribution. This is by design, not because GRPO is inherently better at preserving knowledge.

**SDPO makes real changes with favorable trade-offs.** Despite 2.3x higher KL divergence and mild MC forgetting (-0.39% avg), SDPO:
- **Improves GSM8K by +1.21%** — genuine improvement on math reasoning, a generative task requiring chain-of-thought
- **Improves WikiText perplexity** by -0.105 — the model actually becomes a slightly better language model
- Preserves MMLU almost perfectly (+0.02%) — knowledge retention is intact

**The forgetting is concentrated in reasoning-adjacent benchmarks** (HellaSwag, ARC, Winogrande) while knowledge benchmarks (MMLU) and language modeling (WikiText) are preserved or improved. This suggests SDPO is *reorganizing* reasoning pathways rather than degrading them — trading small MC performance for better generative reasoning.

**Key confounds addressed:**
- **Few-shot matters**: 0-shot ARC showed -1.19% for SDPO; with proper 25-shot it's only -0.43%. The earlier 0-shot results overstated forgetting.
- **KL penalty does the work for GRPO**: GRPO's low forgetting is from the explicit KL constraint (1e-5 coef), not from the training method being inherently gentler.
- **Generative > MC for measuring real capability change**: GSM8K CoT reveals that SDPO actually *gained* capability that MC benchmarks miss. The thinking model's improvements show up in generation, not in next-token likelihood.

## Revised Picture

| Aspect | GRPO | SDPO |
|---|---|---|
| KL from base | 0.0007 (tiny) | 0.0017 (small) |
| MC forgetting | -0.21% (noise) | -0.39% (mild) |
| Generative capability | +0.23% (noise) | **+1.21%** (real gain) |
| WikiText PPL | -0.003 (unchanged) | **-0.105** (improved) |
| Weight change magnitude | 0.023% relative | 0.094% relative |
| Weight change structure | Diffuse (rank 1346) | Surgical (rank 863) |

SDPO achieves a better learning-forgetting trade-off: it makes targeted, low-rank changes large enough to learn new structure (GSM8K +1.21%), while the surgical nature of the changes preserves knowledge (MMLU +0.02%) and even improves language modeling (WikiText -0.1). GRPO's KL constraint keeps it too close to base to learn much of anything.

## Data

- `outputs/forgetting_probes_v2/` — per-benchmark result JSONs (v2, proper few-shot)
- `outputs/forgetting_probes_v2/kl_divergence.json` — KL divergence measurements
- `outputs/forgetting_probes/` — v1 results (0-shot, for reference)
