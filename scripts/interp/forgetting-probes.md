# Forgetting Probes: Base vs GRPO vs SDPO

lm-eval-harness evaluation of general capabilities to measure catastrophic forgetting from fine-tuning.

## Setup

- **Base**: Qwen3-14B (HuggingFace)
- **GRPO**: step 40, ~100M tokens. Converted from FSDP to HF via `fsdp_to_hf.py`
- **SDPO**: step 480, ~125M tokens. Converted from FSDP to HF via `fsdp_to_hf.py`
- **Eval**: lm-eval-harness v0.4.11, 0-shot, bfloat16, batch_size=auto
- **Hardware**: NVIDIA H100 NVL 96GB

## Results

| Benchmark | Base | GRPO | SDPO | GRPO Delta | SDPO Delta |
|---|---|---|---|---|---|
| MMLU | 77.24% | 77.20% | 77.08% | -0.04% | -0.16% |
| HellaSwag | 61.05% | 60.93% | 60.71% | -0.12% | -0.35% |
| ARC-Challenge | 57.42% | 57.25% | 56.23% | -0.17% | **-1.19%** |
| TruthfulQA MC2 | 58.27% | 58.26% | 58.18% | -0.01% | -0.09% |
| Winogrande | 73.09% | 73.32% | 72.61% | +0.24% | -0.47% |
| **Average** | **65.41%** | **65.39%** | **64.96%** | **-0.02%** | **-0.45%** |

## Interpretation

**GRPO causes essentially zero forgetting** (-0.02% average). All individual benchmarks are within noise of the baseline. This is consistent with the SVD finding that GRPO makes tiny, diffuse changes (0.023% relative weight change) — the perturbation is too small to measurably degrade general capabilities.

**SDPO causes mild but consistent forgetting** (-0.45% average). The degradation is present across all 5 benchmarks, with ARC-Challenge showing the largest drop (-1.19%). This is consistent with SDPO making 4x larger relative weight changes (0.094%) — even though the changes are concentrated in few dimensions, they're large enough to slightly disrupt existing capabilities.

**This partially inverts the initial hypothesis.** We expected GRPO's diffuse changes to cause more forgetting than SDPO's surgical ones. In reality:
- GRPO's changes are so small in magnitude that they don't cause forgetting at all
- SDPO's changes are targeted but 4x larger, causing mild degradation
- The **magnitude** of weight changes matters more than their **rank** for predicting forgetting

**Revised picture:**
- GRPO: tiny diffuse changes -> no forgetting, but also questionable whether it's learning much
- SDPO: larger surgical changes -> mild forgetting, but the targeted nature means it's learning real structure (at the cost of slight general capability loss)

This aligns with a "conservation of change" principle: to learn something new, you have to modify weights enough to encode it, and larger modifications inevitably displace some existing knowledge. GRPO avoids this trade-off by barely changing the model at all.

## Caveats

- 0-shot evaluation (no few-shot prompting) — results may differ with standard few-shot setups
- ARC-Challenge drop for SDPO (-1.19%) is borderline significant (stderr ~1.45%) — could be noise
- Both models were trained on research code generation, which has minimal overlap with these benchmarks
- SDPO trained for more gradient steps (480 vs 40), which contributes to larger forgetting

## Data

- `outputs/forgetting_probes/base_results.json`
- `outputs/forgetting_probes/grpo_results.json`
- `outputs/forgetting_probes/sdpo_results.json`
