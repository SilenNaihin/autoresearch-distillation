# Cross-Coder Analysis: GRPO vs SDPO Feature Decomposition

## Motivation

Weight delta SVD showed GRPO makes diffuse changes (eff_rank 1346) while SDPO makes surgical ones (eff_rank 863). Forgetting probes showed SDPO actually *gains* generative capability (+1.21% GSM8K) while mildly trading off MC benchmarks (-0.39%). The question remaining: **what is SDPO learning?**

Weight-space analysis (SVD) tells us *how much* and *where* weights changed. Activation-space analysis (cross-coders) tells us *what computational features* changed. This is the difference between "these neurons moved" and "this model learned to detect recursive patterns."

## Research Questions

1. Does SDPO create genuinely new computational features, or just modify existing ones?
2. Does GRPO mostly preserve/sharpen the base model's feature set?
3. Are SDPO's new features concentrated in the layers where SVD showed the largest changes (20-35)?
4. What do the new features correspond to semantically?

## Hypotheses

**H1: SDPO produces more novel features than GRPO.**
- Metric: Count of features where `||W_dec_ft[i]|| / ||W_dec_base[i]|| > 2` (ft-dominant features)
- Expected: SDPO has 2-3x more ft-dominant features than GRPO
- Rationale: SDPO's low-rank, high-magnitude changes in weight space should manifest as concentrated new features in activation space

**H2: GRPO's feature set is nearly identical to base.**
- Metric: Fraction of features where `|log(ft_norm / base_norm)| < 0.5` (shared features)
- Expected: >90% of GRPO features are shared, vs ~70-80% for SDPO
- Rationale: GRPO's tiny weight changes (0.023% relative) and near-zero KL (0.0007) mean it barely modifies the model's internal representations

**H3: Novel features concentrate in layers 20-35 for SDPO.**
- Metric: Novel feature count per layer
- Expected: 3-5x more novel features at layer 30 vs layer 10 for SDPO; flat distribution for GRPO
- Rationale: SVD showed SDPO's relative changes peak in mid-to-late layers (0.11% at layer 20 vs 0.07% at layer 0)

**H4: SDPO's novel features are semantically meaningful.**
- Metric: Qualitative inspection of top-activating examples for ft-dominant features
- Expected: Features cluster around code structure, mathematical reasoning, or research-specific patterns
- Rationale: SDPO gained GSM8K capability; these features should correspond to chain-of-thought / reasoning circuits

**H5: Neither model loses significant features.**
- Metric: Count of features where `||W_dec_base[i]|| / ||W_dec_ft[i]|| > 3` (base-dominant/lost features)
- Expected: <5% of features are base-dominant for either model
- Rationale: Both models showed minimal forgetting on knowledge benchmarks (MMLU preserved)

## Method

### Phase 1: Activation Similarity (quick, ~30 min)

Before investing in full SAE training, measure activation-space similarity directly:

1. **CKA (Centered Kernel Alignment)** between base and ft activations at each layer
   - CKA ≈ 1.0 means nearly identical representations
   - CKA < 0.95 means meaningful representational shift
   - Compare GRPO-CKA vs SDPO-CKA profiles across layers

2. **Cosine similarity distributions** of activation vectors (base vs ft) at each layer
   - Median cosine sim near 1.0 = representations preserved
   - Long left tail = some inputs processed very differently

This gives us the lay of the land before the expensive SAE step.

### Phase 2: Cross-Coder Training

**Architecture:**

```
Cross-Coder(x_base, x_ft):
  # Separate encoders, shared latent space
  h_base = W_enc_base @ (x_base - b_pre)
  h_ft   = W_enc_ft   @ (x_ft   - b_pre)

  # Combined encoding with TopK sparsity
  z = TopK(ReLU(h_base + h_ft), k=32)

  # Separate decoders
  x̂_base = W_dec_base @ z + b_pre
  x̂_ft   = W_dec_ft   @ z + b_pre

  # Loss
  L = ||x_base - x̂_base||² + ||x_ft - x̂_ft||² + λ * ||z||₁
```

- Input dimension: d = 5120 (Qwen3-14B hidden size)
- Dictionary size: n = 5120 * 8 = 40960 features (8x expansion)
- TopK sparsity: k = 32 active features per token
- L1 penalty: λ = 1e-3 (tune on validation)
- Optimizer: Adam, lr=3e-4, cosine decay
- Training tokens: 500K positions (from ~2K sequences × 256 tokens)

**Why this architecture:**
- Separate encoders allow the model to extract features from either model independently
- Shared latent z forces alignment: a feature must be "the same thing" in both models
- Separate decoders let us measure how much each feature contributes to reconstructing each model
- TopK sparsity is more stable than pure L1 for cross-coders

**Layers to analyze:** 10, 20, 30, 38 (early, mid, peak-change, near-final)

**Models to compare:**
- Cross-coder A: base ↔ GRPO (4 layers × 1 SAE = 4 cross-coders)
- Cross-coder B: base ↔ SDPO (4 layers × 1 SAE = 4 cross-coders)
- Total: 8 cross-coders

### Phase 3: Feature Classification & Analysis

Post-training, classify each learned feature:

```
For feature i:
  base_norm = ||W_dec_base[i, :]||
  ft_norm   = ||W_dec_ft[i, :]||
  ratio     = log2(ft_norm / base_norm)

  if |ratio| < 0.5:    → SHARED (present in both)
  if ratio > 1.0:      → FT-DOMINANT (new in fine-tuned)
  if ratio < -1.0:     → BASE-DOMINANT (lost in fine-tuning)
  else:                 → MODIFIED (shifted)
```

Report:
1. Feature type distribution (shared/new/lost/modified) per model × layer
2. Variance explained by each feature type
3. Top-20 activating examples for the most ft-dominant features (qualitative)
4. Activation frequency distributions per feature type

### Data Collection

**Prompt sources** (diverse to capture different capability domains):

| Source | N seqs | Purpose |
|---|---|---|
| C4 validation | 500 | General English text |
| The Stack (Python) | 500 | Code understanding |
| GSM8K train prompts | 300 | Math reasoning (where SDPO improved) |
| arXiv abstracts | 200 | Research domain |
| **Total** | 1500 | ~384K token positions at 256 tok/seq |

Run each sequence through base, GRPO, and SDPO. Save residual stream activations (post-LayerNorm, pre-next-layer) at layers 10, 20, 30, 38.

Storage: 1500 seqs × 256 tokens × 5120 dims × 4 layers × 3 models × 2 bytes (bf16) ≈ **47 GB**. Fits in memory, or can stream from disk.

## Compute Budget

| Step | GPU hours | Notes |
|---|---|---|
| Activation collection | ~1 hr | 3 models × 1500 seqs, batch inference on 2 GPUs |
| Phase 1 (CKA/cosine) | ~10 min | Lightweight computation on saved activations |
| Phase 2 (8 cross-coders) | ~4 hr | 500K tokens × ~50 epochs each, on GPU |
| Phase 3 (analysis) | ~30 min | Feature classification and visualization |
| **Total** | **~6 hr** | Well within our 2x H100 budget |

## Expected Outcomes

### If hypotheses are confirmed:
SDPO creates a meaningful set of new features (especially in layers 20-35) that correspond to reasoning/code capabilities, while GRPO's feature set is indistinguishable from base. This would definitively establish that SDPO performs "genuine skill acquisition" — it builds new computational circuits — while GRPO merely adjusts the weighting of existing circuits.

### If hypotheses are refuted:
If SDPO doesn't show more novel features than GRPO, the SVD differences might reflect optimizer dynamics rather than meaningful learning. If GRPO shows surprisingly many novel features despite tiny weight changes, it would suggest that small distributed changes can create emergent computational features — a more interesting finding.

### Null result:
If both models show similar feature profiles to base, it would suggest that neither method fundamentally changes what the model computes — they only change output probabilities. This would imply the training task (research code gen) doesn't require new circuits, just better calibration.

## Decision Points

- After Phase 1 (CKA): If CKA > 0.99 for both GRPO and SDPO at all layers, the representational shift may be too small for cross-coders to decompose meaningfully. In this case, we'd switch to a **logit lens** analysis instead (how output predictions change at each layer).
- After Phase 2: If reconstruction loss is >50% of input variance for any cross-coder, increase dictionary size or training time before proceeding to analysis.
- Feature classification thresholds (0.5, 1.0) are heuristic — will calibrate based on the distribution of log-ratios observed.

---

## Phase 1 Results (Executed)

CKA and cosine similarity computed on 600 sequences (C4, code, GSM8K, Wikipedia), 105K token positions, layers 10/20/30/38.

| Layer | GRPO CKA | SDPO CKA | GRPO cos mean | SDPO cos mean | SDPO frac<0.95 |
|---|---|---|---|---|---|
| 10 | 0.999987 | 0.999948 | 0.999969 | 0.999919 | 0.0000 |
| 20 | 0.999986 | 0.999979 | 0.999917 | 0.999817 | 0.0000 |
| 30 | 0.999964 | 0.999966 | 0.999843 | 0.999627 | 0.0004 |
| 38 | 0.999927 | 0.999614 | 0.999829 | 0.999549 | 0.0004 |

**Outcome: CKA > 0.999 everywhere.** Both models' activations are nearly identical to base. SDPO diverges slightly more (especially at layer 38), consistent with SVD findings, but the absolute shift is too small for cross-coders to meaningfully decompose.

**Decision: Pivot to logit lens analysis.** The weight changes modify what the model *outputs* (demonstrated by GSM8K +1.21%, WikiText PPL -0.1) without meaningfully changing internal activation geometry. Logit lens will show how output predictions evolve through the layers and where GRPO vs SDPO diverge from base in prediction space.
