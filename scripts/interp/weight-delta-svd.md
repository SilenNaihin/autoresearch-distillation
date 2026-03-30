# Weight Delta SVD Analysis: GRPO vs SDPO

Singular value decomposition of weight deltas (W_finetuned - W_base) to characterize how GRPO and SDPO modify Qwen3-14B differently.

## Setup

- **Base model**: Qwen3-14B (HuggingFace, safetensors)
- **GRPO**: step 40, ~100M tokens. Reward-weighted policy gradient, 8 rollouts/prompt, KL coef 1e-5
- **SDPO**: step 480, ~125M tokens. DPO-style contrastive with EMA teacher (tau=0.01), 1 rollout/prompt
- **Analysis**: Full SVD of each 2D weight delta, 282 matrices per model (all attention/MLP projections + embeddings)
- **Script**: `scripts/weight_delta_svd.py`

## Results

| Metric | GRPO | SDPO | Ratio (G/S) |
|---|---|---|---|
| Mean Effective Rank | 1346 | 863 | 1.56x |
| Mean Stable Rank | 13.5 | 9.1 | 1.49x |
| Mean Frobenius Norm | 0.0334 | 0.1379 | 0.24x |
| Mean Relative Change | 0.023% | 0.094% | 0.25x |
| Top-1 SV Variance Fraction | 8.7% | 13.9% | 0.62x |
| Top-5 SV Variance Fraction | 19.1% | 29.7% | 0.64x |
| Top-10 SV Variance Fraction | 26.1% | 39.4% | 0.66x |

### Per Layer Type

| Type | GRPO EffRank | SDPO EffRank | GRPO Top1 | SDPO Top1 |
|---|---|---|---|---|
| q_proj | 1331 | 849 | 7.4% | 12.0% |
| k_proj | 538 | 402 | 5.1% | 8.2% |
| v_proj | 492 | 339 | 12.4% | 18.2% |
| o_proj | 1241 | 733 | 12.6% | 20.4% |
| gate_proj | 1956 | 1294 | 7.3% | 11.6% |
| up_proj | 1821 | 1144 | 8.1% | 13.8% |
| down_proj | 2005 | 1237 | 8.1% | 13.2% |
| lm_head | 2042 | 916 | 7.2% | **27.2%** |

### SDPO Per-Layer Depth

| Layer | EffRank | FroNorm | RelChange |
|---|---|---|---|
| 0 | 933 | 0.098 | 0.072% |
| 10 | 1064 | 0.115 | 0.074% |
| 20 | 835 | 0.164 | 0.110% |
| 30 | 818 | 0.155 | 0.105% |
| 35 | 760 | 0.144 | 0.104% |

## Interpretation

**GRPO makes diffuse, high-rank changes.** The weight deltas occupy ~1346 effective dimensions with no dominant singular value direction. This is the signature of reward-weighted policy gradient averaging over many rollouts: it gently re-weights the full parameter space to make high-reward completions slightly more likely. Small relative changes (0.023%) spread across many dimensions = distribution sharpening.

**SDPO makes concentrated, low-rank changes.** The top 10 singular values capture 39% of variance (vs 26% for GRPO). Effective rank is 36% lower despite 4x larger relative changes. This is the signature of learning new structure: the contrastive DPO loss with EMA teacher forces the model to distinguish good from bad completions by carving out specific low-rank subspaces.

**The lm_head is the smoking gun.** SDPO concentrates 27.2% of variance in a single singular value direction (vs 7.2% for GRPO), meaning it's learning a specific new mapping from hidden states to output tokens rather than broadly re-weighting.

**Depth pattern supports higher-level learning.** SDPO's changes are strongest in middle-to-late layers (20-35) where abstract reasoning happens, with early layers (0-10) relatively untouched.

## Caveats

- SDPO trained 12x more steps (480 vs 40) but comparable total tokens (~125M vs ~100M)
- Different loss functions make direct comparison imperfect
- Low effective rank alone doesn't prove "learning" — forgetting probes and cross-coders needed to confirm

## Data

- `outputs/grpo_delta_results.json` — 282 weight matrices, per-param SVD stats
- `outputs/sdpo_delta_results.json` — 282 weight matrices, per-param SVD stats
