"""Phase 1: Activation similarity analysis (CKA + cosine similarity).

Measures representational similarity between base and fine-tuned models
at multiple layers to determine if cross-coder training is warranted.

Usage:
    python activation_similarity.py \
        --base-path /path/to/base \
        --grpo-path /path/to/grpo \
        --sdpo-path /path/to/sdpo \
        --output /path/to/output.json
"""
import argparse
import json
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


LAYERS = [10, 20, 30, 38]
MAX_SEQ_LEN = 256
N_SEQS_PER_SOURCE = 150  # 150 × 4 sources = 600 seqs × 256 tok = ~154K positions


def collect_activations(model, tokenizer, texts: list[str], layers: list[int],
                        device: str, max_len: int = MAX_SEQ_LEN) -> dict[int, torch.Tensor]:
    """Collect residual stream activations at specified layers."""
    hooks = {}
    activations = {l: [] for l in layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is (hidden_states, ...) — take hidden states
            h = output[0] if isinstance(output, tuple) else output
            activations[layer_idx].append(h.detach().cpu())
        return hook_fn

    # Register hooks on decoder layers
    for l in layers:
        hooks[l] = model.model.layers[l].register_forward_hook(make_hook(l))

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_len, padding=False).to(device)
            if inputs.input_ids.shape[1] < 10:
                continue
            model(**inputs)
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(texts)} sequences")

    # Remove hooks
    for h in hooks.values():
        h.remove()

    # Concatenate: [n_tokens, d_model]
    result = {}
    for l in layers:
        if activations[l]:
            result[l] = torch.cat(activations[l], dim=1).squeeze(0).float()
            print(f"    Layer {l}: {result[l].shape}")
        activations[l] = []  # Free memory
    return result


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two activation matrices.

    X, Y: [n_samples, d] — activations from two models at the same positions.
    Returns CKA score in [0, 1].
    """
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices (use dot products for efficiency)
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    XtX = X.T @ X  # [d, d]
    YtY = Y.T @ Y
    YtX = Y.T @ X

    cka = (YtX ** 2).sum() / (torch.sqrt((XtX ** 2).sum()) * torch.sqrt((YtY ** 2).sum()))
    return cka.item()


def batched_linear_cka(X: torch.Tensor, Y: torch.Tensor, batch_size: int = 10000) -> float:
    """Memory-efficient CKA for large activation matrices."""
    n = X.shape[0]
    if n <= batch_size:
        return linear_cka(X, Y)

    # Subsample for CKA (it's a population statistic)
    idx = torch.randperm(n)[:batch_size]
    return linear_cka(X[idx], Y[idx])


def cosine_similarity_stats(X: torch.Tensor, Y: torch.Tensor,
                            batch_size: int = 5000) -> dict:
    """Compute per-token cosine similarity between paired activations."""
    n = X.shape[0]
    all_cos = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cos = F.cosine_similarity(X[start:end], Y[start:end], dim=-1)
        all_cos.append(cos)

    cos = torch.cat(all_cos)
    return {
        "mean": cos.mean().item(),
        "std": cos.std().item(),
        "median": cos.median().item(),
        "p5": cos.quantile(0.05).item(),
        "p25": cos.quantile(0.25).item(),
        "p75": cos.quantile(0.75).item(),
        "p95": cos.quantile(0.95).item(),
        "min": cos.min().item(),
        "frac_below_0.95": (cos < 0.95).float().mean().item(),
        "frac_below_0.90": (cos < 0.90).float().mean().item(),
        "frac_below_0.80": (cos < 0.80).float().mean().item(),
    }


def load_diverse_texts(tokenizer, n_per_source: int = N_SEQS_PER_SOURCE) -> list[str]:
    """Load diverse text from multiple sources."""
    texts = []

    # C4 validation (general English)
    print("  Loading C4...")
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for i, ex in enumerate(c4):
        if i >= n_per_source:
            break
        texts.append(ex["text"][:2000])

    # Code (Python from The Stack or code_search_net)
    print("  Loading code...")
    try:
        code = load_dataset("code_search_net", "python", split="test", streaming=True,
                           trust_remote_code=True)
        for i, ex in enumerate(code):
            if i >= n_per_source:
                break
            texts.append(ex.get("whole_func_string", ex.get("func_code_string", ""))[:2000])
    except Exception as e:
        print(f"  Code dataset failed ({e}), using C4 code-like samples instead")
        # Fallback: more C4
        for i, ex in enumerate(c4):
            if i >= n_per_source:
                break
            texts.append(ex["text"][:2000])

    # Math (GSM8K train)
    print("  Loading GSM8K...")
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for i in range(min(n_per_source, len(gsm))):
            texts.append(gsm[i]["question"] + "\n" + gsm[i]["answer"])
    except Exception as e:
        print(f"  GSM8K failed ({e})")

    # Wikipedia (general knowledge)
    print("  Loading Wikipedia...")
    try:
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True,
                           trust_remote_code=True)
        for i, ex in enumerate(wiki):
            if i >= n_per_source:
                break
            texts.append(ex["text"][:2000])
    except Exception as e:
        print(f"  Wikipedia failed ({e}), using more C4")
        c4_2 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        count = 0
        for ex in c4_2:
            if count >= n_per_source:
                break
            count += 1
        for i, ex in enumerate(c4_2):
            if i >= n_per_source:
                break
            texts.append(ex["text"][:2000])

    print(f"  Total sequences: {len(texts)}")
    return texts


def run_comparison(base_path: str, ft_path: str, ft_name: str,
                   texts: list[str], tokenizer, layers: list[int]) -> dict:
    """Run CKA + cosine similarity comparison for one model pair."""
    print(f"\n{'='*60}")
    print(f"  Comparing base vs {ft_name}")
    print(f"{'='*60}")

    # Load base on GPU 0
    print(f"  Loading base model on GPU 0...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )

    print(f"  Collecting base activations...")
    base_acts = collect_activations(base_model, tokenizer, texts, layers, "cuda:0")
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Load ft on GPU 0 (reuse same GPU to avoid OOM)
    print(f"  Loading {ft_name} model on GPU 0...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )

    print(f"  Collecting {ft_name} activations...")
    ft_acts = collect_activations(ft_model, tokenizer, texts, layers, "cuda:0")
    del ft_model
    torch.cuda.empty_cache()
    gc.collect()

    # Compute metrics per layer
    results = {}
    for l in layers:
        if l not in base_acts or l not in ft_acts:
            continue

        X, Y = base_acts[l], ft_acts[l]
        # Align lengths (should match but safety check)
        min_n = min(X.shape[0], Y.shape[0])
        X, Y = X[:min_n], Y[:min_n]

        print(f"\n  Layer {l} ({min_n} token positions):")

        cka = batched_linear_cka(X, Y)
        print(f"    CKA: {cka:.6f}")

        cos_stats = cosine_similarity_stats(X, Y)
        print(f"    Cosine sim: mean={cos_stats['mean']:.6f} median={cos_stats['median']:.6f} "
              f"p5={cos_stats['p5']:.6f} min={cos_stats['min']:.6f}")
        print(f"    Frac below 0.95: {cos_stats['frac_below_0.95']:.4f}  "
              f"below 0.90: {cos_stats['frac_below_0.90']:.4f}  "
              f"below 0.80: {cos_stats['frac_below_0.80']:.4f}")

        # Activation norm comparison
        base_norms = X.norm(dim=-1)
        ft_norms = Y.norm(dim=-1)
        norm_ratio = (ft_norms / (base_norms + 1e-8))

        results[str(l)] = {
            "cka": cka,
            "cosine": cos_stats,
            "n_tokens": min_n,
            "base_act_norm_mean": base_norms.mean().item(),
            "ft_act_norm_mean": ft_norms.mean().item(),
            "norm_ratio_mean": norm_ratio.mean().item(),
            "norm_ratio_std": norm_ratio.std().item(),
        }

    del base_acts, ft_acts
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--grpo-path", required=True)
    parser.add_argument("--sdpo-path", required=True)
    parser.add_argument("--output", default="/scratch/eval_results_v2/activation_similarity.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading diverse text sources...")
    texts = load_diverse_texts(tokenizer)

    # Run comparisons
    grpo_results = run_comparison(args.base_path, args.grpo_path, "GRPO",
                                  texts, tokenizer, LAYERS)
    sdpo_results = run_comparison(args.base_path, args.sdpo_path, "SDPO",
                                  texts, tokenizer, LAYERS)

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  Summary: CKA across layers")
    print(f"{'='*60}")
    print(f"  {'Layer':>5}  {'GRPO CKA':>10}  {'SDPO CKA':>10}  {'Delta':>10}")
    for l in LAYERS:
        sl = str(l)
        g_cka = grpo_results.get(sl, {}).get("cka", 0)
        s_cka = sdpo_results.get(sl, {}).get("cka", 0)
        print(f"  {l:>5}  {g_cka:>10.6f}  {s_cka:>10.6f}  {g_cka - s_cka:>+10.6f}")

    print(f"\n  Summary: Cosine similarity (mean) across layers")
    print(f"  {'Layer':>5}  {'GRPO cos':>10}  {'SDPO cos':>10}  {'GRPO <0.95':>12}  {'SDPO <0.95':>12}")
    for l in LAYERS:
        sl = str(l)
        g_cos = grpo_results.get(sl, {}).get("cosine", {}).get("mean", 0)
        s_cos = sdpo_results.get(sl, {}).get("cosine", {}).get("mean", 0)
        g_f95 = grpo_results.get(sl, {}).get("cosine", {}).get("frac_below_0.95", 0)
        s_f95 = sdpo_results.get(sl, {}).get("cosine", {}).get("frac_below_0.95", 0)
        print(f"  {l:>5}  {g_cos:>10.6f}  {s_cos:>10.6f}  {g_f95:>12.4f}  {s_f95:>12.4f}")

    # Decision
    min_sdpo_cka = min(sdpo_results.get(str(l), {}).get("cka", 1.0) for l in LAYERS)
    max_sdpo_frac = max(sdpo_results.get(str(l), {}).get("cosine", {}).get("frac_below_0.95", 0)
                        for l in LAYERS)

    print(f"\n  Decision point:")
    print(f"    Min SDPO CKA: {min_sdpo_cka:.6f}")
    print(f"    Max SDPO frac<0.95: {max_sdpo_frac:.4f}")
    if min_sdpo_cka > 0.99 and max_sdpo_frac < 0.01:
        print(f"    → Representations very similar. Cross-coders may not find much.")
        print(f"      Consider logit lens analysis instead.")
    elif min_sdpo_cka > 0.95:
        print(f"    → Moderate representational shift. Cross-coders should find some features.")
        print(f"      Proceed with Phase 2.")
    else:
        print(f"    → Substantial representational shift. Cross-coders highly warranted.")
        print(f"      Proceed with Phase 2.")

    # Save
    output = {
        "grpo": grpo_results,
        "sdpo": sdpo_results,
        "config": {
            "layers": LAYERS,
            "n_texts": len(texts),
            "max_seq_len": MAX_SEQ_LEN,
        }
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
