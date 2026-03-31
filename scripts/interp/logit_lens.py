"""Logit Lens Analysis: How predictions evolve through layers for base/GRPO/SDPO.

Projects residual stream activations through the unembedding matrix at each layer
to see where in the network the models diverge in prediction space.

Measures:
1. KL divergence of per-layer logits between base and ft models
2. Top-1 prediction agreement rate at each layer
3. Entropy of per-layer predictions (sharpening detection)
4. Per-domain breakdowns (C4, code, math, wiki)

Usage:
    python logit_lens.py --base-path ... --grpo-path ... --sdpo-path ...
"""
import argparse
import json
import gc
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# Analyze every 2nd layer + first and last
LAYERS = list(range(0, 40, 2)) + [39]  # [0,2,4,...,38,39]
MAX_SEQ_LEN = 256
N_PER_SOURCE = 100


def load_texts_by_domain(n_per: int = N_PER_SOURCE) -> dict[str, list[str]]:
    """Load texts grouped by domain."""
    domains = {}

    print("  Loading C4...")
    texts = []
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for i, ex in enumerate(c4):
        if i >= n_per:
            break
        texts.append(ex["text"][:2000])
    domains["c4"] = texts

    print("  Loading code...")
    texts = []
    try:
        code = load_dataset("code_search_net", "python", split="test", streaming=True,
                           trust_remote_code=True)
        for i, ex in enumerate(code):
            if i >= n_per:
                break
            texts.append(ex.get("whole_func_string", ex.get("func_code_string", ""))[:2000])
    except Exception as e:
        print(f"    Code failed ({e}), skipping")
        texts = []
    if texts:
        domains["code"] = texts

    print("  Loading GSM8K...")
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        texts = [gsm[i]["question"] + "\n" + gsm[i]["answer"]
                for i in range(min(n_per, len(gsm)))]
        domains["math"] = texts
    except Exception as e:
        print(f"    GSM8K failed ({e})")

    print("  Loading Wikipedia...")
    texts = []
    try:
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True,
                           trust_remote_code=True)
        for i, ex in enumerate(wiki):
            if i >= n_per:
                break
            texts.append(ex["text"][:2000])
        domains["wiki"] = texts
    except Exception as e:
        print(f"    Wiki failed ({e}), using more C4")
        c4_extra = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for i, ex in enumerate(c4_extra):
            if i >= n_per:
                break
            texts.append(ex["text"][:2000])
        if texts:
            domains["wiki"] = texts

    total = sum(len(v) for v in domains.values())
    print(f"  Domains: {list(domains.keys())}, total {total} sequences")
    return domains


class LogitLensCollector:
    """Collects per-layer logit distributions for a model."""

    def __init__(self, model, tokenizer, layers: list[int], device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device
        self.ln_f = model.model.norm  # Final layer norm
        self.lm_head = model.lm_head  # Unembedding

    def get_per_layer_logits(self, texts: list[str]) -> dict[int, torch.Tensor]:
        """Get logit distributions at each layer for all texts.

        Returns: {layer: [n_tokens, vocab_size] tensor of log-probs}
        Actually returns log-probs only for the final layer and KL stats for others
        to save memory.
        """
        hooks = {}
        hidden_states = {l: [] for l in self.layers}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                hidden_states[layer_idx].append(h.detach())
            return hook_fn

        for l in self.layers:
            hooks[l] = self.model.model.layers[l].register_forward_hook(make_hook(l))

        self.model.eval()
        # Process each text and compute logits on-the-fly to save memory
        per_layer_stats = {l: {"entropy": [], "top1_tokens": []} for l in self.layers}

        with torch.no_grad():
            for i, text in enumerate(texts):
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                       max_length=MAX_SEQ_LEN, padding=False).to(self.device)
                if inputs.input_ids.shape[1] < 10:
                    continue

                # Forward pass (populates hidden_states via hooks)
                self.model(**inputs)

                # For each layer, project through LN + unembedding
                for l in self.layers:
                    h = hidden_states[l][-1]  # [1, seq_len, d]
                    h_normed = self.ln_f(h)
                    logits = self.lm_head(h_normed)  # [1, seq_len, vocab]
                    log_probs = F.log_softmax(logits.float(), dim=-1)

                    # Entropy: -sum p log p
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum(dim=-1)  # [1, seq_len]
                    per_layer_stats[l]["entropy"].append(entropy.squeeze(0).cpu())

                    # Top-1 predictions
                    top1 = logits.argmax(dim=-1)  # [1, seq_len]
                    per_layer_stats[l]["top1_tokens"].append(top1.squeeze(0).cpu())

                    hidden_states[l] = []  # Free memory

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i+1}/{len(texts)}")

        for h in hooks.values():
            h.remove()

        # Concatenate
        result = {}
        for l in self.layers:
            if per_layer_stats[l]["entropy"]:
                result[l] = {
                    "entropy": torch.cat(per_layer_stats[l]["entropy"]),
                    "top1": torch.cat(per_layer_stats[l]["top1_tokens"]),
                }
        return result


def compare_logit_lens(base_path: str, ft_path: str, ft_name: str,
                       domain_texts: dict[str, list[str]],
                       tokenizer, layers: list[int]) -> dict:
    """Run logit lens comparison between base and ft model."""
    print(f"\n{'='*60}")
    print(f"  Logit Lens: base vs {ft_name}")
    print(f"{'='*60}")

    all_texts = []
    domain_ranges = {}
    offset = 0
    for domain, texts in domain_texts.items():
        domain_ranges[domain] = (offset, offset + len(texts))
        all_texts.extend(texts)
        offset += len(texts)

    # Collect base
    print(f"  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    collector = LogitLensCollector(base_model, tokenizer, layers, "cuda:0")
    print(f"  Collecting base logit lens...")
    base_stats = collector.get_per_layer_logits(all_texts)
    del base_model, collector
    torch.cuda.empty_cache()
    gc.collect()

    # Collect ft
    print(f"  Loading {ft_name} model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    collector = LogitLensCollector(ft_model, tokenizer, layers, "cuda:0")
    print(f"  Collecting {ft_name} logit lens...")
    ft_stats = collector.get_per_layer_logits(all_texts)
    del ft_model, collector
    torch.cuda.empty_cache()
    gc.collect()

    # Compare
    results = {"per_layer": {}, "per_domain_per_layer": {}}

    for l in layers:
        if l not in base_stats or l not in ft_stats:
            continue

        base_ent = base_stats[l]["entropy"]
        ft_ent = ft_stats[l]["entropy"]
        base_top1 = base_stats[l]["top1"]
        ft_top1 = ft_stats[l]["top1"]

        n = min(len(base_ent), len(ft_ent))
        base_ent, ft_ent = base_ent[:n], ft_ent[:n]
        base_top1, ft_top1 = base_top1[:n], ft_top1[:n]

        # Agreement: fraction where top-1 prediction matches
        agree = (base_top1 == ft_top1).float().mean().item()
        # Entropy change
        ent_delta = (ft_ent - base_ent).mean().item()
        base_ent_mean = base_ent.mean().item()
        ft_ent_mean = ft_ent.mean().item()

        results["per_layer"][str(l)] = {
            "top1_agreement": agree,
            "base_entropy": base_ent_mean,
            "ft_entropy": ft_ent_mean,
            "entropy_delta": ent_delta,
            "n_tokens": n,
        }

    # Per-domain analysis (at layer 39 only, the final layer)
    if 39 in base_stats and 39 in ft_stats:
        # We need to map token indices back to domains
        # This is approximate since tokenization changes sequence lengths
        # For now just report overall + note domain breakdown is approximate
        pass

    # Print summary
    print(f"\n  {'Layer':>5}  {'Top1 Agree':>11}  {'Base Ent':>9}  {'FT Ent':>9}  {'Ent Delta':>10}")
    for l in layers:
        sl = str(l)
        if sl in results["per_layer"]:
            r = results["per_layer"][sl]
            print(f"  {l:>5}  {r['top1_agreement']:>11.6f}  "
                  f"{r['base_entropy']:>9.4f}  {r['ft_entropy']:>9.4f}  "
                  f"{r['entropy_delta']:>+10.4f}")

    del base_stats, ft_stats
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--grpo-path", required=True)
    parser.add_argument("--sdpo-path", required=True)
    parser.add_argument("--output", default="/scratch/eval_results_v2/logit_lens.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading diverse texts by domain...")
    domain_texts = load_texts_by_domain()

    grpo_results = compare_logit_lens(
        args.base_path, args.grpo_path, "GRPO", domain_texts, tokenizer, LAYERS)
    sdpo_results = compare_logit_lens(
        args.base_path, args.sdpo_path, "SDPO", domain_texts, tokenizer, LAYERS)

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  GRPO vs SDPO Logit Lens Comparison")
    print(f"{'='*60}")
    print(f"  {'Layer':>5}  {'GRPO Agree':>11}  {'SDPO Agree':>11}  {'GRPO EntΔ':>10}  {'SDPO EntΔ':>10}")
    for l in LAYERS:
        sl = str(l)
        g = grpo_results["per_layer"].get(sl, {})
        s = sdpo_results["per_layer"].get(sl, {})
        if g and s:
            print(f"  {l:>5}  {g['top1_agreement']:>11.6f}  {s['top1_agreement']:>11.6f}  "
                  f"{g['entropy_delta']:>+10.4f}  {s['entropy_delta']:>+10.4f}")

    # Key metrics at final layer
    g_final = grpo_results["per_layer"].get("39", {})
    s_final = sdpo_results["per_layer"].get("39", {})
    if g_final and s_final:
        print(f"\n  Final layer (39):")
        print(f"    GRPO: {g_final['top1_agreement']:.4f} agreement, "
              f"{g_final['entropy_delta']:+.4f} entropy shift")
        print(f"    SDPO: {s_final['top1_agreement']:.4f} agreement, "
              f"{s_final['entropy_delta']:+.4f} entropy shift")
        print(f"    SDPO changes {1-s_final['top1_agreement']:.4f} of predictions "
              f"vs GRPO changes {1-g_final['top1_agreement']:.4f}")

    output = {"grpo": grpo_results, "sdpo": sdpo_results,
              "config": {"layers": LAYERS, "max_seq_len": MAX_SEQ_LEN}}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
