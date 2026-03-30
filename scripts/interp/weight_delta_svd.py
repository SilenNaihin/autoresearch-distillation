"""Weight Delta SVD Analysis: GRPO vs SDPO vs Base

Loads FSDP-sharded checkpoints for GRPO and SDPO, computes weight deltas
against the base Qwen3-14B model, and compares effective rank, Frobenius norm,
and singular value spectra.

Hypothesis: GRPO makes diffuse, high-rank changes (distribution sharpening),
            SDPO makes targeted, low-rank changes (genuine learning).

Usage:
    python weight_delta_svd.py [--base-path PATH] [--grpo-path PATH] [--sdpo-path PATH]
"""

import argparse
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

# Stub out missing DTensor modules/classes for cross-version compatibility
# (checkpoints saved with different torch versions may reference moved classes)
if 'torch.distributed._mesh_layout' not in sys.modules:
    _mesh_mod = types.ModuleType('torch.distributed._mesh_layout')
    class _MeshLayout:
        def __init__(self, *args, **kwargs): pass
    _mesh_mod._MeshLayout = _MeshLayout
    sys.modules['torch.distributed._mesh_layout'] = _mesh_mod

import torch
import torch.distributed.tensor._dtensor_spec as _dtensor_spec
if not hasattr(_dtensor_spec, 'ShardOrderEntry'):
    class ShardOrderEntry:
        def __init__(self, *args, **kwargs): pass
    _dtensor_spec.ShardOrderEntry = ShardOrderEntry

from safetensors import safe_open


def load_base_model(base_path: str) -> dict[str, torch.Tensor]:
    """Load base model from safetensors files."""
    print(f"Loading base model from {base_path}")
    state_dict = {}
    safetensor_files = sorted(Path(base_path).glob("*.safetensors"))
    for f in safetensor_files:
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in st.keys():
                state_dict[key] = st.get_tensor(key)
    print(f"  Loaded {len(state_dict)} parameters")
    return state_dict


def load_fsdp_checkpoint(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Load and merge FSDP-sharded checkpoint into a single state dict.

    FSDP shards split each parameter's flat storage across ranks.
    We concatenate them and reshape to the original parameter shape
    using the metadata from the HF config.
    """
    actor_path = Path(ckpt_path) / "actor" if not str(ckpt_path).endswith("actor") else Path(ckpt_path)

    # Find all shard files
    shard_files = sorted(actor_path.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No FSDP shard files found in {actor_path}")

    world_size = len(shard_files)
    print(f"Loading FSDP checkpoint from {actor_path} (world_size={world_size})")

    # Load all shards
    shards = []
    for f in sorted(shard_files, key=lambda x: int(str(x).split("rank_")[1].split(".")[0])):
        print(f"  Loading {f.name}...")
        shard = torch.load(str(f), map_location="cpu", weights_only=False)
        shards.append(shard)

    # Merge shards - stored as DTensors with Shard(dim=0) placement
    # Each rank holds a slice of rows; concatenate to reconstruct full params
    merged = {}
    all_keys = list(shards[0].keys())
    for key in all_keys:
        chunks = []
        for r in range(world_size):
            t = shards[r][key]
            # Extract local tensor from DTensor if needed
            if hasattr(t, '_local_tensor'):
                t = t._local_tensor
            chunks.append(t)

        if chunks[0].dim() >= 1 and all(c.shape == chunks[0].shape for c in chunks):
            # Shard(dim=0): concatenate along first dimension
            merged[key] = torch.cat(chunks, dim=0)
        else:
            # Replicated (scalars, norms) — just take rank 0
            merged[key] = chunks[0]

    print(f"  Merged {len(merged)} parameters")
    return merged


def compute_delta_stats(base_sd: dict, ft_sd: dict, name: str) -> list[dict]:
    """Compute weight delta SVD statistics between base and fine-tuned model."""
    results = []

    # Try to match keys - FSDP might use different naming
    base_keys = set(base_sd.keys())
    ft_keys = set(ft_sd.keys())

    # Find matching keys (handle potential prefix differences)
    matched = base_keys & ft_keys
    if not matched:
        # Try stripping common prefixes
        base_stripped = {k.replace("model.", ""): k for k in base_keys}
        ft_stripped = {k.replace("model.", ""): k for k in ft_keys}
        common = set(base_stripped.keys()) & set(ft_stripped.keys())
        print(f"  Direct key match: 0. Trying stripped: {len(common)}")
        key_pairs = [(base_stripped[k], ft_stripped[k]) for k in common]
    else:
        print(f"  Matched {len(matched)} keys directly")
        key_pairs = [(k, k) for k in matched]

    skipped = 0
    for base_key, ft_key in sorted(key_pairs):
        p_base = base_sd[base_key]
        p_ft = ft_sd[ft_key]

        # Only analyze 2D weight matrices
        if p_base.dim() != 2 or min(p_base.shape) <= 1:
            skipped += 1
            continue

        if p_base.shape != p_ft.shape:
            print(f"  Shape mismatch: {base_key} {p_base.shape} vs {p_ft.shape}")
            skipped += 1
            continue

        delta = (p_ft.float() - p_base.float())

        # SVD
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        # Metrics
        fro_norm = S.norm().item()
        spectral_norm = S[0].item()
        nuclear_norm = S.sum().item()

        # Effective rank: (sum S)^2 / (sum S^2)
        eff_rank = (S.sum() ** 2 / (S ** 2).sum()).item() if (S ** 2).sum() > 0 else 0

        # Stable rank: ||A||_F^2 / ||A||_2^2
        stable_rank = (fro_norm ** 2 / (spectral_norm ** 2)) if spectral_norm > 0 else 0

        # Fraction of variance in top-k singular values
        total_var = (S ** 2).sum().item()
        top1_frac = (S[0] ** 2).item() / total_var if total_var > 0 else 0
        top5_frac = (S[:5] ** 2).sum().item() / total_var if total_var > 0 else 0
        top10_frac = (S[:10] ** 2).sum().item() / total_var if total_var > 0 else 0

        # Relative change (Frobenius norm of delta / Frobenius norm of base)
        base_norm = p_base.float().norm().item()
        relative_change = fro_norm / base_norm if base_norm > 0 else 0

        # Layer type detection
        layer_type = "other"
        for lt in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "embed"]:
            if lt in base_key:
                layer_type = lt
                break

        # Extract layer number
        layer_num = -1
        parts = base_key.split(".")
        for p in parts:
            if p.isdigit():
                layer_num = int(p)
                break

        results.append({
            "param": base_key,
            "model": name,
            "shape": list(p_base.shape),
            "layer_num": layer_num,
            "layer_type": layer_type,
            "fro_norm": fro_norm,
            "spectral_norm": spectral_norm,
            "nuclear_norm": nuclear_norm,
            "eff_rank": eff_rank,
            "stable_rank": stable_rank,
            "relative_change": relative_change,
            "top1_var_frac": top1_frac,
            "top5_var_frac": top5_frac,
            "top10_var_frac": top10_frac,
            "top20_sv": S[:20].tolist(),
        })

    print(f"  Analyzed {len(results)} weight matrices, skipped {skipped}")
    return results


def print_summary(results: list[dict], name: str):
    """Print summary statistics for a model's weight deltas."""
    if not results:
        print(f"\n{'='*60}")
        print(f"  {name}: No results")
        return

    import statistics

    eff_ranks = [r["eff_rank"] for r in results]
    stable_ranks = [r["stable_rank"] for r in results]
    fro_norms = [r["fro_norm"] for r in results]
    rel_changes = [r["relative_change"] for r in results]
    top1_fracs = [r["top1_var_frac"] for r in results]
    top10_fracs = [r["top10_var_frac"] for r in results]

    print(f"\n{'='*60}")
    print(f"  {name} Weight Delta Summary")
    print(f"{'='*60}")
    print(f"  Params analyzed:     {len(results)}")
    print(f"  Effective rank:      mean={statistics.mean(eff_ranks):.1f}  median={statistics.median(eff_ranks):.1f}  std={statistics.stdev(eff_ranks):.1f}")
    print(f"  Stable rank:         mean={statistics.mean(stable_ranks):.1f}  median={statistics.median(stable_ranks):.1f}")
    print(f"  Frobenius norm:      mean={statistics.mean(fro_norms):.4f}  max={max(fro_norms):.4f}")
    print(f"  Relative change:     mean={statistics.mean(rel_changes):.6f}  max={max(rel_changes):.6f}")
    print(f"  Top-1 SV var frac:   mean={statistics.mean(top1_fracs):.4f}  (higher = more concentrated)")
    print(f"  Top-10 SV var frac:  mean={statistics.mean(top10_fracs):.4f}")

    # Per-layer-type breakdown
    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r)

    print(f"\n  Per layer type:")
    print(f"  {'Type':<12} {'Count':>5} {'MeanEffRank':>12} {'MeanRelChg':>12} {'MeanTop1Frac':>13}")
    for lt in sorted(by_type.keys()):
        items = by_type[lt]
        print(f"  {lt:<12} {len(items):>5} {statistics.mean([r['eff_rank'] for r in items]):>12.1f} {statistics.mean([r['relative_change'] for r in items]):>12.6f} {statistics.mean([r['top1_var_frac'] for r in items]):>13.4f}")

    # Per-layer breakdown (aggregate by layer number)
    by_layer = defaultdict(list)
    for r in results:
        if r["layer_num"] >= 0:
            by_layer[r["layer_num"]].append(r)

    print(f"\n  Per layer (every 5th):")
    print(f"  {'Layer':>5} {'MeanEffRank':>12} {'MeanFroNorm':>12} {'MeanRelChg':>12}")
    for layer_num in sorted(by_layer.keys()):
        if layer_num % 5 == 0:
            items = by_layer[layer_num]
            print(f"  {layer_num:>5} {statistics.mean([r['eff_rank'] for r in items]):>12.1f} {statistics.mean([r['fro_norm'] for r in items]):>12.4f} {statistics.mean([r['relative_change'] for r in items]):>12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Weight Delta SVD Analysis")
    parser.add_argument("--base-path", type=str,
                       default="/home/azureuser/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18",
                       help="Path to base model (safetensors)")
    parser.add_argument("--grpo-path", type=str,
                       default="/home/azureuser/checkpoints/qwen3-14b-grpo-mcts/global_step_40",
                       help="Path to GRPO checkpoint")
    parser.add_argument("--sdpo-path", type=str,
                       default="/home/azureuser/checkpoints/qwen3-14b-sdpo/global_step_480",
                       help="Path to SDPO checkpoint")
    parser.add_argument("--output", type=str, default="weight_delta_results.json",
                       help="Output JSON file")
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO analysis")
    parser.add_argument("--skip-sdpo", action="store_true", help="Skip SDPO analysis")
    args = parser.parse_args()

    # Load base model
    base_sd = load_base_model(args.base_path)

    all_results = []

    # GRPO analysis
    if not args.skip_grpo:
        print(f"\n--- GRPO Analysis ---")
        try:
            grpo_sd = load_fsdp_checkpoint(args.grpo_path)
            grpo_results = compute_delta_stats(base_sd, grpo_sd, "GRPO")
            all_results.extend(grpo_results)
            print_summary(grpo_results, "GRPO")
            del grpo_sd
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"GRPO failed: {e}")
            import traceback; traceback.print_exc()

    # SDPO analysis
    if not args.skip_sdpo:
        print(f"\n--- SDPO Analysis ---")
        try:
            sdpo_sd = load_fsdp_checkpoint(args.sdpo_path)
            sdpo_results = compute_delta_stats(base_sd, sdpo_sd, "SDPO")
            all_results.extend(sdpo_results)
            print_summary(sdpo_results, "SDPO")
            del sdpo_sd
        except Exception as e:
            print(f"SDPO failed: {e}")
            import traceback; traceback.print_exc()

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison if both available
    grpo_r = [r for r in all_results if r["model"] == "GRPO"]
    sdpo_r = [r for r in all_results if r["model"] == "SDPO"]

    if grpo_r and sdpo_r:
        print(f"\n{'='*60}")
        print(f"  GRPO vs SDPO Comparison")
        print(f"{'='*60}")

        import statistics
        g_eff = statistics.mean([r["eff_rank"] for r in grpo_r])
        s_eff = statistics.mean([r["eff_rank"] for r in sdpo_r])
        g_rel = statistics.mean([r["relative_change"] for r in grpo_r])
        s_rel = statistics.mean([r["relative_change"] for r in sdpo_r])
        g_top1 = statistics.mean([r["top1_var_frac"] for r in grpo_r])
        s_top1 = statistics.mean([r["top1_var_frac"] for r in sdpo_r])

        print(f"  {'Metric':<25} {'GRPO':>12} {'SDPO':>12} {'Ratio G/S':>12}")
        print(f"  {'Mean Effective Rank':<25} {g_eff:>12.1f} {s_eff:>12.1f} {g_eff/s_eff:>12.2f}")
        print(f"  {'Mean Relative Change':<25} {g_rel:>12.6f} {s_rel:>12.6f} {g_rel/s_rel:>12.2f}")
        print(f"  {'Mean Top-1 SV Var Frac':<25} {g_top1:>12.4f} {s_top1:>12.4f} {g_top1/s_top1:>12.2f}")

        if g_eff > s_eff:
            print(f"\n  >> GRPO has HIGHER effective rank ({g_eff:.1f} vs {s_eff:.1f})")
            print(f"     Consistent with diffuse/high-rank changes (distribution sharpening)")
        else:
            print(f"\n  >> SDPO has HIGHER effective rank ({s_eff:.1f} vs {g_eff:.1f})")
            print(f"     Unexpected - SDPO changes are more distributed than GRPO")

        if s_top1 > g_top1:
            print(f"  >> SDPO has MORE concentrated singular values (top-1: {s_top1:.4f} vs {g_top1:.4f})")
            print(f"     Consistent with targeted/surgical changes (genuine learning)")
        else:
            print(f"  >> GRPO has MORE concentrated singular values (top-1: {g_top1:.4f} vs {s_top1:.4f})")


if __name__ == "__main__":
    main()
