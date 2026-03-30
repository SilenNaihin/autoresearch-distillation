"""Convert FSDP-sharded checkpoint to HuggingFace format.

Merges DTensor shards, maps keys to HF naming, and saves as safetensors
with config/tokenizer copied from the base model.

Usage:
    python fsdp_to_hf.py --ckpt-path /path/to/global_step_X \
                         --base-path /path/to/base/model \
                         --output-path /path/to/output
"""
import argparse
import json
import shutil
import sys
import types
from pathlib import Path

# Stub missing DTensor modules for cross-version compatibility
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

from safetensors.torch import save_file


def load_and_merge_fsdp(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Load FSDP shards and merge into single state dict."""
    actor_path = Path(ckpt_path) / "actor" if not str(ckpt_path).endswith("actor") else Path(ckpt_path)

    shard_files = sorted(actor_path.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No FSDP shard files found in {actor_path}")

    world_size = len(shard_files)
    print(f"Loading FSDP checkpoint from {actor_path} (world_size={world_size})")

    shards = []
    for f in sorted(shard_files, key=lambda x: int(str(x).split("rank_")[1].split(".")[0])):
        print(f"  Loading {f.name}...")
        shard = torch.load(str(f), map_location="cpu", weights_only=False)
        shards.append(shard)

    merged = {}
    for key in list(shards[0].keys()):
        chunks = []
        for r in range(world_size):
            t = shards[r][key]
            if hasattr(t, '_local_tensor'):
                t = t._local_tensor
            chunks.append(t)

        if chunks[0].dim() >= 1 and all(c.shape == chunks[0].shape for c in chunks):
            merged[key] = torch.cat(chunks, dim=0)
        else:
            merged[key] = chunks[0]

    print(f"  Merged {len(merged)} parameters")
    del shards
    return merged


def save_as_hf(state_dict: dict, base_path: str, output_path: str):
    """Save merged state dict as HuggingFace safetensors model."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(base_path)

    # Copy config and tokenizer files from base model
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = base / fname
        if src.exists():
            shutil.copy2(str(src), str(out / fname))
            print(f"  Copied {fname}")

    # Build shard index matching base model's sharding
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            base_index = json.load(f)
        weight_map = base_index["weight_map"]
    else:
        # Single file fallback
        weight_map = {k: "model.safetensors" for k in state_dict}

    # Group parameters by their target shard file
    shard_groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        shard_name = weight_map.get(key)
        if shard_name is None:
            print(f"  Warning: {key} not in base index, putting in last shard")
            shard_name = "model-extra.safetensors"
        if shard_name not in shard_groups:
            shard_groups[shard_name] = {}
        shard_groups[shard_name][key] = tensor

    # Save each shard
    new_weight_map = {}
    total_size = 0
    for shard_name in sorted(shard_groups.keys()):
        tensors = shard_groups[shard_name]
        shard_path = out / shard_name
        print(f"  Saving {shard_name} ({len(tensors)} tensors)...")
        save_file(tensors, str(shard_path))
        for k, v in tensors.items():
            new_weight_map[k] = shard_name
            total_size += v.numel() * v.element_size()

    # Write index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Saved {len(state_dict)} parameters in {len(shard_groups)} shards")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"  Output: {out}")


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HuggingFace format")
    parser.add_argument("--ckpt-path", required=True, help="Path to FSDP checkpoint (global_step_X dir)")
    parser.add_argument("--base-path", required=True, help="Path to base HF model (for config/tokenizer)")
    parser.add_argument("--output-path", required=True, help="Output directory for HF model")
    args = parser.parse_args()

    state_dict = load_and_merge_fsdp(args.ckpt_path)
    save_as_hf(state_dict, args.base_path, args.output_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
