#!/usr/bin/env python3
"""Print the pinned download plan; only --execute downloads model weights."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    parser.add_argument("--registry", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model = json.loads(args.registry.read_text(encoding="utf-8"))[args.model]
    print(json.dumps(model, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("Plan only; add --execute tomorrow to download.")
        return
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=model["repo_id"], revision=model["revision"],
                                  local_dir=model["local_dir"],
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md", "LICENSE*"]))
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        if not (path / shard).is_file():
            raise FileNotFoundError(shard)
    receipt = {**model, "weight_files": [{"file": f, "bytes": (path / f).stat().st_size} for f in shards],
               "verification": "Pinned HF snapshot and shard presence; not independent weight checksum audit"}
    (path / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(shards)} shards to {path}")


if __name__ == "__main__":
    main()
