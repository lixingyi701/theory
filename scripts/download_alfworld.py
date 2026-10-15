#!/usr/bin/env python3
"""Download official text-only data archives. No detector or pretrained agents."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

URLS = [
    "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip",
]


def safe_extract(archive: Path, destination: Path):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes destination")
        zipped.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/alfworld"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"archives": URLS, "destination": str(args.data_root)}, indent=2))
    if not args.execute:
        print("Plan only; --execute downloads and extracts the official text data.")
        return
    cache = args.data_root / "archives"
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for url in URLS:
        target = cache / url.rsplit("/", 1)[1]
        if not target.exists():
            temporary = target.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(target)
        h = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        records.append({"url": url, "bytes": target.stat().st_size, "sha256": h.hexdigest()})
        safe_extract(target, args.data_root)
    (args.data_root / "download_receipt.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
