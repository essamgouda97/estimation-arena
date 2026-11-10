#!/usr/bin/env python3
"""Upload the models and their cards to Hugging Face and rewrite models.lock.json.

    HF_TOKEN=... uv run --with huggingface_hub python scripts/publish-models.py --owner NAME

Each folder under models/ holds a repo's card (README.md). PUBLISH lists each repo's files and
the paths in this repo that use them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena.models import LOCK_FILE, prefer_ipv4, sha256  # noqa: E402

PUBLISH = {
    "estimation-self-taught": {"net.npz": ["bots/bitter-net/net.npz", "bots/bitter-search/net.npz"]},
    "estimation-copycat": {"net.npz": ["bots/net-pimc/net.npz"]},
    "estimation-copycat-practice": {"net.npz": ["bots/rl-cards/net.npz", "bots/rl-hybrid/net.npz"]},
    "estimation-rl-v2": {
        "final.pt": ["runs/m4-rules-v2/checkpoints/final.pt"],
        "rlv2.npz": ["runs/m4-rules-v2/rlv2.npz"],
    },
    "estimation-guess-net": {"guess.npz": ["bots/guess-search/guess.npz"]},
    "estimation-hybrid-v4": {"hybrid.pt": ["runs/expert/round-1/hybrid.pt"]},  # --owner yosifelessawi
}


def main() -> int:
    prefer_ipv4()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # its uploader picks its own addresses
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--owner", required=True, help="Hugging Face user or organization")
    parser.add_argument("--only", nargs="*", default=list(PUBLISH), help="repo names to publish")
    args = parser.parse_args()

    api = HfApi()
    lock = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.is_file() else {}
    for name in args.only:
        repo = f"{args.owner}/{name}"
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=ROOT / "models" / name / "README.md", path_in_repo="README.md",
                        repo_id=repo, commit_message="Update the model card")
        for file, paths in PUBLISH[name].items():
            digests = {sha256(ROOT / p) for p in paths}
            if len(digests) != 1:
                raise SystemExit(f"{name}: {paths} are not the same file")
            commit = api.upload_file(path_or_fileobj=ROOT / paths[0], path_in_repo=file, repo_id=repo,
                                     commit_message=f"Add {file}")
            for p in paths:
                lock[p] = {"repo": repo, "revision": commit.oid, "file": file, "sha256": next(iter(digests))}
            print(f"{repo}/{file} at {commit.oid}")
    LOCK_FILE.write_text(json.dumps(dict(sorted(lock.items())), indent=1) + "\n")
    print(f"wrote {LOCK_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
