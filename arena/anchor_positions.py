"""Play every pair of anchors for position data: their imported ladder has scores only.

    .venv/bin/python -m arena.anchor_positions --sets 20 --seed 1157764928 --workers 4

Writes ladder/positions.json after each pair; a pair already there is not played again.
"""

from __future__ import annotations

import argparse
import json

from .bots import ROOT
from .positions import encode_scores, summarize
from .run import load_anchors, run_sets

OUT = ROOT / "ladder" / "positions.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m arena.anchor_positions", description=__doc__.split("\n")[0])
    p.add_argument("--sets", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    data = json.loads(OUT.read_text()) if OUT.is_file() else {"sets": a.sets, "seed": a.seed, "matches": []}
    done = {(m["a"], m["b"]) for m in data["matches"]}
    specs = [x["spec"] for x in load_anchors()]
    # cheap pairs first, so an interrupted run still covers the most
    for x, y in sorted(((x, y) for i, x in enumerate(specs) for y in specs[i + 1 :]),
                       key=lambda q: sum(s.startswith(("hybrid", "search", "pimc")) for s in q)):  # fmt: skip
        if (x, y) in done:
            continue
        run = run_sets((x, y, x, y), a.sets, a.seed, a.workers)
        summary, pairs = summarize(run.results, (x, y, x, y))
        data["matches"].append({"a": x, "b": y, "summary": summary, "pairs": pairs["weighted"], "king_pairs": pairs["king"],
                                "scores": encode_scores(run.results)})  # fmt: skip
        OUT.write_text(json.dumps(data, indent=1) + "\n")
        w = pairs["weighted"][0]
        print(f"{x} vs {y}: {w['a_wins']}-{w['draws']}-{w['b_wins']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
