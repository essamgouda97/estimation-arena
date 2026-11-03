"""Sweep a python submission's options against one opponent, in duplicate sets.

    uv run python -m arena.sweep --bot table-pressure --vs heuristic --sets 40 --workers 5 \\
        --grid pressure=0,0.5,1 shift=-0.25,0,0.25

Every combination plays the same deals (same seed), so the rows compare like with like. Each row
is the submission's set wins, draws and losses against the opponent, its first-place share and
its RSA (score advantage per game) with a 95% interval. Sort by RSA; trust a difference only
when the intervals do not overlap. The sweep tunes on --seed; rate the winner on another seed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from estimation.arena.report import build_report

from .bots import SUB_PREFIX, load_manifest
from .run import run_sets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.sweep", description=__doc__.split("\n")[0])
    parser.add_argument("--bot", required=True)
    parser.add_argument("--vs", default="heuristic", help="opponent spec (heuristic)")
    parser.add_argument("--grid", nargs="+", required=True, help="key=v1,v2,... per option")
    parser.add_argument("--sets", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, help="write the rows as JSON here")
    args = parser.parse_args(argv)
    load_manifest(args.bot)
    keys, values = [], []
    for item in args.grid:
        key, _, vals = item.partition("=")
        keys.append(key)
        values.append(vals.split(","))
    rows = []
    print(f"{args.bot} vs {args.vs}, {args.sets} sets, seed {args.seed}", flush=True)
    for combo in itertools.product(*values):
        spec = SUB_PREFIX + args.bot + "".join(f":{k}={v}" for k, v in zip(keys, combo))
        run = run_sets((spec, args.vs, spec, args.vs), args.sets, args.seed, args.workers)
        report = build_report(run)
        me = next(b for b in report.bots if b.name == spec)
        pair = next(p for p in report.pairs if spec in (p.a, p.b))
        won, lost = (pair.a_wins, pair.b_wins) if pair.a == spec else (pair.b_wins, pair.a_wins)
        row = {
            **dict(zip(keys, combo)),
            "won": won,
            "draws": pair.draws,
            "lost": lost,
            "first": me.first.mean,
            "rsa": me.rsa.mean,
            "rsa_low": me.rsa.low,
            "rsa_high": me.rsa.high,
            "wall_s": round(run.wall_seconds),
        }
        rows.append(row)
        label = " ".join(f"{k}={v}" for k, v in zip(keys, combo))
        print(
            f"  {label:<34} sets {won:>3}-{pair.draws}-{lost:<3} first {me.first.mean:5.1%}"
            f"  RSA {me.rsa.mean:+7.1f} [{me.rsa.low:+.1f}, {me.rsa.high:+.1f}]",
            flush=True,
        )
    rows.sort(key=lambda r: r["rsa"], reverse=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
    best = rows[0]
    print("best:", " ".join(f"{k}={best[k]}" for k in keys), f"RSA {best['rsa']:+.1f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
