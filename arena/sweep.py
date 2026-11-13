"""Sweep a python submission's options against one opponent, in duplicate sets.

    uv run python -m arena.sweep --bot table-pressure --vs heuristic --sets 40 --workers 5 \\
        --grid pressure=0,0.5,1 shift=-0.25,0,0.25

Every combination plays the same deals (same seed), so the rows compare like with like. Each row
is the submission's set wins, draws and losses against the opponent, its first-place share and
its RSA, then the metrics of arena.positions, each with a 95% interval. Sort by RSA; trust a difference only
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
from .positions import summarize
from .run import run_sets


def _keep_sample(spec: str) -> None:
    """Give the live board one real decision to show. A failure here never stops the match."""
    from .samples import save_live

    try:
        save_live(spec)
    except Exception as error:  # the bot may not be explainable; its match still counts
        print(f"no sample for the live board: {error}", file=sys.stderr)


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
    combos = list(itertools.product(*values))
    for combo in combos:
        spec = SUB_PREFIX + args.bot + "".join(f":{k}={v}" for k, v in zip(keys, combo))
        if len(combos) == 1:
            _keep_sample(spec)
        run = run_sets((spec, args.vs, spec, args.vs), args.sets, args.seed, args.workers, kind="entry" if len(combos) == 1 else "tuning")
        report = build_report(run)
        places, _ = summarize(run.results, (spec, args.vs, spec, args.vs))
        pr, king, delta = (places[spec][k] for k in ("reward", "king", "delta"))
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
            "positions": places[spec]["shares"],
            "position_reward": pr["mean"],
            "king": king["mean"],
            "delta": delta["mean"],
        }
        rows.append(row)
        label = " ".join(f"{k}={v}" for k, v in zip(keys, combo))
        print(
            f"  {label:<34} sets {won:>3}-{pair.draws}-{lost:<3} first {me.first.mean:5.1%}"
            f"  RSA {me.rsa.mean:+7.1f} [{me.rsa.low:+.1f}, {me.rsa.high:+.1f}]"
            f"  delta {delta['mean']:+.1f} [{delta['low']:+.1f}, {delta['high']:+.1f}]"
            f"  king {king['mean']:.1%} [{king['low']:.1%}, {king['high']:.1%}]"
            f"  position {pr['mean']:+.2f} [{pr['low']:+.2f}, {pr['high']:+.2f}]"
            f"  K/SK/SKo/Ko " + "/".join(f"{x:.0%}" for x in places[spec]["shares"]),
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
