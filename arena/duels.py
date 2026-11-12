"""Play the direct games between rated bots, for the ladder.

    uv run python -m arena.duels due                       # what still needs playing, best first
    uv run python -m arena.duels play a b --seed S         # play one batch and save it
    uv run python -m arena.duels estimate a b              # seconds one batch should take

Every pair plays. A close pair at the top keeps playing fresh deals, a batch at a time.
A bot the ladder was started for is a challenger: it meets the top bots before anything else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from itertools import combinations
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bots import ROOT, SUB_PREFIX, load_manifest
from .direct import DIRECT_DIR, current, is_clear, load_matches, match_file, record
from .leaderboard import build
from .positions import encode_scores, summarize
from .run import GAME_ROUNDS, RESULTS_DIR, bot_hash, engine_version, run_sets

BATCH_SETS = 20
MAX_SETS = 100
TOP = 4
MARGIN = 1.25  # a batch can run slower than the games it was estimated from


def pair_seed(seed: int, a: str, b: str, batch: int) -> int:
    """Different deals for every pair and batch, so one bad run of cards can't decide several
    results. The Annual Computer Poker Competition had to rerun its 2016 finals over this."""
    first, second = sorted((a, b))
    return int.from_bytes(hashlib.sha256(f"{seed}|{first}|{second}|{batch}".encode()).digest()[:4], "big")


def due(root: Path = ROOT, first: Sequence[str] = ()) -> list[tuple[str, str]]:
    """Pairs with no current match, then close pairs among the top bots; best bots first.
    The bots named in ``first`` are challengers: their matches come before every other, and each
    challenger meets the best-ranked bot first, then the next, down the table."""
    results = [json.loads(p.read_text()) for p in sorted((root / RESULTS_DIR.relative_to(ROOT)).glob("*.json"))]
    order = [r["key"][len(SUB_PREFIX) :] for r in build(root)["rows"] if not r["anchor"] and not r.get("provisional")]
    rank = {name: k for k, name in enumerate(order)}
    done = {tuple(sorted((m["a"], m["b"]))): m for m in current(load_matches(root), results)}
    unplayed, close = [], []
    for a, b in combinations(order, 2):
        match = done.get(tuple(sorted((a, b))))
        if match is None:
            unplayed.append((a, b))
            continue
        wins, draws, losses = record(match)
        if rank[a] < TOP and rank[b] < TOP and not is_clear(wins, losses) and wins + draws + losses < MAX_SETS:
            close.append((a, b))
    def order_of(pair: tuple[str, str]) -> tuple[int, int, int]:
        challengers = [name for name in pair if name in first]
        if not challengers:
            return (1, rank[pair[0]] + rank[pair[1]], rank[pair[0]])
        opponent = min((rank[name] for name in pair if name not in first), default=rank[pair[0]])
        return (0, opponent, rank[challengers[0]])

    return sorted(unplayed, key=order_of) + close


def estimate(a: str, b: str, sets: int = BATCH_SETS, root: Path = ROOT) -> float:
    """Seconds one batch should take, so the ladder starts only what it can finish.

    Each bot's games against Random cost about what its own moves cost, so a pair costs the sum of
    the two. A batch this pair already played is a better guide, and wins when it is longer."""
    guess = 0.0
    for name in (a, b):
        path = root / RESULTS_DIR.relative_to(ROOT) / f"{name}.json"
        versus = json.loads(path.read_text())["versus"] if path.is_file() else []
        cheap = [v for v in versus if v.get("spec") == "random" and v.get("wall_s")] or [v for v in versus if v.get("wall_s")]
        guess += min((v["wall_s"] / v["sets"] for v in cheap), default=0.0) * sets
    first, second = sorted((a, b))
    path = match_file(root, first, second)
    played = json.loads(path.read_text())["batches"] if path.is_file() else []
    before = max((x["wall_s"] / x["sets"] for x in played if x.get("wall_s")), default=0.0) * sets
    return MARGIN * max(guess, before)


def play(
    a: str, b: str, seed: int, workers: int = 4, sets: int = BATCH_SETS, root: Path = ROOT,
    rounds: int = GAME_ROUNDS, folder: Path = DIRECT_DIR, stop_when_settled: bool = False,
) -> dict[str, Any]:  # fmt: skip
    """One more batch for this pair, added to its file. No seed is saved: the ladder's is secret.
    ``rounds`` and ``folder`` are for the weekly full-game ladder (arena.full)."""
    first, second = sorted((a, b))
    path = match_file(root, first, second, folder)
    hashes = (bot_hash(load_manifest(first)), bot_hash(load_manifest(second)))
    engine = engine_version(rounds)
    match = json.loads(path.read_text()) if path.is_file() else None
    if not match or (match["hash_a"], match["hash_b"]) != hashes or match["engine"] != engine:
        match = {"a": first, "b": second, "hash_a": hashes[0], "hash_b": hashes[1], "engine": engine, "rounds": rounds,
                 "batches": [], "pairs": [], "position_pairs": [], "king_pairs": []}  # fmt: skip
    me, other = SUB_PREFIX + first, SUB_PREFIX + second
    lineup = (me, other, me, other)
    run = run_sets(lineup, sets, pair_seed(seed, first, second, len(match["batches"])), workers, rounds=rounds,
                   stop_when_settled=stop_when_settled)  # fmt: skip
    _, by_position = summarize(run.results, lineup)
    match["pairs"] += by_position["delta"]
    match["position_pairs"] += by_position["weighted"]
    match["king_pairs"] += by_position["king"]
    match["scores"] = ",".join(filter(None, (match.get("scores"), encode_scores(run.results))))
    match["batches"].append({"sets": len(run.results), "played_at": datetime.now(UTC).isoformat(timespec="seconds"), "wall_s": run.wall_seconds})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(match, indent=1) + "\n")
    return match


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.duels", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    todo = commands.add_parser("due", help="print the pairs to play, one a:b per line")
    todo.add_argument("--first", nargs="*", default=[], help="challengers: their matches go first, against the top bots")
    one = commands.add_parser("play", help="play one batch for a pair and print its file")
    one.add_argument("a")
    one.add_argument("b")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--workers", type=int, default=4)
    one.add_argument("--stop-when-settled", action="store_true", help="end the batch once more sets could not change its result")
    cost = commands.add_parser("estimate", help="print the seconds one batch for a pair should take")
    cost.add_argument("a")
    cost.add_argument("b")
    args = parser.parse_args(argv)
    if args.command == "due":
        for a, b in due(first=args.first):
            print(f"{a}:{b}")
        return 0
    if args.command == "estimate":
        print(round(estimate(args.a, args.b)))
        return 0
    play(args.a, args.b, args.seed, args.workers, stop_when_settled=args.stop_when_settled)
    wins, draws, losses = record(json.loads(match_file(ROOT, args.a, args.b).read_text()))
    print(f"{min(args.a, args.b)} {wins}-{draws}-{losses} {max(args.a, args.b)}")
    print(match_file(ROOT, args.a, args.b).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
