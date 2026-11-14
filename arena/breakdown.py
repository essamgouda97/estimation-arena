"""Where one bot wins and loses points against another, round by round.

    uv run python -m arena.breakdown sub:guess-search sub:hybrid-v4 --deals 40 --workers 4 --out DIR

Rated games keep only their scores, and their deals are secret. So this plays the two bots on
public deals, each deal twice with the seats swapped, records every move, and splits each bot's
round scores by what it was doing: calling the trump, going With, bidding Dash, or following.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .replay import game_seed, record_game
from .run import GAME_ROUNDS

FIRST_DEAL = 10_000  # clear of the deals the watch games use
PLAY_BASE, DASH_CALL = 68, 1
ROLES = ("caller", "with", "dash", "follower")


def _beats(card: int, best: int, trump: int) -> bool:
    return card > best if card // 13 == best // 13 else card // 13 == trump


def tricks_won(round_: dict[str, Any]) -> list[int]:
    won, trick = [0, 0, 0, 0], []
    for seat, action in round_["moves"]:
        if action < PLAY_BASE:
            continue
        trick.append((seat, action - PLAY_BASE))
        if len(trick) == 4:
            best = trick[0]
            for played in trick[1:]:
                if _beats(played[1], best[1], round_["trump"]):
                    best = played
            won[best[0]] += 1
            trick = []
    return won


def seat_rounds(game: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per seat per round that was played out: who, in what role, what it said, won and scored."""
    rows = []
    for round_ in game["rounds"]:
        if round_["caller"] is None or "after" not in round_ or None in round_["estimates"]:
            continue  # everyone passed, or the game was cut off mid-round
        won = tricks_won(round_)
        dash = {seat for seat, action in round_["moves"] if action == DASH_CALL}
        for seat in range(4):
            role = "caller" if seat == round_["caller"] else "with" if seat in round_["with"] else "dash" if seat in dash else "follower"
            rows.append({"bot": game["seats"][seat], "round": round_["round"], "role": role, "said": round_["estimates"][seat],
                         "won": won[seat], "points": round_["after"][seat] - round_["before"][seat]})  # fmt: skip
    return rows


def _play(job: tuple[str, str, int]) -> dict[str, Any]:
    a, b, number = job
    seats = [a, b, a, b] if number % 2 == 0 else [b, a, b, a]
    return record_game(seats, game_seed(a, b) + FIRST_DEAL + number // 2, rounds=GAME_ROUNDS)


def play(a: str, b: str, deals: int, workers: int) -> list[dict[str, Any]]:
    """``deals`` public deals, each played with the seats both ways round."""
    jobs = [(a, b, number) for number in range(2 * deals)]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_play, jobs))


def summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Per bot and role (and ``all``): rounds, share of the bot's rounds, how often it made what it
    said, its average points, and its average tricks won beyond what it said."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["bot"], row["role"]].append(row)
        groups[row["bot"], "all"].append(row)
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (bot, role), group in groups.items():
        n = len(group)
        out[bot][role] = {"rounds": n, "share": n / len(groups[bot, "all"]), "made": sum(r["won"] == r["said"] for r in group) / n,
                          "points": sum(r["points"] for r in group) / n, "said": sum(r["said"] for r in group) / n,
                          "over": sum(r["won"] > r["said"] for r in group) / n, "under": sum(r["won"] < r["said"] for r in group) / n}  # fmt: skip
    return out


def report(rows: list[dict[str, Any]], a: str, b: str) -> str:
    stats, lines = summary(rows), []
    for bot in (a, b):
        lines.append(f"{bot}")
        lines.append(f"  {'role':<9}{'rounds':>7}{'share':>7}{'made':>7}{'too many':>9}{'too few':>8}{'said':>6}{'points':>8}")
        for role in ("all", *ROLES):
            s = stats[bot].get(role)
            if s:
                lines.append(f"  {role:<9}{s['rounds']:>7}{s['share']:>7.0%}{s['made']:>7.0%}{s['over']:>9.0%}{s['under']:>8.0%}{s['said']:>6.1f}{s['points']:>+8.1f}")
    lines.append("points by round (first bot, second bot):")
    for number in sorted({r["round"] for r in rows}):
        mean = lambda bot: (lambda g: sum(g) / max(len(g), 1))([r["points"] for r in rows if r["bot"] == bot and r["round"] == number])  # noqa: E731
        lines.append(f"  round {number}: {mean(a):+.1f}  {mean(b):+.1f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.breakdown", description=__doc__.split("\n")[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--deals", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, help="keep the games and the rows here")
    args = parser.parse_args(argv)
    games = play(args.a, args.b, args.deals, args.workers)
    rows = [row for game in games for row in seat_rounds(game)]
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "games.json").write_text(json.dumps(games, separators=(",", ":")))
        (args.out / "rows.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(report(rows, args.a, args.b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
