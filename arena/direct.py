"""Direct games: every rated bot against every other, and what they say about the standings.

Benchmarks alone can't rank the bots against each other, because skill in a card game need not
be transitive. docs/fair-ranking.md gives the reasoning and its sources. Reading these files
needs no engine; arena.duels plays them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from math import comb
from pathlib import Path
from typing import Any

DIRECT_DIR = Path("results") / "direct"
SIGNIFICANCE = 0.05
# A match stops early once its result would happen by chance between equal bots less than once in
# a thousand, checked after every set from the 10th. Checking that often, a 40-set match between
# equal bots still stops early less than 0.5% of the time (worked out exactly, with no draws).
# Chess engine testing stops matches early too: Stockfish's Fishtest uses a sequential test.
SETTLED = 0.001
SETTLED_AFTER = 10


def match_file(root: Path, a: str, b: str, folder: Path = DIRECT_DIR) -> Path:
    """One file per pair of bots, whichever way round they are named."""
    first, second = sorted((a, b))
    return root / folder / f"{first}.vs.{second}.json"


def game_file(a: str, b: str) -> str:
    """The watchable game's file for two bots' code names, whichever way round."""
    slug = lambda name: re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")  # noqa: E731
    first, second = sorted((slug(a), slug(b)))
    return f"{first}.vs.{second}.json"


def load_matches(root: Path, folder: Path = DIRECT_DIR) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((root / folder).glob("*.vs.*.json"))]


def current(
    matches: Iterable[dict[str, Any]], results: Iterable[dict[str, Any]], engine: str | None = None
) -> list[dict[str, Any]]:
    """The matches both of whose bots are still as they were: a changed bot or engine voids them.

    Matches are played in the results' game length, unless ``engine`` names another one (the
    weekly full-game ladder): then they must be played in that, on the results' engine commit."""
    rated = {r["bot"]: r for r in results}
    kept = []
    for match in matches:
        bots = (rated.get(match["a"]), rated.get(match["b"]))
        same = (lambda r: r.get("engine") == match["engine"]) if engine is None else (
            lambda r: match["engine"] == engine and r.get("engine", "").split("/")[0] == engine.split("/")[0]
        )  # fmt: skip
        if all(r and same(r) for r in bots) and (
            bots[0]["bot_hash"], bots[1]["bot_hash"]
        ) == (match["hash_a"], match["hash_b"]):  # fmt: skip
            kept.append(match)
    return kept


def record(match: dict[str, Any]) -> tuple[int, int, int]:
    """Sets won by ``a``, drawn, and won by ``b``, under weighted rank."""
    a = f"sub:{match['a']}"
    wins = draws = losses = 0
    for pair in match["position_pairs"]:
        mine, theirs = (pair["a_wins"], pair["b_wins"]) if pair["a"] == a else (pair["b_wins"], pair["a_wins"])
        wins, draws, losses = wins + mine, draws + pair["draws"], losses + theirs
    return wins, draws, losses


def chance(wins: int, losses: int) -> float:
    """How likely a result at least this one-sided is between equal bots: a two-sided sign test,
    draws set aside."""
    played = wins + losses
    if not played:
        return 1.0
    return min(1.0, 2 * sum(comb(played, k) for k in range(min(wins, losses) + 1)) / 2**played)


def is_clear(wins: int, losses: int) -> bool:
    """Whether a result is unlikely between equal bots."""
    return chance(wins, losses) < SIGNIFICANCE


def is_settled(outcomes: str) -> bool:
    """Whether a match can stop: more sets could not plausibly change who is better.
    ``outcomes`` is one letter per set, W, D or L."""
    return len(outcomes) >= SETTLED_AFTER and chance(outcomes.count("W"), outcomes.count("L")) < SETTLED


def table(matches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The matches for a page: who, the record, and whether it is clear."""
    rows = []
    for match in matches:
        wins, draws, losses = record(match)
        rows.append({"a": f"sub:{match['a']}", "b": f"sub:{match['b']}", "a_wins": wins, "draws": draws,
                     "b_wins": losses, "sets": wins + draws + losses, "clear": is_clear(wins, losses)})  # fmt: skip
    return rows


def runoff(keys: Sequence[str], pairs: Iterable[Any]) -> list[str]:
    """Bots from winner to first out: repeatedly drop the one scoring least against those left.

    A bot can't rise by beating up weak bots, because weak bots leave first. ``keys`` must be in
    standings order, which breaks ties: the lower of two equal bots goes first.
    """
    alive = list(keys)
    records = list(pairs)
    out = []
    while len(alive) > 1:
        points = dict.fromkeys(alive, 0.0)
        sets = dict.fromkeys(alive, 0)
        for p in records:
            if p.a in points and p.b in points:
                total = p.a_wins + p.draws + p.b_wins
                points[p.a] += p.a_wins + p.draws / 2
                points[p.b] += p.b_wins + p.draws / 2
                sets[p.a] += total
                sets[p.b] += total
        scores = {k: points[k] / sets[k] if sets[k] else 0.0 for k in alive}
        lowest = min(scores.values())
        loser = next(k for k in reversed(alive) if scores[k] == lowest)
        alive.remove(loser)
        out.append(loser)
    return alive + out[::-1]
