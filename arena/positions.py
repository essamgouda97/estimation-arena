"""Final positions in a Bola: King, Sub-King, Sub-Kooz, Kooz (last), and the metrics built on them.

Three views of the same games, each with its own set winner and so its own Elo:

- weighted: King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1, the two ends counting most. An even
  match is 0. The leaderboard's default ranking.
- king: 1 for King, 0 for anything else. An even match is 25%.
- delta: my seats' mean score minus the opponent's seats' mean score, per game. An even match is
  0. Unbounded, so one exploding game can move the mean; its Elo uses only who won each set.

Tied seats share the positions they occupy. A set goes to the bot with the higher value over its
seats and rotations. The raw scores are kept (``encode_scores``), so a new metric needs no replay.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from estimation.arena.duplicate import SetResult, slot_at

NAMES = ("King", "Sub-King", "Sub-Kooz", "Kooz")
REWARDS = (1.0, 0.2, -0.2, -1.0)


def position_shares(scores: Sequence[int], seat: int) -> list[float]:
    """The share of each of the four positions ``seat`` holds, ties split evenly."""
    above = sum(s > scores[seat] for s in scores)
    tied = sum(s == scores[seat] for s in scores)
    return [1 / tied if above <= p < above + tied else 0.0 for p in range(4)]


def position_reward(scores: Sequence[int], seat: int) -> float:
    return sum(share * r for share, r in zip(position_shares(scores, seat), REWARDS))


def encode_scores(results: Iterable[SetResult]) -> str:
    """Every game's four seat scores, compactly: a result file stays small and readable in git."""
    return ",".join(" ".join(map(str, scores)) for r in results for scores in r.scores)


def decode_scores(text: str) -> list[SetResult]:
    games = [tuple(int(x) for x in game.split()) for game in text.split(",")]
    return [SetResult(k, 0, tuple(games[k * 4 : k * 4 + 4]), (0,) * 4) for k in range(len(games) // 4)]


def _per_set(result: SetResult, lineup: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {b: {"shares": [0.0] * 4, "score": 0.0, "games": 0} for b in dict.fromkeys(lineup)}
    for rotation, scores in enumerate(result.scores):
        for seat in range(4):
            side = out[lineup[slot_at(seat, rotation)]]
            side["shares"] = [a + b for a, b in zip(side["shares"], position_shares(scores, seat))]
            side["score"] += scores[seat]
            side["games"] += 1
    for side in out.values():
        side["shares"] = [x / side["games"] for x in side["shares"]]
        side["score"] /= side["games"]
        side["reward"] = sum(s * r for s, r in zip(side["shares"], REWARDS))
        side["king"] = side["shares"][0]
    for b, side in out.items():
        others = [o["score"] for name, o in out.items() if name != b]
        side["delta"] = side["score"] - sum(others) / len(others) if others else 0.0
    return out


def _interval(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n)
    return {"mean": mean, "low": mean - half, "high": mean + half, "sets": n}


def _pairs(sets: Sequence[dict[str, dict[str, Any]]], bots: Sequence[str], key: str) -> list[dict[str, Any]]:
    pairs = []
    for i, a in enumerate(bots):
        for b in bots[i + 1 :]:
            wins = [0, 0, 0]
            for s in sets:
                diff = s[a][key] - s[b][key]
                wins[0 if diff > 1e-9 else 2 if diff < -1e-9 else 1] += 1
            pairs.append({"a": a, "b": b, "a_wins": wins[0], "draws": wins[1], "b_wins": wins[2]})
    return pairs


def summarize(results: Iterable[SetResult], lineup: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Each bot's position shares and metrics with 95% intervals over sets, and the set-by-set
    pair records under each metric ("weighted", "king", "delta")."""
    sets = [_per_set(r, lineup) for r in results]
    bots = list(dict.fromkeys(lineup))
    summary = {
        b: {
            "shares": [sum(s[b]["shares"][p] for s in sets) / len(sets) for p in range(4)],
            "reward": _interval([s[b]["reward"] for s in sets]),
            "king": _interval([s[b]["king"] for s in sets]),
            "delta": _interval([s[b]["delta"] for s in sets]),
        }
        for b in bots
    }
    return summary, {"weighted": _pairs(sets, bots, "reward"), "king": _pairs(sets, bots, "king"), "delta": _pairs(sets, bots, "delta")}


def set_outcomes(results: Iterable[SetResult], lineup: Sequence[str], me: str) -> str:
    """One letter per set for ``me`` under weighted rank: W, L or D."""
    out = []
    for r in results:
        sides = _per_set(r, lineup)
        other = max(s["reward"] for b, s in sides.items() if b != me)
        diff = sides[me]["reward"] - other
        out.append("W" if diff > 1e-9 else "L" if diff < -1e-9 else "D")
    return "".join(out)
