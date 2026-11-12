"""How far a rating could be off, worked out from the same games that set it.

A rating is a best guess from a limited number of sets. ``Margins`` says how wide that guess is:
a bot shown as 1,840 ±60 would land between 1,780 and 1,900 in 19 of 20 replays of the same
schedule on other deals. Two bots whose gap is inside its own margin are too close to call.

The maths is the standard one for a Bradley-Terry fit: the curvature of the likelihood at the
fitted ratings gives how much every pair of ratings can move together (BayesElo reports its
margins the same way). A draw counts as half a win and half a loss, which leaves the margins a
little wide, never too narrow.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from estimation.arena.ratings import ANCHOR, DEFAULT_PRIOR, ELO_SCALE, PairRecord

NINETEEN_IN_TWENTY = 1.96  # standard errors either side of a 95% range


class Margins:
    """The uncertainty of one rating fit. ``pairs`` and ``ratings`` are what went into and came
    out of ``bradley_terry``; the anchor is fixed, so it has no uncertainty of its own."""

    def __init__(
        self, pairs: Iterable[PairRecord], ratings: Mapping[str, float], anchor: str = ANCHOR, prior: float = DEFAULT_PRIOR
    ) -> None:
        self.ratings = dict(ratings)
        free = [bot for bot in sorted(self.ratings) if bot != anchor]
        self._at = {bot: k for k, bot in enumerate(free)}
        sets: dict[tuple[str, str], float] = {}
        for p in pairs:
            if p.a != p.b and p.a in self.ratings and p.b in self.ratings and p.a_wins + p.draws + p.b_wins:
                key = (min(p.a, p.b), max(p.a, p.b))
                sets[key] = sets.get(key, 0.0) + p.a_wins + p.draws + p.b_wins
        information = np.zeros((len(free), len(free)))
        per_point = math.log(10) / ELO_SCALE
        for (a, b), played in sets.items():
            win = 1 / (1 + 10 ** ((self.ratings[b] - self.ratings[a]) / ELO_SCALE))
            weight = (played + prior) * win * (1 - win) * per_point**2
            for bot, other in ((a, b), (b, a)):
                if bot in self._at:
                    information[self._at[bot], self._at[bot]] += weight
                    if other in self._at:
                        information[self._at[bot], self._at[other]] -= weight
        self._spread = np.linalg.inv(information) if free else information

    def _weights(self, share: Mapping[str, float]) -> np.ndarray:
        weights = np.zeros(len(self._at))
        for bot, amount in share.items():
            if bot in self._at:
                weights[self._at[bot]] += amount
        return weights

    def _margin(self, share: Mapping[str, float]) -> float:
        weights = self._weights(share)
        return NINETEEN_IN_TWENTY * math.sqrt(max(float(weights @ self._spread @ weights), 0.0))

    def of(self, bot: str, field: Sequence[str]) -> float | None:
        """The ± on ``bot``'s rating, measured against the average of the bots in ``field``.

        Measured against Random alone, every strong bot would carry the doubt of the whole chain
        of results down to Random, though that doubt moves them all together and can't change
        their order. Against the field's average it drops out."""
        field = [other for other in field if other in self.ratings]
        if bot not in self.ratings or not field:
            return None
        share = {other: -1 / len(field) for other in field}
        share[bot] = share.get(bot, 0.0) + 1
        return self._margin(share)

    def separates(self, a: str, b: str) -> bool:
        """Whether the gap between two rated bots is wider than its own margin."""
        if a not in self.ratings or b not in self.ratings:
            return False
        return abs(self.ratings[a] - self.ratings[b]) > self._margin({a: 1.0, b: -1.0})
