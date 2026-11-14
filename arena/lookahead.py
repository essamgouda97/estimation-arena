"""What a look-ahead bot needs to play for the game, not the round.

A look-ahead scores each imagined end of the round. Scored in points, it plays every round alike.
A game is decided by finishing position, so ``PositionScorer`` scores an imagined round by the
position reward (arena.positions.REWARDS) the seat can expect once the game ends: exact in the
last round, and before it the rounds still to come are a random swing per seat.
``StandingsScorer`` replaces that swing with a small net that learned, in self-play, where a seat
finishes from the scores so far (jaxenv.selfplay saves it beside the policy).

The game's length is public: arena.run puts it in ``ESTIMATION_GAME_ROUNDS`` for every bot.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Sequence

import numpy as np

from estimation.bots.scoring import estimates_of
from estimation.engine import Observation, RoundOutcome, RulesConfig, score_round

from .positions import REWARDS

GAME_ROUNDS_ENV = "ESTIMATION_GAME_ROUNDS"  # rounds in the game being played; unset or 0: a whole game
WHOLE_GAME = 18
ROUND_SPREAD = 19.0  # points: sd of one seat's score in one round (120 self-play rounds, 2026-11-12)
SCORE_NUDGE = 1e-4  # per point of round lead: breaks ties between cards that finish alike

_half = np.random.default_rng(0).standard_normal((64, 4))
_SWINGS = np.concatenate([_half, -_half])  # the same swings for every outcome, so they compare
_UP_TO = np.cumsum((0.0, *REWARDS))  # reward of the first k positions together


def rounds_after(round_no: int) -> int:
    """Rounds still to play once round ``round_no`` (from 1) is over."""
    total = int(os.environ.get(GAME_ROUNDS_ENV) or 0) or WHOLE_GAME
    return max(total - round_no, 0)


def position_value(scores: Sequence[float], seat: int, spread: float = 0.0) -> float:
    """``seat``'s expected position reward if the game ended on ``scores`` plus, per seat, a
    random swing of sd ``spread``. Tied seats share the positions they hold."""
    totals = np.asarray(scores, np.float64)[None]
    if spread:
        totals = totals + spread * _SWINGS
    mine = totals[:, seat : seat + 1]
    above = (totals > mine).sum(1)
    tied = (totals == mine).sum(1)
    return float(((_UP_TO[above + tied] - _UP_TO[above]) / tied).mean())


class PositionScorer:
    """Scores final trick counts of the round in ``obs`` from ``obs.seat``'s side: the position
    reward it can expect at the end of the game, nudged by its lead in the round."""

    def __init__(self, obs: Observation, rules: RulesConfig) -> None:
        self._me, self._scores, self._rules = obs.seat, obs.scores, rules
        self._spread = ROUND_SPREAD * math.sqrt(rounds_after(obs.round_no))
        self._cache: dict[tuple[int, ...], float] = {}
        self._outcome = RoundOutcome(
            estimates=estimates_of(obs),
            tricks=(13, 0, 0, 0),
            caller=obs.caller,
            dash_callers=obs.dash_callers,
            with_players=tuple(w.seat for w in obs.with_players),
            last_estimator=obs.last_estimator,
            risks=obs.risks or 0,
            carry=obs.carry,
            round_no=obs.round_no,
            trump=obs.trump,
            fixed_trump=obs.fixed_trump,
        )

    def value(self, tricks: Sequence[int]) -> float:
        key = tuple(tricks)
        if key not in self._cache:
            won = score_round(dataclasses.replace(self._outcome, tricks=key), self._rules).scores
            lead = won[self._me] - (sum(won) - won[self._me]) / 3
            after = [have + got for have, got in zip(self._scores, won)]
            self._cache[key] = position_value(after, self._me, self._spread) + SCORE_NUDGE * lead
        return self._cache[key]


def standings_view(scores: Sequence[float], seat: int, share_left: float, carry: int) -> np.ndarray:
    """What the standings net reads for ``seat``, as jaxenv.selfplay.standings_view writes it:
    the other three's leads over it (largest first, in units of 200 points), the share of the
    game still to play, and the carry."""
    leads = sorted(((s - scores[seat]) / 200.0 for k, s in enumerate(scores) if k != seat), reverse=True)
    return np.array([*np.clip(leads, -5.0, 5.0), share_left, math.log2(max(carry, 1)) / 10.0])


def standings_net(path: str):
    """The standings net saved beside a policy (jaxenv.selfplay): from the scores after a round,
    the share of the game left and the carry, a seat's expected position reward at the end of
    the game. It was learned from games, where ``position_value`` assumes a random swing."""
    data = np.load(path)
    layers, k = [], 0
    while f"sw{k}" in data:
        layers.append((data[f"sw{k}"].astype(np.float64), data[f"sb{k}"].astype(np.float64)))
        k += 1
    if not layers:
        raise ValueError(f"{path} holds no standings net")

    def value(scores: Sequence[float], seat: int, rounds_left: int, carry: int) -> float:
        total = int(os.environ.get(GAME_ROUNDS_ENV) or 0) or WHOLE_GAME
        x = standings_view(scores, seat, rounds_left / total, carry)
        for w, b in layers[:-1]:
            x = np.maximum(x @ w + b, 0.0)
        return float((x @ layers[-1][0] + layers[-1][1])[0])

    return value


class StandingsScorer(PositionScorer):
    """As ``PositionScorer``, with the learned standings net in place of the random swing. The
    last round of a game needs no guess and is scored exactly."""

    def __init__(self, obs: Observation, rules: RulesConfig, net) -> None:
        super().__init__(obs, rules)
        self._net, self._left = net, rounds_after(obs.round_no)

    def value(self, tricks: Sequence[int]) -> float:
        key = tuple(tricks)
        if not self._left:
            return super().value(key)
        if key not in self._cache:
            result = score_round(dataclasses.replace(self._outcome, tricks=key), self._rules)
            won = result.scores
            lead = won[self._me] - (sum(won) - won[self._me]) / 3
            after = [have + got for have, got in zip(self._scores, won)]
            self._cache[key] = self._net(after, self._me, self._left, result.next_carry) + SCORE_NUDGE * lead
        return self._cache[key]


def skipped_round_value(obs: Observation) -> float:
    """The position value of a round nobody scores in."""
    return position_value(obs.scores, obs.seat, ROUND_SPREAD * math.sqrt(rounds_after(obs.round_no)))
