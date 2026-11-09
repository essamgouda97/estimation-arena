"""kooz-bot: a deep-search Hybrid that only Dash Calls when the Dash Call is clearly worth it.

Built on estimation-engine's HybridBot, with nets from our own from-scratch training run
(``configs/baseline.toml``):

- **Bidding:** BiddingNet picks every Dash Call, bid, With, Pass and estimate, except that a Dash
  Call is only made when BiddingNet values it at ``dash_min`` points or more (10 by default).
  BiddingNet overrates Dash Calls against strong players; below +10 they lose points
  (docs/results.md).
- **Card play:** for each card it imagines ``samples`` deals of the unseen cards (64), placed by
  GuessNet's guesses and kept by how well they explain everyone's bidding (``deals=both``), plays
  every legal card (``top_k=0``) out with CardPlayNet for all four seats, and plays the best mean.

It reads only its own Observation. Options: ``checkpoint`` (a path from the repo root),
``dash_min``, and any HybridConfig field, for example ``samples=32``.

Arena entry point: ``make(seed, rules, **options)``.
"""

from pathlib import Path

import numpy as np
from estimation.bots.hybrid.bot import HybridBot, HybridConfig
from estimation.bots.registry import configure
from estimation.engine import DASH_CALL, NO_DASH_CALL, Observation, Phase, RulesConfig
from estimation.rl.encoder import encode
from estimation.rl.networks import REWARD_SCALE

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = "runs/kooz-bot/final.pt"
SEARCH = {"samples": "64", "top_k": "0", "deals": "both"}
DASH_MIN = 10.0


class KoozBot:
    """HybridBot with a floor on the value of a Dash Call."""

    def __init__(self, hybrid: HybridBot, dash_min: float) -> None:
        self.hybrid = hybrid
        self.dash_min = dash_min

    def __repr__(self) -> str:
        return f"KoozBot(dash_min={self.dash_min}, {self.hybrid!r})"

    def dash_value(self, obs: Observation) -> float:
        """BiddingNet's value of a Dash Call here, in points."""
        x = encode(obs, self.hybrid.config.rules)[np.newaxis]
        return float(self.hybrid.policy.bid_values(x)[0][DASH_CALL]) * REWARD_SCALE

    def act(self, obs: Observation, legal: list[int]) -> int:
        if (
            obs.phase is Phase.DASH
            and DASH_CALL in legal
            and NO_DASH_CALL in legal
            and self.dash_value(obs) < self.dash_min
        ):
            return NO_DASH_CALL
        return self.hybrid.act(obs, legal)


def make(
    seed: int,
    rules: RulesConfig,
    checkpoint: str = CHECKPOINT,
    dash_min: str | float = DASH_MIN,
    **options: str,
) -> KoozBot:
    path = Path(checkpoint)
    config = HybridConfig(rules=rules, checkpoint=str(path if path.is_absolute() else ROOT / path))
    hybrid = HybridBot(seed, configure(config, {**SEARCH, **options}, "kooz-bot"))
    return KoozBot(hybrid, float(dash_min))
