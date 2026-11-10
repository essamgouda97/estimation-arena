"""Self-taught + look-ahead with a guess net of our own (train/guess_net.py).

The same bot as bots/bitter-search, except the hidden hands it imagines: each unseen card goes
to a seat in proportion to our guess net's chance that the seat holds it, and the drawn deals
are then kept by how well they explain the bidding, as Hybrid v4 does with Yosif's guess net.

Options: ``guess`` (path from this folder), ``net`` and any HybridConfig field, e.g.
``sub:guess-search:samples=64``. It imagines 256 deals per card by default.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from estimation.bots.hybrid.bot import BOTH, HybridConfig
from estimation.bots.registry import configure
from estimation.engine import Observation, RulesConfig

from jaxenv.encode_np import encode
from train.guess_net import chances_np, load_np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bitter_search_bot", HERE.parent / "bitter-search" / "bot.py")
_bitter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bitter)


class GuessSearch(_bitter.BitterSearch):
    def __init__(self, seed: int, config: HybridConfig, net: str, guess: str) -> None:
        super().__init__(seed, replace(config, deals="bidding"), net, "net")  # RL v2 has no guess net
        self.config = replace(config, deals=BOTH)
        self._guess = load_np(guess)
        self._legal: list[int] = []

    def act(self, obs: Observation, legal: list[int]) -> int:
        self._legal = legal
        return super().act(obs, legal)

    def guess_chances(self, obs: Observation) -> list[list[float]]:
        """Our net's chance of each other seat (the next one first) holding each card."""
        x = encode(obs, self._legal)[np.newaxis]
        chances = chances_np(self._guess, x)[0].astype(np.float64)
        if self.config.guess_temperature != 1.0:
            chances = chances ** (1.0 / self.config.guess_temperature)
        return chances.tolist()


def make(seed: int, rules: RulesConfig, net: str = "../bitter-search/net.npz", guess: str = "guess.npz",
         **options: str) -> GuessSearch:  # fmt: skip
    config = configure(HybridConfig(rules=rules), {"top_k": "2", "samples": "256", **options}, "guess-search")
    paths = [Path(p) if Path(p).is_absolute() else HERE / p for p in (net, guess)]
    return GuessSearch(seed, config, str(paths[0]), str(paths[1]))
