"""Rung 4 screening bot: Yosif's Hybrid search with parts swapped.

    sub:mix-hybrid:prior=ours:roll=ours:bid=rl:samples=16

- ``prior``: which card net picks the ``top_k`` cards to search (``ours`` = bots/rl-cards, ``rl`` = RL v2).
- ``roll``: which card net plays every seat in the rollouts.
- ``bid``: ``rl`` (RL v2 BiddingNet, as the Hybrid) or ``heuristic``.
Every other option is a HybridConfig field (samples, oversample, beliefs, top_k, ...).
"""

from functools import cache
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401  (import order: the encoder imports the bots package)
from estimation.bots.heuristic import HeuristicBot, HeuristicConfig
from estimation.bots.hybrid.bot import HybridBot, HybridConfig
from estimation.bots.pimc import best_index
from estimation.bots.registry import configure
from estimation.engine import Observation, Phase, RulesConfig, play_action
from estimation.engine.actions import PLAY_BASE
from estimation.rl.encoder import encode

ROOT = Path(__file__).resolve().parents[3]
OUR_NET = ROOT / "bots" / "rl-cards" / "net.npz"


@cache
def our_net(path: str):
    data = np.load(path)
    layers, k = [], 0
    while f"w{k}" in data:
        layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
        k += 1

    def scores(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        for w, b in layers[:-1]:
            x = np.maximum(x @ w + b, 0.0)
        w, b = layers[-1]
        return x @ w + b

    return scores


class _Policy:
    """RL v2's BiddingNet with a chosen CardPlayNet stand-in for the rollouts."""

    def __init__(self, base, card_values) -> None:
        self.bidding = base.bidding
        self.bid_values = base.bid_values
        self.card_values = card_values


class MixHybrid(HybridBot):
    def __init__(self, seed, config, prior, roll, heuristic) -> None:
        super().__init__(seed, config)
        base = self._policy
        self._prior = base.card_values if prior == "rl" else our_net(str(OUR_NET))
        self._policy = _Policy(base, base.card_values if roll == "rl" else our_net(str(OUR_NET)))
        self._heuristic = heuristic

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        if obs.phase is not Phase.PLAY and self._heuristic is not None:
            return self._heuristic.act(obs, legal)
        x = encode(obs, self.config.rules)[np.newaxis]
        if obs.phase is not Phase.PLAY:
            values = self._policy.bid_values(x)[0]
            return max(legal, key=lambda action: values[action])
        advantage = self._prior(x)[0]
        cards = [action - PLAY_BASE for action in legal]
        search = self.search(obs, cards, [float(advantage[c]) for c in cards])
        return play_action(search.cards[best_index(search.means)])


def make(seed: int, rules: RulesConfig, prior: str = "ours", roll: str = "ours", bid: str = "rl", **options: str) -> MixHybrid:
    config = configure(HybridConfig(rules=rules), options, "mix-hybrid")
    heuristic = HeuristicBot(HeuristicConfig(super_call_min=rules.super_call_min)) if bid == "heuristic" else None
    return MixHybrid(seed, config, prior, roll, heuristic)
