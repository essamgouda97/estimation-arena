"""Rung 4: Yosif's Hybrid search, with our RL card net in place of RL v2's CardPlayNet.

The Hybrid (``estimation.bots.hybrid``) bids with RL v2's BiddingNet, samples hidden hands
weighted by how well they explain the other seats' bids, and plays each candidate card out
with CardPlayNet for all four seats. This bot keeps all of that and swaps one part: our card
net (``net``, trained by self-play PPO) picks the ``top_k`` cards to search and plays every seat
in the rollouts. Screening (experiments/rung4.md) showed the swap alone beats the Hybrid.

Options: ``net`` (path from this folder) and any HybridConfig field, e.g.
``sub:rl-hybrid:samples=32:top_k=2``.
"""

from functools import cache
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401  (import order: the encoder imports the bots package)
from estimation.bots.hybrid.bot import HybridBot, HybridConfig
from estimation.bots.pimc import best_index
from estimation.bots.registry import configure
from estimation.engine import Observation, Phase, RulesConfig, play_action
from estimation.engine.actions import PLAY_BASE
from estimation.rl.encoder import encode

HERE = Path(__file__).resolve().parent


@cache
def card_net(path: str):
    data = np.load(HERE / path)
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
    """RL v2's BiddingNet (bids and beliefs) with our card net for the rollouts."""

    def __init__(self, base, card_values) -> None:
        self.bidding = base.bidding
        self.bid_values = base.bid_values
        self.card_values = card_values


class RLHybrid(HybridBot):
    def __init__(self, seed: int, config: HybridConfig, net: str) -> None:
        super().__init__(seed, config)
        self._net = card_net(net)
        self._policy = _Policy(self._policy, self._net)

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        x = encode(obs, self.config.rules)[np.newaxis]
        if obs.phase is not Phase.PLAY:
            values = self._policy.bid_values(x)[0]
            return max(legal, key=lambda action: values[action])
        return play_action(self.explain(obs, [a - PLAY_BASE for a in legal], x)[0])

    def explain(self, obs: Observation, cards: list[int], x: np.ndarray | None = None):
        """The card it plays and why: the net's probability per legal card and the search's
        mean value of the cards it searched (for the leaderboard's sample moves)."""
        x = encode(obs, self.config.rules)[np.newaxis] if x is None else x
        z = self._net(x)[0]
        zl = np.array([z[c] for c in cards])
        prob = np.exp(zl - zl.max())
        prob /= prob.sum()
        search = self.search(obs, cards, [float(z[c]) for c in cards])
        chosen = search.cards[best_index(search.means)]
        why = {"net": dict(zip(cards, prob.tolist()))}
        if search.deals:
            why["search"] = dict(zip(search.cards, search.means))
        return chosen, why


def make(seed: int, rules: RulesConfig, net: str = "net.npz", **options: str) -> RLHybrid:
    return RLHybrid(seed, configure(HybridConfig(rules=rules), options, "rl-hybrid"), net)
