"""Rung 3: net-pimc guided by the self-play RL card net (bots/rl-cards) instead of the imitation net.

Everything below is net-pimc's design; only the net (trained further by PPO self-play) and the
default width (top 2) differ.

Rung 2 design: a learned prior steers PIMC's search, AlphaZero style.

PIMC rolls every legal card out on the same N sampled deals: about 5 cards x 30 deals = 150
rollouts per decision for the PIMC anchor. Most of that is spent on cards that are clearly bad.
This bot asks a small net (trained by train/imitate.py on PIMC-100's decisions) which ``top``
cards look best, and rolls out only those, on ``samples`` deals. The default, top=3 and
samples=50, does the anchor's ~150 rollouts, but on the cards that matter.

On held-out data, the best card among the net's top 3 loses 0.16 points per decision against
PIMC-100's judgement; PIMC-30 loses 0.54 (experiments/rung2.md).

Options: ``sub:net-pimc:top=2:samples=75``. Bidding and estimates are HeuristicBot's, as PIMC's.
"""

from functools import cache
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401  (import order: the encoder imports the bots package)
from estimation.bots import PIMCBot, PIMCConfig
from estimation.bots.heuristic import HeuristicConfig
from estimation.bots.pimc import best_index
from estimation.engine import Observation, RulesConfig
from estimation.rl.encoder import encode

HERE = Path(__file__).resolve().parent


@cache
def load_net(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    data = np.load(HERE / path)
    layers, k = [], 0
    while f"w{k}" in data:
        layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
        k += 1
    return layers


def card_scores(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for w, b in layers[:-1]:
        x = np.maximum(x @ w + b, 0.0)
    w, b = layers[-1]
    return x @ w + b


class NetPIMCBot(PIMCBot):
    def __init__(self, seed: int, config: PIMCConfig, net: str, top: int) -> None:
        super().__init__(seed, config)
        self.layers = load_net(net)
        self.top = top

    def explain(self, obs: Observation, cards: list[int]) -> tuple[int, dict[str, dict[int, float]]]:
        """The card it plays and why: the net's probability per legal card, and the search's
        mean score for the 3 cards it kept (for the leaderboard's sample moves)."""
        z = card_scores(self.layers, encode(obs, self.config.rules).astype(np.float32))
        zl = np.array([z[c] for c in cards])
        prob = np.exp(zl - zl.max())
        prob /= prob.sum()
        candidates = sorted(cards, key=lambda c: -z[c])[: self.top]
        if len(candidates) == 1:
            return candidates[0], {"net": dict(zip(cards, prob.tolist()))}
        totals = self.evaluate(obs, candidates, self.sample_deals(obs))
        search = {c: t / self.config.samples for c, t in zip(candidates, totals)}
        return candidates[best_index(totals)], {"net": dict(zip(cards, prob.tolist())), "search": search}

    def choose(self, obs: Observation, cards: list[int]) -> int:
        scores = card_scores(self.layers, encode(obs, self.config.rules).astype(np.float32))
        candidates = sorted(cards, key=lambda c: -scores[c])[: self.top]
        if len(candidates) == 1:
            return candidates[0]
        totals = self.evaluate(obs, candidates, self.sample_deals(obs))
        return candidates[best_index(totals)]


def make(seed: int, rules: RulesConfig, net: str = "../rl-cards/net.npz", top: str = "2", samples: str = "50") -> NetPIMCBot:
    heuristic = HeuristicConfig(super_call_min=rules.super_call_min)
    config = PIMCConfig(samples=int(samples), heuristic=heuristic, rules=rules)
    return NetPIMCBot(seed, config, net, int(top))
