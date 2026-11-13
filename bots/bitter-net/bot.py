"""Bitter-lesson bot: one net for every decision, trained from nothing by self-play on the GPU.

No hand evaluation, no imitation, no search: jaxenv.encode_np's raw view in, the best legal
action out. Training: jaxenv.selfplay.
"""

from pathlib import Path

import numpy as np

from estimation.engine import Observation, RulesConfig
from jaxenv.encode_np import PLAY_BASE, encode

HERE = Path(__file__).resolve().parent


class BitterNet:
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.layers, k = [], 0
        while f"w{k}" in data:
            self.layers.append((data[f"w{k}"], data[f"b{k}"]))
            k += 1
        self.head = (data["pw"], data["pb"])

    def logits(self, obs: Observation, legal: list[int]) -> np.ndarray:
        x = encode(obs, legal)
        for w, b in self.layers:
            x = np.maximum(x @ w + b, 0.0)
        return x @ self.head[0] + self.head[1]

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        logits = self.logits(obs, legal)
        return max(legal, key=lambda a: logits[a])

    def explain(self, obs: Observation, cards: list[int]) -> tuple[int, dict[str, dict[int, float]]]:
        """The card it plays and the net's probability per legal card (for "how it plays")."""
        z = self.logits(obs, [PLAY_BASE + c for c in cards])
        values = np.array([z[PLAY_BASE + c] for c in cards])
        prob = np.exp(values - values.max())
        prob /= prob.sum()
        return cards[int(values.argmax())], {"net": dict(zip(cards, prob.tolist()))}


def make(seed: int, rules: RulesConfig, net: str = "net.npz") -> BitterNet:
    path = Path(net)
    return BitterNet(path if path.is_absolute() else HERE / path)
