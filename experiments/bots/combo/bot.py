"""Diagnostic bot: Dash Calls, bids and estimates from one bot, card play from another.

    sub:combo:bid=rl;checkpoint=runs/m4-rules-v2/checkpoints/final.pt:cards=sub;net-pimc

Specs use ``;`` where a lineup spec would use ``:`` (the option separator). Used to find out
whether a gap between two bots sits in their bidding or their card play.
"""

from estimation.engine import Observation, Phase, RulesConfig

from arena.bots import maker


class Combo:
    def __init__(self, bidder, player) -> None:
        self.bidder, self.player = bidder, player

    def act(self, obs: Observation, legal: list[int]) -> int:
        bot = self.player if obs.phase is Phase.PLAY else self.bidder
        return bot.act(obs, legal)


def make(seed: int, rules: RulesConfig, bid: str = "heuristic", cards: str = "heuristic") -> Combo:
    spec = lambda s: s.replace(";", ":")  # noqa: E731
    return Combo(maker(spec(bid), seed, rules), maker(spec(cards), seed + 1, rules))
