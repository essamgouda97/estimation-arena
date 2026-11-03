"""Rung 1 of the ladder plan: a scripted bot that beats HeuristicBot.

HeuristicBot estimates the tricks its hand is worth and ignores the table. But the 13 tricks are
shared: if the estimates before mine already claim most of them (Over), tricks will be scarce and
my hand will take fewer than it is "worth"; if they claim few (Under), spare tricks get pushed
onto everyone. This bot shades its estimate by that pressure:

    implied = (estimates made so far) + (my expected tricks) + (seats still to estimate) x 3.25
    target  = expected + shift - pressure x (implied - 13) / 4

``shift`` corrects any bias in the hand evaluator, ``pressure`` is how strongly to react to the
table, ``risk`` is HeuristicBot's risk aversion for the last estimator. Bidding and card play are
HeuristicBot's, unchanged, so any gain comes from the estimates alone. The defaults are the
values the sweep picked (see experiments/rung1.md); every one can be overridden from a lineup:

    sub:table-pressure:pressure=0.5:shift=-0.25:risk=0.5
"""

from estimation.bots.hand_eval import trump_values
from estimation.bots.heuristic import HeuristicBot, HeuristicConfig, pick_estimate
from estimation.engine import Observation, RulesConfig, estimate_action
from estimation.engine.actions import is_bid, is_estimate

DEFAULTS = {"shift": 0.0, "pressure": 0.0, "risk": 0.25}
FAIR_SHARE = 13 / 4


class TablePressureBot(HeuristicBot):
    __slots__ = ("pressure", "shift")

    def __init__(self, config: HeuristicConfig, shift: float, pressure: float) -> None:
        super().__init__(config)
        self.shift = shift
        self.pressure = pressure

    def estimate(self, obs: Observation, legal: list[int]) -> int:
        values = trump_values(obs.hand, self.config.values)
        fixed = obs.fixed_trump
        if fixed is not None and any(is_bid(action) for action in legal):
            call = self._super_call(values, fixed, legal)
            if call >= 0:
                return call
        trump = obs.trump
        if trump is None:
            raise ValueError("the trump is known before anyone estimates")
        options = [action - estimate_action(0) for action in legal if is_estimate(action)]
        known = [e for seat, e in enumerate(obs.estimates) if e is not None and seat != obs.seat]
        unknown = 3 - len(known)
        expected = values[trump] + self.shift
        implied = sum(known) + expected + unknown * FAIR_SHARE
        target = expected - self.pressure * (implied - 13) / 4
        total_before = sum(known) if obs.last_estimator == obs.seat else None
        k = pick_estimate(target, options, total_before, self.config.risk_aversion)
        return estimate_action(k)


def make(seed: int, rules: RulesConfig, **options: str) -> TablePressureBot:
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"table-pressure has no option {sorted(unknown)}; it has {sorted(DEFAULTS)}")
    o = {key: float(options.get(key, value)) for key, value in DEFAULTS.items()}
    config = HeuristicConfig(super_call_min=rules.super_call_min, risk_aversion=o["risk"])
    return TablePressureBot(config, shift=o["shift"], pressure=o["pressure"])
