"""Rung 1 of the ladder plan: a scripted bot that beats HeuristicBot.

HeuristicBot estimates the tricks its hand is worth and ignores the table. But the 13 tricks are
shared: if the estimates before mine already claim most of them (Over), tricks will be scarce and
my hand will take fewer than it is "worth"; if they claim few (Under), spare tricks get pushed
onto everyone. This bot shades its estimate by that pressure:

    implied = (estimates made so far) + (my expected tricks) + (seats still to estimate) x 3.25
    target  = expected + shift - pressure x (implied - 13) / 4

``shift`` corrects any bias in the hand evaluator, ``pressure`` is how strongly to react to the
table, ``risk`` is HeuristicBot's risk aversion for the last estimator.

``sure=1`` changes card play in one way: HeuristicBot chases tricks until its estimate is reached,
even when it already holds enough unbeatable trumps to get there, and so takes extra tricks by
accident. With ``sure=1`` the bot counts its trumps above every trump still out; once they cover
the tricks it still needs, it plays to lose and lets those trumps win later. Bidding is
HeuristicBot's, unchanged. The defaults are the
values the sweep picked (see experiments/rung1.md); every one can be overridden from a lineup:

    sub:table-pressure:pressure=0.5:shift=-0.25:risk=0.5
"""

from estimation.bots.card_play import choose_card
from estimation.bots.hand_eval import trump_values
from estimation.bots.heuristic import HeuristicBot, HeuristicConfig, pick_estimate
from estimation.engine import SANS, Observation, RulesConfig, estimate_action, play_action
from estimation.engine.actions import is_bid, is_estimate
from estimation.engine.cards import FULL_DECK, SUIT_MASKS

DEFAULTS = {"shift": 0.0, "pressure": 0.0, "risk": 0.25, "sure": 1.0}
FAIR_SHARE = 13 / 4


class TablePressureBot(HeuristicBot):
    __slots__ = ("pressure", "shift", "sure")

    def __init__(
        self, config: HeuristicConfig, shift: float, pressure: float, sure: bool = False
    ) -> None:
        super().__init__(config)
        self.shift = shift
        self.pressure = pressure
        self.sure = sure

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


    def play(self, obs: Observation) -> int:
        if not self.sure:
            return super().play(obs)
        hand = obs.hand
        played = 0
        for record in obs.tricks:
            for card in record.cards:
                played |= 1 << card
        trick = [card for _, card in obs.trick]
        for card in trick:
            played |= 1 << card
        me = obs.seat
        estimate, trump = obs.estimates[me], obs.trump
        if estimate is None or trump is None:
            raise ValueError("estimates and trump are known before play")
        out = FULL_DECK & ~played & ~hand
        need = estimate - obs.tricks_won[me]
        want = need > 0 and sure_trumps(hand, trump, out) < need
        return play_action(choose_card(hand, trick, trump, want, out))


def sure_trumps(hand: int, trump: int, out: int) -> int:
    """Trumps in ``hand`` above every trump still out: tricks nobody can take from me."""
    if trump == SANS:
        return 0
    mine = hand & SUIT_MASKS[trump]
    theirs = out & SUIT_MASKS[trump]
    if theirs:
        mine &= ~((2 << (theirs.bit_length() - 1)) - 1)  # keep only cards above their top trump
    return mine.bit_count()


def make(seed: int, rules: RulesConfig, **options: str) -> TablePressureBot:
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"table-pressure has no option {sorted(unknown)}; it has {sorted(DEFAULTS)}")
    o = {key: float(options.get(key, value)) for key, value in DEFAULTS.items()}
    config = HeuristicConfig(super_call_min=rules.super_call_min, risk_aversion=o["risk"])
    return TablePressureBot(config, shift=o["shift"], pressure=o["pressure"], sure=o["sure"] > 0)
