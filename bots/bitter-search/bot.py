"""Search on top of the bitter-lesson net (bots/bitter-net), on Yosif's Hybrid search.

The net bids alone, picks the ``top_k`` cards to search and, with ``rollout=net``, plays all four
seats to the end of the round. The Hybrid search supplies the imagined deals and the score of each
card. ``rollout=cards`` plays them out with the rl-cards net instead, and ``beliefs=false`` keeps
the deals as drawn; RL v2's bid net is only used for those beliefs.

The net reads jaxenv's raw view, not the engine's RL features, so the rollout builds that view:
a per-seat template of what stays fixed during card play, and each step writes what changes.
The tests check it against jaxenv.encode_np at every decision of real games.

``objective=position`` scores each imagined round by the finishing position it leads to
(arena.lookahead) instead of by points; ``objective=standings`` does so with the standings net
saved beside the policy (jaxenv.selfplay) instead of a fixed guess at the rounds to come. ``bid_search=true`` looks ahead on the bid too: the net's
``bid_top_k`` favourite calls are each tried on ``bid_samples`` imagined deals, the net finishes
the bidding and the card play for all four seats, and the best average is called.

Options: ``net``, ``rollout``, ``objective`` and any HybridConfig field, e.g.
``sub:bitter-search:top_k=2``.
"""

from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401  (import order: the encoder imports the bots package)
from arena.lookahead import PositionScorer, StandingsScorer, rounds_after, skipped_round_value, standings_net
from estimation.bots.hybrid.beliefs import PublicBidding
from estimation.bots.hybrid.bot import HybridBot, HybridConfig
from estimation.bots.hybrid.rollout import Start, _beats, _only_card, bits
from estimation.bots.knowledge import read_knowledge
from estimation.bots.pimc import best_index
from estimation.bots.registry import configure
from estimation.bots.scoring import RoundScorer, estimates_of, standings_weights
from estimation.engine import Observation, Phase, RulesConfig, play_action
from estimation.engine.cards import SUIT_MASKS
from jaxenv.encode_np import PLAY_BASE, SIZE, encode, told_rounds_left

HERE = Path(__file__).resolve().parent
PLAYED, TABLE, SHOWN, WON, LEADER, MASK = 52, 260, 468, 580, 594, SIZE - 120
_ONE = np.uint64(1)
_SUITS = np.array(SUIT_MASKS, dtype=np.uint64)


@cache
def load_net(path: str, rounds_left: int = 0):
    """The policy at ``path`` for a round with ``rounds_left`` rounds after it: raw views in,
    a score per move out."""
    data = np.load(path)
    layers, k = [], 0
    while f"w{k}" in data:
        layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
        k += 1
    layers[0] = tuple(part.astype(np.float32) for part in told_rounds_left(*layers[0], rounds_left))
    head = data["pw"].astype(np.float32), data["pb"].astype(np.float32)

    def logits(x: np.ndarray) -> np.ndarray:
        for w, b in layers:
            x = np.maximum(x @ w + b, 0.0)
        return x @ head[0] + head[1]

    return logits


@cache
def load_cards(path: str):
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


def templates_of(obs: Observation) -> np.ndarray:
    """``(4, SIZE)``: each seat's raw view of ``obs``'s public state with an empty hand."""
    return np.stack([encode(replace(obs, seat=seat, hand=0, current_player=seat), []) for seat in range(4)])


def write_view(x, templates, hand, legal, played, voids, acting, trick_cards, tlen, leaders, won) -> None:
    """Write each game's acting seat's view into ``x``, exactly as jaxenv.encode_np would."""
    games = x.shape[0]
    ar = np.arange(games)
    np.take(templates, acting, axis=0, out=x)
    x[:, :52] = bits(hand)
    x[:, TABLE:SHOWN] = 0.0
    for i in range(tlen):
        rel = (leaders + i - acting) & 3
        x[ar, TABLE + 52 * rel + trick_cards[:, i]] = 1.0
    for r in range(4):
        seat = (acting + r) & 3
        x[:, PLAYED + 52 * r : PLAYED + 52 * (r + 1)] = bits(played[ar, seat])
        for suit in range(4):
            x[:, SHOWN + 4 * r + suit] = (voids[ar, seat] >> suit) & 1
        x[:, WON + r] = won[ar, seat] / 13
    x[:, LEADER : LEADER + 5] = 0.0
    x[ar, LEADER + ((leaders - acting) & 3)] = 1.0
    x[:, MASK : MASK + PLAY_BASE] = 0.0
    x[:, MASK + PLAY_BASE :] = bits(legal)


def play_out(net, templates, start: Start, hands, played, voids, first_cards) -> np.ndarray:
    """Every game to the end of the round, every seat playing the net's top legal card; each
    game's tricks won, ``(B, 4)``. As estimation.bots.hybrid.rollout.play_out, on the raw view.
    ``hands``, ``played`` and ``voids`` are used up: pass copies. ``first_cards`` is each game's
    first card, or None when the net plays from the start."""
    games = hands.shape[0]
    ar = np.arange(games, dtype=np.int64)
    trump, tlen, tricks_done = start.trump, len(start.trick), start.tricks_done
    trick_cards = np.zeros((games, 4), np.int64)
    best_card, best_at = -1, 0
    for i, c in enumerate(start.trick):
        trick_cards[:, i] = c
        if i == 0 or _beats(c, best_card, trump):
            best_card, best_at = c, i
    best = np.full(games, best_card, np.int64)
    best_pos = np.full(games, best_at, np.int64)
    leaders = np.full(games, start.leader, np.int64)
    won = np.tile(np.asarray(start.won, np.int64), (games, 1))
    buf = np.empty((games, SIZE), np.float32)
    card = None if first_cards is None else first_cards.astype(np.int64)
    while True:
        acting = (leaders + tlen) & 3
        hand = hands[ar, acting]
        led = trick_cards[:, 0] // 13
        if card is None:
            if tlen:
                follow = hand & _SUITS[led]
                legal = np.where(follow != 0, follow, hand)
            else:
                legal = hand
            card = _only_card(legal)
            rows = np.flatnonzero((legal & (legal - _ONE)) != 0)
            if rows.size:
                sub = buf[: rows.size]
                write_view(sub, templates, hand[rows], legal[rows], played[rows], voids[rows], acting[rows],
                           trick_cards[rows], tlen, leaders[rows], won[rows])  # fmt: skip
                values = net(sub)[:, PLAY_BASE:]
                card[rows] = np.where(bits(legal[rows]).astype(bool), values, -np.inf).argmax(axis=1)
        bit = np.left_shift(_ONE, card.astype(np.uint64))
        hands[ar, acting] = hand ^ bit
        played[ar, acting] |= bit
        if tlen:
            off = card // 13 != led
            voids[ar[off], acting[off]] |= np.left_shift(1, led[off]).astype(np.uint8)
            suit, best_suit = card // 13, best // 13
            beats = ((suit == best_suit) & (card > best)) | ((suit != best_suit) & (suit == trump))
            best = np.where(beats, card, best)
            best_pos = np.where(beats, tlen, best_pos)
        else:
            best = card.copy()
            best_pos = np.zeros(games, np.int64)
        trick_cards[:, tlen] = card
        card = None
        tlen += 1
        if tlen == 4:
            winners = (leaders + best_pos) & 3
            won[ar, winners] += 1
            leaders = winners
            tlen = 0
            tricks_done += 1
            if tricks_done == 13:
                return won


def finish_bidding(net, state: PublicBidding, hands: list[int]) -> None:
    """Step ``state`` to the end of the bidding, every seat making the net's top legal call from
    its hand in ``hands``."""
    while state.bidding:
        seat, legal = state.cur, state.legal_actions()
        action = legal[0]
        if len(legal) > 1:
            logits = net(encode(replace(state.observation(seat), hand=hands[seat]), legal))
            action = max(legal, key=lambda a: logits[a])
        state.step(seat, action)


def play_round(net, view: Observation, hands: list[int]) -> list[int]:
    """Tricks won by each seat when the net plays the whole round in ``view`` from ``hands``."""
    start = Start(trick=[], leader=view.leader, won=(0, 0, 0, 0), tricks_done=0,
                  estimates=estimates_of(view), trump=view.trump)  # fmt: skip
    won = play_out(net, templates_of(view), start, np.array([hands], dtype=np.uint64),
                   np.zeros((1, 4), np.uint64), np.zeros((1, 4), np.uint8), None)  # fmt: skip
    return won[0].tolist()


class _CardsPolicy:
    def __init__(self, base, card_values) -> None:
        self.bidding = base.bidding
        self.bid_values = base.bid_values
        self.card_values = card_values


class BitterSearch(HybridBot):
    def __init__(self, seed: int, config: HybridConfig, net: str, rollout: str, objective: str = "score") -> None:
        if objective not in ("score", "position", "standings"):
            raise ValueError(f"objective must be score, position or standings, not {objective!r}")
        super().__init__(seed, config)
        self._net_path = net
        self._for_position = objective != "score"
        self._standings = standings_net(net) if objective == "standings" else None  # learned beside the policy
        self._own_rollout = rollout == "net"
        if not self._own_rollout:
            self._policy = _CardsPolicy(self._policy, load_cards(str(HERE / "../rl-cards/net.npz")))

    def net(self, obs: Observation):
        """The policy for the round in ``obs``, knowing how many rounds follow it."""
        return load_net(self._net_path, rounds_after(obs.round_no))

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        logits = self.net(obs)(encode(obs, legal))
        if obs.phase is not Phase.PLAY:
            if self.config.bid_search:
                return self.best_call(obs, sorted(legal, key=lambda a: -logits[a])[: self.config.bid_top_k])
            return max(legal, key=lambda a: logits[a])
        cards = [a - PLAY_BASE for a in legal]
        search = self.search(obs, cards, [float(logits[PLAY_BASE + c]) for c in cards])
        return play_action(search.cards[best_index(search.means)])

    def best_call(self, obs: Observation, calls: list[int]) -> int:
        """The best of ``calls`` (the net's favourite first) on average over imagined deals."""
        if len(calls) == 1:
            return calls[0]
        return calls[best_index(self.try_calls(obs, calls, self.sample_deals(obs, self.config.bid_samples)))]

    def try_calls(self, obs: Observation, calls: list[int], deals: list[list[int]]) -> list[float]:
        """For each call, the sum over ``deals`` of what the round is worth to ``obs.seat`` after
        making it, the net finishing the bidding and playing the cards for every seat."""
        me, rules, net = obs.seat, self.config.rules, self.net(obs)
        before = PublicBidding(obs, rules)
        for seat, action in obs.log:
            before.step(seat, action)
        weights = standings_weights(obs.scores, me) if self.config.leader_weight else None
        totals = [0.0] * len(calls)
        for hands in deals:
            for k, call in enumerate(calls):
                state = before.clone()
                state.step(me, call)
                finish_bidding(net, state, hands)
                if state.phase is not Phase.PLAY:  # the round was skipped: nobody scores
                    totals[k] += skipped_round_value(obs) if self._for_position else 0.0
                    continue
                view = state.observation(me)
                totals[k] += self.scorer(view, weights).value(play_round(net, view, hands))
        return totals

    def scorer(self, obs: Observation, weights):
        """What a finished round is worth: finishing position, or points against the others."""
        if self._standings:
            return StandingsScorer(obs, self.config.rules, self._standings)
        return PositionScorer(obs, self.config.rules) if self._for_position else RoundScorer(obs, self.config.rules, weights)

    def explain(self, obs: Observation, cards: list[int]) -> tuple[int, dict[str, dict[int, float]]]:
        """The card it plays and why: the net's probability per legal card and the search's mean
        value of the cards it searched (for "how it plays")."""
        logits = self.net(obs)(encode(obs, [PLAY_BASE + c for c in cards]))
        values = np.array([logits[PLAY_BASE + c] for c in cards])
        prob = np.exp(values - values.max())
        prob /= prob.sum()
        search = self.search(obs, cards, values.tolist())
        why = {"net": dict(zip(cards, prob.tolist()))}
        if search.deals:
            why["search"] = dict(zip(search.cards, search.means))
        return search.cards[best_index(search.means)], why

    def evaluate(self, obs: Observation, cards: list[int], deals: list[list[int]]) -> list[float]:
        if not self._own_rollout:
            return super().evaluate(obs, cards, deals)
        return self.card_values(obs, cards, deals).sum(axis=0).tolist()

    def card_values(self, obs: Observation, cards: list[int], deals: list[list[int]]) -> np.ndarray:
        """``(deals, cards)``: what the round is worth to us on each imagined deal after each
        card, the net playing every seat from there."""
        know = read_knowledge(obs)
        n_cards, n_deals = len(cards), len(deals)
        games = n_cards * n_deals  # game g plays deal g // n_cards with card g % n_cards
        start = Start(
            trick=[card for _, card in obs.trick],
            leader=obs.leader,
            won=obs.tricks_won,
            tricks_done=len(obs.tricks),
            estimates=estimates_of(obs),
            trump=obs.trump,
        )
        won = play_out(
            self.net(obs),
            templates_of(obs),
            start,
            np.repeat(np.array(deals, dtype=np.uint64), n_cards, axis=0),
            np.tile(np.array(know.played_by, dtype=np.uint64), (games, 1)),
            np.tile(np.array(know.voids, dtype=np.uint8), (games, 1)),
            np.tile(np.array(cards, dtype=np.int64), n_deals),
        )
        weights = standings_weights(obs.scores, obs.seat) if self.config.leader_weight else None
        scorer = self.scorer(obs, weights)
        return np.array([scorer.value(tricks) for tricks in won.tolist()]).reshape(n_deals, n_cards)


def make(seed: int, rules: RulesConfig, net: str = "net.npz", rollout: str = "net", objective: str = "score",
         **options: str) -> BitterSearch:  # fmt: skip
    path = Path(net)
    config = configure(HybridConfig(rules=rules), options, "bitter-search")
    return BitterSearch(seed, config, str(path if path.is_absolute() else HERE / path), rollout, objective)
