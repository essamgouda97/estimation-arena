"""A live table: one Estimation game where each seat is a person or a bot.

A person's seat is a WebSocket client: the browser page, or any program that speaks the same
JSON (so an agent on another machine can take a seat as well). A bot seat runs here, from any
spec the arena knows (``heuristic``, ``hybrid``, ``sub:rl-hybrid``, ...). Every client gets its
own view: the public state, plus its own hand and legal actions when it holds a seat.

The table never shows a seat another seat's cards. Invite links carry a per-seat secret; the
spectator link carries none and sees no hands.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from estimation.engine import Bola, Phase, cards_of
from estimation.engine.actions import (
    DASH_CALL, ESTIMATE_BASE, NO_DASH_CALL, PASS, PLAY_BASE, WITH, bid_of, is_bid, is_estimate, is_play,
)  # fmt: skip
from estimation.engine.config import DEFAULT_CONFIG

from .bots import maker

TRUMPS = ("♣", "♦", "♥", "♠", "NT")
RANKS = "23456789TJQKA"
PHASES = {Phase.DASH: "dash", Phase.AUCTION: "auction", Phase.ESTIMATION: "estimate", Phase.PLAY: "play", Phase.DONE: "done"}
BOT_PAUSE_S = 0.7  # between bot moves, so people can follow
TRICK_PAUSE_S = 1.4  # a finished trick stays on the table this long
ROUND_PAUSE_S = 3.0  # the round's result stays up this long


def card_label(card: int) -> str:
    return RANKS[card % 13].replace("T", "10") + TRUMPS[card // 13]


def action_label(action: int) -> str:
    if action == NO_DASH_CALL:
        return "No Dash"
    if action == DASH_CALL:
        return "Dash Call"
    if is_bid(action):
        tricks, trump = bid_of(action)
        return f"{tricks} {TRUMPS[trump]}"
    if action == WITH:
        return "With"
    if action == PASS:
        return "Pass"
    if is_estimate(action):
        return f"Estimate {action - ESTIMATE_BASE}"
    if is_play(action):
        return card_label(action - PLAY_BASE)
    return str(action)


@dataclass
class Seat:
    kind: str  # "human" or "bot"
    name: str
    spec: str = ""  # bot spec
    token: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    bot: Any = None
    clients: int = 0


@dataclass
class Table:
    id: str
    seats: list[Seat]
    seed: int
    bola: Bola = field(init=False)
    clients: set = field(default_factory=set)  # (websocket, seat index or None)
    events: list[dict[str, Any]] = field(default_factory=list)  # recent moves, newest last
    pause_until: float = 0.0
    last_trick: dict[str, Any] | None = None
    last_round: dict[str, Any] | None = None
    games: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created: float = field(default_factory=time.time)
    _runner: asyncio.Task | None = None

    def __post_init__(self) -> None:
        self.new_game(self.seed)

    def new_game(self, seed: int) -> None:
        self.seed = seed
        self.bola = Bola(seed, DEFAULT_CONFIG)
        self.events, self.last_trick, self.last_round = [], None, None
        self.games += 1
        for k, seat in enumerate(self.seats):
            if seat.kind == "bot":
                if seat.bot is not None and hasattr(seat.bot, "close"):
                    seat.bot.close()
                seat.bot = maker(seat.spec, seed * 4 + k, DEFAULT_CONFIG)  # one bot per game

    def seat_of(self, token: str | None) -> int | None:
        if not token:
            return None
        return next((k for k, s in enumerate(self.seats) if s.kind == "human" and secrets.compare_digest(s.token, token)), None)

    # ---- moves -------------------------------------------------------------------------
    def _step(self, seat: int, action: int) -> None:
        bola = self.bola
        rnd = bola.round
        before = rnd.observation(seat)
        tricks_before, round_no = len(before.tricks), before.round_no
        bola.step(action)
        self.events = (self.events + [{"seat": seat, "label": action_label(action), "round": round_no}])[-40:]
        now = time.monotonic()
        tricks = rnd.observation(seat).tricks  # rnd is the round just played, even if it ended
        if len(tricks) != tricks_before:
            t = tricks[-1]
            self.last_trick = {"cards": [[(t.leader + i) & 3, c] for i, c in enumerate(t.cards)], "winner": t.winner}
            self.pause_until = now + TRICK_PAUSE_S
        else:
            self.pause_until = max(self.pause_until, now + BOT_PAUSE_S)
        if bola.round is not rnd or bola.is_over:
            result = bola.history[-1]
            self.last_round = {
                "round": result.round_no,
                "skipped": result.skipped,
                "seats": [{"estimate": r.estimate, "tricks": r.tricks, "won": r.won, "score": sc}
                          for r, sc in zip(result.seats, result.scores)],
            }
            self.last_trick = None if result.skipped else self.last_trick
            self.pause_until = now + ROUND_PAUSE_S

    async def play_human(self, seat: int, action: int) -> str | None:
        async with self.lock:
            bola = self.bola
            if bola.is_over:
                return "the game is over"
            if bola.current_player != seat:
                return "not your turn"
            if action not in bola.legal_actions():
                return "that move is not legal"
            self._step(seat, action)
        await self.broadcast()
        self.kick()
        return None

    def kick(self) -> None:
        """Run the bots until a person must move (or the game ends)."""
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run_bots())

    async def _run_bots(self) -> None:
        while True:
            bola = self.bola
            if bola.is_over:
                return
            seat = bola.current_player
            if seat is None or self.seats[seat].kind != "bot":
                return
            wait = self.pause_until - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            async with self.lock:
                if bola is not self.bola or bola.current_player != seat:
                    continue
                obs, legal = bola.observation(seat), bola.legal_actions()
                bot = self.seats[seat].bot
                action = legal[0] if len(legal) == 1 else await asyncio.to_thread(bot.act, obs, legal)
                if action not in legal:  # a broken bot must not stall the table
                    action = legal[0]
                self._step(seat, action)
            await self.broadcast()

    # ---- views -------------------------------------------------------------------------
    def view(self, seat: int | None) -> dict[str, Any]:
        bola = self.bola
        obs = bola.observation(seat if seat is not None else 0)
        rnd = bola.round
        hands = [len(cards_of(rnd.observation(k).hand)) for k in range(4)]
        names = [s.name for s in self.seats]
        bids = [{"seat": s, "label": action_label(a)} for s, a in obs.log if a < PLAY_BASE]
        me: dict[str, Any] | None = None
        if seat is not None:
            legal = bola.legal_actions() if bola.current_player == seat and not bola.is_over else []
            me = {
                "seat": seat,
                "hand": cards_of(obs.hand),
                "legal": [{"id": a, "label": action_label(a), "card": a - PLAY_BASE if is_play(a) else None} for a in legal],
            }
        return {
            "table": self.id,
            "game": self.games,
            "over": bola.is_over,
            "winners": list(bola.winners()) if bola.is_over else [],
            "round": obs.round_no,
            "phase": PHASES[obs.phase],
            "turn": bola.current_player,
            "carry": obs.carry,
            "scores": list(bola.scores),
            "trump": None if obs.trump is None else TRUMPS[obs.trump],
            "fixed_trump": None if obs.fixed_trump is None else TRUMPS[obs.fixed_trump],
            "caller": obs.caller,
            "top_bid": None if obs.top_bid is None else action_label(obs.top_bid),
            "top_bidder": obs.top_bidder,
            "dash": list(obs.dash_callers),
            "estimates": list(obs.estimates),
            "won": list(obs.tricks_won),
            "trick": [[s, c] for s, c in obs.trick],
            "last_trick": self.last_trick,
            "last_round": self.last_round,
            "bids": bids[-16:],
            "events": self.events[-12:],
            "hands": hands,
            "seats": [{"name": n, "kind": s.kind, "spec": s.spec, "online": s.clients > 0} for n, s in zip(names, self.seats)],
            "me": me,
        }

    async def broadcast(self) -> None:
        dead = []
        for ws, seat in list(self.clients):
            try:
                await ws.send_json({"type": "state", "state": self.view(seat)})
            except Exception:  # noqa: BLE001  (a closed socket is dropped)
                dead.append((ws, seat))
        for item in dead:
            self.clients.discard(item)
