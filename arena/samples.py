"""Sample moves for the leaderboard: every bot plays the same deal, a few of its card choices.

    uv run python -m arena.samples        # writes site/samples.json

Seat 0 is the bot, seats 1-3 HeuristicBot, deal ``SEED``. Every card decision with a choice in
the first round is kept: the bot's own view (hand, trick so far, trump, estimate) and the card it
played; the leaderboard shows the most telling one. Bots that can say why add scores per card:
``explain(obs, cards)`` for submissions that define it, the search values for PIMC-type anchors,
the card net's probabilities for RL v2, both for the Hybrid. Anchors whose files are missing (RL v2, Hybrid without runs/) are skipped, and samples
already in the file for them are kept.
"""

from __future__ import annotations

import json
import math
from typing import Any

import estimation.bots  # noqa: F401
from estimation.bots import HeuristicBot, PIMCBot
from estimation.bots.hybrid.bot import HybridBot
from estimation.bots.pimc import best_index
from estimation.bots.rl import RLBot
from estimation.rl.encoder import encode
from estimation.engine import Bola, Phase, cards_of
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG

from .bots import ROOT, SUB_PREFIX, list_submissions, maker
from .run import load_anchors

SEED = 31_337
OUT = ROOT / "site" / "samples.json"


def sample(spec: str) -> dict[str, Any]:
    bot = maker(spec, SEED, DEFAULT_CONFIG)
    others = HeuristicBot()
    bola = Bola(SEED)
    frames: list[dict[str, Any]] = []
    while not bola.is_over and bola.round_no == 1:
        seat = bola.current_player
        obs, legal = bola.observation(seat), bola.legal_actions()
        if seat != 0:
            bola.step(others.act(obs, legal))
            continue
        if obs.phase is not Phase.PLAY or len(legal) == 1:
            bola.step(bot.act(obs, legal))
            continue
        cards = [a - PLAY_BASE for a in legal]
        scores: dict[str, dict[int, float]] = {}
        if hasattr(bot, "explain"):
            chosen, scores = bot.explain(obs, cards)
        elif isinstance(bot, PIMCBot):
            cand, totals = bot.search(obs, cards)
            chosen = cand[best_index(totals)]
            n = getattr(bot.config, "samples", 1)
            scores = {"search": {c: t / n for c, t in zip(cand, totals)}}
        elif isinstance(bot, (HybridBot, RLBot)):
            z = bot._policy.card_values(encode(obs, DEFAULT_CONFIG)[None])[0]
            net = _softmax({c: float(z[c]) for c in cards})
            scores = {"net": net}
            if isinstance(bot, HybridBot):
                search = bot.search(obs, cards, [float(z[c]) for c in cards])
                chosen = search.cards[best_index(search.means)]
                if search.deals:
                    scores["search"] = dict(zip(search.cards, search.means))
            else:
                chosen = max(cards, key=lambda c: z[c])
        else:
            chosen = bot.act(obs, legal) - PLAY_BASE
        frames.append({
            "trump": obs.trump,
            "estimate": obs.estimates[0],
            "won": obs.tricks_won[0],
            "hand": cards_of(obs.hand),
            "trick": [[s, c] for s, c in obs.trick],
            "legal": cards,
            "chosen": chosen,
            "scores": {k: {str(c): round(v, 3) for c, v in d.items()} for k, d in scores.items()},
        })
        bola.step(PLAY_BASE + chosen)
    if hasattr(bot, "close"):
        bot.close()
    return {"seed": SEED, "frames": frames}


def _softmax(values: dict[int, float]) -> dict[int, float]:
    top = max(values.values())
    exp = {c: math.exp(v - top) for c, v in values.items()}
    total = sum(exp.values())
    return {c: e / total for c, e in exp.items()}


def main() -> int:
    old = json.loads(OUT.read_text()) if OUT.is_file() else {}
    specs = [SUB_PREFIX + n for n in list_submissions()] + [a["spec"] for a in load_anchors()]
    out = dict(old)
    for spec in specs:
        out[spec] = sample(spec)
        print(f"{spec}: {len(out[spec]['frames'])} moves")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
