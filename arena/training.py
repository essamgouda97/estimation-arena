"""How each bot on the board came to play as it does: for the live board's bot panels.

Kept here, not in the manifests, because a manifest is part of a bot's hash and changing the hash
would drop its games against the anchors. Every number comes from the trainer's code or the
experiment notes (train/, jaxenv/selfplay.py, experiments/, the engine's estimation.rl).

An entry has a ``summary``, then ``stages`` in order. A stage draws as a strip of steps; a
``repeat`` draws its loop back to the first step. ``curves`` names a TensorBoard run.
"""

from __future__ import annotations

from typing import Any

_RULES = "Nothing is trained. Every decision comes from rules written by hand."
_NOT_TRAINED = {"summary": _RULES, "stages": []}

_IMITATE = {
    "title": "Copy a strong teacher",
    "steps": [
        ["PIMC-100 plays 802 games, 602k decisions recorded", "data"],
        ["the net guesses the teacher's card", "net"],
        ["compare with its card and its value of each card", "reward"],
        ["nudge the net's weights", "update"],
    ],
    "repeat": "12 passes over the games",
}
_RL_CARDS = {
    "title": "Then improve by playing itself",
    "steps": [
        ["play 512 rounds, half against itself, half against its starting net", "play"],
        ["reward: round score minus the other three's", "reward"],
        ["PPO update, kept near the starting net", "update"],
    ],
    "repeat": "650 times, in two runs of 280 and 370",
}
_RL_V2 = [
    {
        "title": "Copy PIMC's card play",
        "steps": [
            ["four PIMC bots play whole games, every card recorded", "data"],
            ["the card net learns PIMC's choices and values", "net"],
            ["nudge the weights", "update"],
        ],
        "repeat": "up to 24 passes over the games",
    },
    {
        "title": "Then self-play, with the bid net starting from nothing",
        "steps": [
            ["play against itself, recent versions, Heuristic and a light PIMC", "play"],
            ["reward: that round's result, for every decision in it", "reward"],
            ["each net learns the value of what it chose", "update"],
            ["every 5 rounds of this, a version joins the opponents", "data"],
        ],
        "repeat": "once per batch of games",
    },
]
_BITTER = {
    "title": "Self-play from nothing, on the GPU",
    "steps": [
        ["4,096 games at once: half itself, half a past version", "play"],
        ["128 decisions per game, then stop to learn", "deal"],
        ["reward only at the end of the game: King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1", "reward"],
        ["a critic that sees hidden hands judges each move", "net"],
        ["PPO update; every 25 rounds a version joins the 16 past ones", "update"],
    ],
    "repeat": "5,000 times, about 4.6 hours",
}

TRAINING: dict[str, dict[str, Any]] = {
    "random": {"summary": "Nothing is trained. It plays a random legal move.", "stages": []},
    "heuristic": _NOT_TRAINED,
    "pimc": {"summary": "Nothing is trained. It searches at play time, using Heuristic to play imagined rounds.", "stages": []},
    "search": {"summary": "Nothing is trained. It searches at play time, using hand-written rules.", "stages": []},
    "rl:checkpoint=runs/m4-rules-v2/checkpoints/final.pt": {
        "summary": "Yosif's two nets: card play starts by copying PIMC, then both nets learn from self-play.",
        "stages": _RL_V2,
    },
    "hybrid": {
        "summary": "The search isn't trained. It uses RL v2's two nets, trained like this.",
        "stages": _RL_V2,
    },
    "lowcard-wire": _NOT_TRAINED,
    "highcard-wire": _NOT_TRAINED,
    "cautious-heuristic": _NOT_TRAINED,
    "table-pressure": _NOT_TRAINED,
    "pimc-clone": {"summary": "A net that only copies PIMC-100's card play. It bids with Heuristic.", "stages": [_IMITATE], "curves": "full-512x3"},
    "net-pimc": {
        "summary": "The search isn't trained. Its net only copies PIMC-100, then picks which cards to search.",
        "stages": [_IMITATE],
        "curves": "full-512x3",
    },
    "rl-cards": {
        "summary": "Card play only. It starts as a copy of PIMC-100, then self-play improves it. It bids with Heuristic.",
        "stages": [_IMITATE, _RL_CARDS],
        "curves": "rl-cards-cards2",
    },
    "rl-pimc": {
        "summary": "The search isn't trained. It uses rl-cards' net, trained like this.",
        "stages": [_IMITATE, _RL_CARDS],
        "curves": "rl-cards-cards2",
    },
    "rl-hybrid": {
        "summary": "The search isn't trained. Its card net is rl-cards'; its bid net is RL v2's.",
        "stages": [_IMITATE, _RL_CARDS],
        "curves": "rl-cards-cards2",
    },
    "bitter-net": {
        "summary": "One net for every decision, starting from nothing. No copying, no hand-written rules.",
        "stages": [_BITTER],
        "curves": "bitter-run2",
        "lab": True,
    },
    "bitter-search": {
        "summary": "The search isn't trained. It uses bitter-net's net, trained like this.",
        "stages": [_BITTER],
        "curves": "bitter-run2",
        "lab": True,
    },
}


def training_of(key: str) -> dict[str, Any] | None:
    """A rating key's entry: anchors by spec, submissions by bot name."""
    name = key[4:].split(":")[0] if key.startswith("sub:") else key
    return TRAINING.get(name)
