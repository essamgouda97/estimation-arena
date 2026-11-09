"""How each bot on the board came to play as it does: for the live board's bot panels.

Kept here, not in the manifests, because a manifest is part of a bot's hash and changing the hash
would drop its games against the anchors. Every number comes from the trainer's code or the
experiment notes (train/, jaxenv/selfplay.py, experiments/, the engine's estimation.rl).

An entry has a ``summary``, then ``stages`` in order. A stage draws as a strip of steps; a
``repeat`` draws its loop back to the first step. ``curves`` names a TensorBoard run.
"""

from __future__ import annotations

from typing import Any

_RULES = "Nothing is trained. A person wrote every rule."
_NOT_TRAINED = {"summary": _RULES, "stages": []}

_IMITATE = {
    "title": "Copy a strong player",
    "steps": [
        ["Look-ahead plays 802 games; its 602,000 decisions are saved", "data"],
        ["the model guesses which card it chose", "net"],
        ["check the guess against its real choice", "reward"],
        ["adjust the model a little", "update"],
    ],
    "repeat": "12 times through all the games",
}
_RL_CARDS = {
    "title": "Then improve by practising",
    "steps": [
        ["play 512 rounds: half against itself, half against its starting version", "play"],
        ["reward: its round score minus the other three's", "reward"],
        ["adjust the model, staying close to where it started", "update"],
    ],
    "repeat": "650 times",
}
_RL_V2 = [
    {
        "title": "Copy Look-ahead's card play",
        "steps": [
            ["four Look-ahead bots play full games, every card saved", "data"],
            ["the card model learns their choices", "net"],
            ["adjust the model a little", "update"],
        ],
        "repeat": "up to 24 times through the games",
    },
    {
        "title": "Then practise, with the bidding model starting from nothing",
        "steps": [
            ["play itself, recent versions, Basic rules and a lighter Look-ahead", "play"],
            ["reward: how that round went, for every decision in it", "reward"],
            ["each model learns what its choices were worth", "update"],
            ["every 5 rounds, a copy joins the opponents", "data"],
        ],
        "repeat": "once per batch of games",
    },
]
_BITTER = {
    "title": "Practice from nothing",
    "steps": [
        ["4,096 games at once: half against itself, half against a past version", "play"],
        ["128 decisions per game, then pause to learn", "data"],
        ["reward only when the game ends: King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1", "reward"],
        ["a coach who can see every hand judges each move", "net"],
        ["adjust the model; every 25 rounds a copy joins the 16 past versions", "update"],
    ],
    "repeat": "5,000 times, about 4.6 hours",
}

TRAINING: dict[str, dict[str, Any]] = {
    "llm-claude-api": {
        "summary": "Nothing here is trained on this game. It is Anthropic's general model, given the rules in words at every move.",
        "stages": [],
    },
    "hybrid-v4": {
        "summary": "Three models, for bids, cards and guessing who holds each hidden card. No human games were used.",
        "stages": [
            {
                "title": "Practice",
                "steps": [
                    ["play itself", "play"],
                    ["each model learns from those games", "update"],
                ],
            },
            {
                "title": "Then copy a deeper search",
                "steps": [
                    ["a deeper search plays", "data"],
                    ["the models learn its choices", "net"],
                    ["adjust the models", "update"],
                ],
            },
        ],
    },
    "random": {"summary": "Nothing is trained. It plays a random legal move.", "stages": []},
    "heuristic": _NOT_TRAINED,
    "pimc": {"summary": "Nothing is trained. It works things out during play, using Basic rules on imagined hands.", "stages": []},
    "search": {"summary": "Nothing is trained. It works things out during play, using rules a person wrote.", "stages": []},
    "rl:checkpoint=runs/m4-rules-v2/checkpoints/final.pt": {
        "summary": "Two models: the card model starts by copying Look-ahead, then both learn by playing.",
        "stages": _RL_V2,
    },
    "hybrid": {
        "summary": "The look-ahead isn't trained. It uses Yosif's two models, trained like this.",
        "stages": _RL_V2,
    },
    "lowcard-wire": _NOT_TRAINED,
    "highcard-wire": _NOT_TRAINED,
    "cautious-heuristic": _NOT_TRAINED,
    "table-pressure": _NOT_TRAINED,
    "pimc-clone": {"summary": "A model that only copies Look-ahead's card play. It bids with Basic rules.", "stages": [_IMITATE], "curves": "full-512x3"},
    "net-pimc": {
        "summary": "The look-ahead isn't trained. Its model only copies Look-ahead, then picks which cards to try.",
        "stages": [_IMITATE],
        "curves": "full-512x3",
    },
    "rl-cards": {
        "summary": "Card play only. It starts by copying Look-ahead, then improves by playing. It bids with Basic rules.",
        "stages": [_IMITATE, _RL_CARDS],
        "curves": "rl-cards-cards2",
    },
    "rl-pimc": {
        "summary": "The look-ahead isn't trained. It uses the model from Copycat with practice, trained like this.",
        "stages": [_IMITATE, _RL_CARDS],
        "curves": "rl-cards-cards2",
    },
    "rl-hybrid": {
        "summary": "The look-ahead isn't trained. Its card model comes from Copycat with practice; its bidding model is Yosif's.",
        "stages": [_IMITATE, _RL_CARDS],
        "curves": "rl-cards-cards2",
    },
    "bitter-net": {
        "summary": "One model for every decision, starting from nothing. Nothing copied and no rules written in.",
        "stages": [_BITTER],
        "curves": "bitter-run2",
        "lab": True,
    },
    "bitter-search": {
        "summary": "The look-ahead isn't trained. It uses the Self-taught model, trained like this.",
        "stages": [_BITTER],
        "curves": "bitter-run2",
        "lab": True,
    },
    "kooz-bot": {
        "summary": "The look-ahead isn't trained. It uses three models of its own, for bids, cards and guessing who holds each hidden card, trained from scratch by playing itself, with no human games.",
        "stages": _RL_V2,
    },
}


def training_of(key: str) -> dict[str, Any] | None:
    """A rating key's entry: anchors by spec, submissions by bot name."""
    name = key[4:].split(":")[0] if key.startswith("sub:") else key
    return TRAINING.get(name)
