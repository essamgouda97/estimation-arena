"""What readers see for each bot: a name and an explanation that need no knowledge of the code.

Kept here, not in the manifests or anchors.json: a manifest is part of a bot's hash, and changing
the hash would drop its games against the benchmarks. Code names stay in results and logs.
"""

from __future__ import annotations

from typing import Any

from .bots import SUB_PREFIX

_RL = "rl:checkpoint=runs/m4-rules-v2/checkpoints/final.pt"
_BASIC_BID = ["bid the tricks the hand should win", "bid"]
_BEST = ["play the card with the best average score", "pick"]

BOTS: dict[str, dict[str, Any]] = {
    "random": {
        "name": "Random",
        "about": "Plays any legal move. It's the bottom of the scale.",
        "method": [["make any legal bid", "bid"], ["list the legal cards", "rule"], ["pick one at random", "pick"]],
    },
    "heuristic": {
        "name": "Basic rules",
        "about": "Follows a short list of rules a person wrote.",
        "method": [_BASIC_BID, ["still need tricks? win as cheaply as possible", "rule"], ["no? throw the lowest card", "rule"], ["play that card", "pick"]],
    },
    "pimc": {
        "name": "Look-ahead",
        "about": "Imagines what the others might hold, plays each card to the end of the round using Basic rules, and picks what scored best.",
        "method": [_BASIC_BID, ["imagine the hidden hands x30", "deal"], ["try each card: Basic rules plays out the round", "rollout"], _BEST],
    },
    "search": {
        "name": "Look-ahead, tuned rules",
        "about": "Like Look-ahead, but it plays each card out with rules tuned for that job.",
        "method": [_BASIC_BID, ["imagine the hidden hands", "deal"], ["try each card: a tuned rule plays out the round", "rollout"], _BEST],
    },
    _RL: {
        "name": "Trained bot",
        "about": "Two trained models, one for bidding and one for cards. It started by copying Look-ahead, then improved by playing itself.",
        "method": [["learned from thousands of practice games", "data"], ["bidding model rates every bid", "bid"], ["card model rates every legal card", "net"], ["play its top card", "pick"]],
    },
    "hybrid": {
        "name": "Trained bot + look-ahead",
        "about": "Trained bot's models choose which cards are worth considering. It then tries each one on imagined hands.",
        "method": [["bidding model picks the bid", "bid"], ["card model keeps its top 3 cards", "net"], ["imagine hands that fit everyone's bids x16", "deal"], ["try each card: the models play out the round", "rollout"], _BEST],
    },
    "lowcard-wire": {
        "name": "Always lowest card",
        "about": "A simple example bot. It never bids and always plays its lowest card.",
        "method": [["never bid; call the minimum", "bid"], ["list the legal cards", "rule"], ["play the lowest", "pick"]],
    },
    "highcard-wire": {
        "name": "Always highest card",
        "about": "A simple example bot. It never bids and always plays its highest card.",
        "method": [["never bid; call the minimum", "bid"], ["list the legal cards", "rule"], ["play the highest", "pick"]],
    },
    "cautious-heuristic": {
        "name": "Cautious rules",
        "about": "Basic rules, set to take fewer risks.",
        "method": [_BASIC_BID, ["last to call? avoid the risky number", "rule"], ["still need tricks? win cheaply, else throw low", "rule"], ["play that card", "pick"]],
    },
    "table-pressure": {
        "name": "Rules + trump count",
        "about": "Basic rules, except it stops chasing tricks once its unbeatable trumps cover what it still needs.",
        "method": [_BASIC_BID, ["count my trumps nobody can beat", "rule"], ["they cover the tricks I need? throw low now", "rule"], ["otherwise play like Basic rules", "pick"]],
    },
    "pimc-clone": {
        "name": "Copycat",
        "about": "A trained model that learned its card play by copying Look-ahead. It bids with Basic rules and does no look-ahead itself.",
        "method": [_BASIC_BID, ["card model rates every legal card", "net"], ["play its top card", "pick"]],
    },
    "net-pimc": {
        "name": "Copycat + look-ahead",
        "about": "Copycat's model picks the most promising cards, so Look-ahead can spend its effort trying only those.",
        "method": [_BASIC_BID, ["card model keeps its top 3 cards", "net"], ["imagine the hidden hands x50", "deal"], ["try each card: Basic rules plays out the round", "rollout"], _BEST],
    },
    "rl-cards": {
        "name": "Copycat with practice",
        "about": "Copycat's model, improved further by playing itself. It bids with Basic rules.",
        "method": [_BASIC_BID, ["card model rates every legal card", "net"], ["play its top card", "pick"]],
    },
    "rl-pimc": {
        "name": "Copycat with practice + look-ahead",
        "about": "The practised model picks its 2 best cards, then Look-ahead tries them on imagined hands.",
        "method": [_BASIC_BID, ["card model keeps its top 2 cards", "net"], ["imagine the hidden hands x50", "deal"], ["try each card: Basic rules plays out the round", "rollout"], _BEST],
    },
    "llm-claude-api": {
        "name": "Claude Opus 5.5",
        "about": "Anthropic's general AI model, not built for this game. For each move it reads the rules and its own view of the table in words, then picks one. It has no tools and remembers nothing between moves.",
        "method": [["put the rules and this seat's view into words", "data"], ["Claude thinks it over and picks a move", "net"], ["play that move", "pick"]],
    },
    "hybrid-v4": {
        "name": "Hybrid v4",
        "about": "One model bids, one chooses cards, and one guesses who holds the hidden cards. It tries its top 2 cards on 32 imagined deals built from those guesses.",
        "method": [["the bid model picks the bid", "bid"], ["the card model keeps its top 2 cards", "net"], ["imagine 32 deals from where the guess model thinks the cards are", "deal"], ["play each kept card to the end", "rollout"], ["play the best average score", "pick"]],
    },
    "rl-hybrid": {
        "name": "Hybrid with Copycat cards",
        "about": "Trained bot + look-ahead with its card model swapped for Copycat with practice. It keeps the original bidding model.",
        "method": [["Trained bot's bidding model picks the bid", "bid"], ["Copycat with practice keeps its top 3 cards", "net"], ["imagine hands that fit everyone's bids x16", "deal"], ["try each card: the card model plays out the round", "rollout"], _BEST],
    },
    "bitter-net": {
        "name": "Self-taught",
        "about": "One trained model makes every decision. It began by playing randomly and learned only by playing itself, with no strategy written in and nothing copied.",
        "method": [["learned from millions of practice decisions", "data"], ["the model rates every legal move", "net"], ["play its top move", "pick"]],
    },
    "bitter-search": {
        "name": "Self-taught + look-ahead",
        "about": "The Self-taught model bids alone and picks its best cards, then tries each one on imagined hands.",
        "method": [["the model picks the bid", "bid"], ["the model keeps its top 3 cards", "net"], ["imagine the hidden hands x16", "deal"], ["try each card: the model plays out the round", "rollout"], _BEST],
    },
    "kooz-bot": {
        "name": "Kooz bot",
        "about": "Look-ahead on models trained from scratch. It imagines 64 deals from its own guess model, tries every legal card, and only makes a Dash Call when its bid model rates it clearly worth it.",
    },
    "guess-search": {
        "name": "Self-taught + guesses",
        "about": "Self-taught + look-ahead with a second model of its own that guesses who holds each hidden card. It learned those guesses only from its own games, and imagines many more hands from them.",
        "method": [["the model picks the bid", "bid"], ["the model keeps its top 2 cards", "net"], ["imagine 256 hands from where its guess model thinks the cards are", "deal"], ["try each card: the model plays out the round", "rollout"], _BEST],
    },
}

# Who made each bot, spelled one way. A manifest is part of a bot's hash, so names are tidied here.
_AUTHORS = {"Essam": "Essam Gouda"}


def author_name(author: str) -> str:
    return _AUTHORS.get(author, author)


def base_key(key: str) -> str:
    """The merged bot a test version belongs to: ``sub:bot:variant`` gives ``sub:bot``."""
    return SUB_PREFIX + _bot_id(key) if key.startswith(SUB_PREFIX) else key


# Test versions of a bot, as they appear in its key after the bot's own name.
_VARIANTS = {
    "run2-final": "",
    "run3-final": "trained to come first and nothing else",
    "rollout=net": "",
    "rollout=cards": "plays out with Copycat's model",
    "beliefs=false": "ignores what the bids reveal",
}
_BY_CODE_NAME = {"RL v2": _RL, "Hybrid (B-Tig)": "hybrid", "PIMC": "pimc", "Search bot": "search", "Heuristic": "heuristic", "Random": "random"}


def _bot_id(key: str) -> str:
    return key[len(SUB_PREFIX) :].split(":")[0] if key.startswith(SUB_PREFIX) else key


def about_bot(key: str) -> dict[str, Any]:
    return BOTS.get(_bot_id(key), {})


def plain_name(key: str, fallback: str = "") -> str:
    """A bot's name for readers, with what makes a test version different."""
    entry = about_bot(key)
    if not entry:
        return fallback or key
    parts = key[len(SUB_PREFIX) :].split(":")[1:] if key.startswith(SUB_PREFIX) else []
    notes = [note for note in (_VARIANTS.get(part, f"test: {part}") for part in parts) if note]
    return entry["name"] + (f" ({', '.join(notes)})" if notes else "")


def benchmark_name(code_name: str) -> str:
    return plain_name(_BY_CODE_NAME.get(code_name, code_name), code_name)
