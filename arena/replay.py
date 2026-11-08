"""Games people can watch: one public game per pairing, with every card, for the game browser.

    uv run python -m arena.replay due                 # pairings with no current game, best first
    uv run python -m arena.replay play KEY_A KEY_B    # play one and save it under site/games/

Rated games use a secret seed, and showing their cards would reveal those deals. So these games
are extra: the same bots, on public deals, and they never count towards a rating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from estimation.engine.bola import Bola
from estimation.engine.config import DEFAULT_CONFIG

from .bots import ROOT, SUB_PREFIX, load_manifest, maker
from .direct import game_file
from .leaderboard import SITE_DIR, build
from .run import bot_hash, engine_version

GAMES_DIR = Path("games")
PUBLIC_SEED = 20261107


def game_seed(a: str, b: str) -> int:
    first, second = sorted((a, b))
    return int.from_bytes(hashlib.sha256(f"{PUBLIC_SEED}|{first}|{second}".encode()).digest()[:4], "big")


def _version(key: str) -> str:
    return bot_hash(load_manifest(key[len(SUB_PREFIX) :].split(":")[0])) if key.startswith(SUB_PREFIX) else key


def _cards(mask: int) -> list[int]:
    return [card for card in range(52) if mask >> card & 1]


def record_game(seats: list[str], seed: int, rounds: int = 0, bots: list[Any] | None = None) -> dict[str, Any]:
    """Play one Bola with ``seats`` and return everything needed to watch it. ``bots`` replaces
    the ones it would make, for rebuilding a game that was already played."""
    bots = bots or [maker(spec, seed + seat, DEFAULT_CONFIG) for seat, spec in enumerate(seats)]
    game = Bola(seed, DEFAULT_CONFIG)
    played: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    while not game.is_over:
        seat = game.current_player
        view = game.observation(seat)
        if current is None or view.round_no != current["round"]:
            if current is not None:
                current["after"] = list(view.scores)
                if rounds and len(played) >= rounds:
                    current = None
                    break
            current = {"round": view.round_no, "before": list(view.scores), "carry": view.carry,
                       "fixed": view.fixed_trump, "trump": None, "caller": None, "with": [], "estimates": [None] * 4,
                       "hands": [_cards(game.observation(k).hand) for k in range(4)], "moves": []}  # fmt: skip
            played.append(current)
        if view.trump is not None:
            current["trump"] = view.trump
        if view.caller is not None:
            current["caller"] = view.caller
            current["with"] = [w[0] for w in view.with_players]
        current["estimates"] = [view.estimates[k] for k in range(4)]
        action = bots[seat].act(view, game.legal_actions())
        current["moves"].append([seat, action])
        game.step(action)
    final = list(game.scores)  # what play_bola scores the game by; the last observation can lag it
    if current is not None:
        current["after"] = final
    return {"seats": seats, "seed": seed, "rounds": played, "final": final, "finished": game.is_over}


def pairings(root: Path = ROOT) -> list[tuple[str, str]]:
    """Every rated bot against every other and every benchmark, best bots first."""
    rows = [r for r in build(root)["rows"] if r["elo"] is not None and not r.get("provisional")]
    return [(a["key"], b["key"]) for k, a in enumerate(rows) for b in rows[k + 1 :]
            if not (a["anchor"] and b["anchor"])]  # fmt: skip


def is_current(path: Path, a: str, b: str) -> bool:
    if not path.is_file():
        return False
    saved = json.loads(path.read_text())
    return saved.get("engine") == engine_version() and saved.get("versions") == {a: _version(a), b: _version(b)}


def names_of(root: Path = ROOT) -> dict[str, str]:
    return {row["key"]: row["code_name"] for row in build(root)["rows"]}


def due(root: Path = ROOT, out: Path = SITE_DIR) -> list[tuple[str, str]]:
    names = names_of(root)
    return [(a, b) for a, b in pairings(root) if not is_current(out / GAMES_DIR / game_file(names[a], names[b]), a, b)]


def play(a: str, b: str, root: Path = ROOT, out: Path = SITE_DIR) -> Path:
    game = record_game([a, b, a, b], game_seed(a, b))
    game |= {"engine": engine_version(), "versions": {a: _version(a), b: _version(b)}}
    path = out / GAMES_DIR / game_file(*(names_of(root)[k] for k in (a, b)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(game, separators=(",", ":")) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.replay", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("due", help="print the pairings to play, one per line, tab separated")
    one = commands.add_parser("play", help="play one pairing and print its file")
    one.add_argument("a")
    one.add_argument("b")
    args = parser.parse_args(argv)
    if args.command == "due":
        for a, b in due():
            print(f"{a}\t{b}")
        return 0
    print(play(args.a, args.b).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
