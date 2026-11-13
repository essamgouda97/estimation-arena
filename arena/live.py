"""Matches as they play, for the live board: each match keeps one small JSON file up to date.

A match writes only where the folder exists (the home server), so CI and laptops write nothing.
One file per bot, opponent and seed: replaying a match replaces its file.

The kind says what a match means to the ratings:

- rating: an arena.run match. Its results file takes over when the bot finishes.
- entry: a single arena.sweep match. These files are the only record, so the board keeps them.
- tuning: one row of a sweep grid. Shown while it plays, never rated.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from estimation.arena.duplicate import SetResult

from .bots import SUB_PREFIX
from .positions import set_outcomes, summarize

LIVE = Path(os.environ.get("ESTIMATION_LIVE", Path.home() / "workspace" / "estimation-data" / "live"))


def public(spec: str) -> tuple[str, str]:
    """The rating key and display name of a spec, without the paths in its options:
    ``sub:bitter-net:net=/data/run2-final.npz`` is ``sub:bitter-net:run2-final``, "run2-final"."""
    if not spec.startswith(SUB_PREFIX):
        return spec, spec
    bot, *options = spec[len(SUB_PREFIX) :].split(":")
    if any("/" in o for o in options):
        values = [Path(o.partition("=")[2]).stem for o in options]
        return ":".join([SUB_PREFIX + bot, *values]), " ".join(values)
    return spec, " ".join([bot, *options])


class Match:
    """Writes a match's progress after each set. Never lets a board problem stop the games."""

    def __init__(self, lineup: Sequence[str], sets: int, seed: int, kind: str = "rating") -> None:
        self.lineup = tuple(lineup)
        names = list(dict.fromkeys(lineup))
        self.path = None
        if len(names) != 2 or not LIVE.is_dir():
            return
        self.me, self.other = names
        key, name = public(self.me)
        digest = hashlib.sha1(f"{key}|{self.other}|{seed}".encode()).hexdigest()[:16]
        self.path = LIVE / f"{digest}.json"
        self.head = {"id": digest, "pid": os.getpid(), "kind": kind, "key": key, "name": name,
                     "opponent": self.other, "sets": sets, "seed": seed, "started_at": time.time()}  # fmt: skip
        self.update([])

    def update(self, results: Sequence[SetResult], finished: bool = False) -> None:
        if self.path is None:
            return
        data: dict[str, Any] = {**self.head, "done": len(results), "finished": finished, "updated_at": time.time()}
        if results:
            summary, pairs = summarize(results, self.lineup)
            mine = summary[self.me]
            data |= {
                "strip": set_outcomes(results, self.lineup, self.me),
                "shares": mine["shares"],
                "reward": mine["reward"],
                "king": mine["king"],
                "delta": mine["delta"],
                "pairs": {metric: [{**p, "a": public(p["a"])[0], "b": public(p["b"])[0]} for p in rows]
                          for metric, rows in pairs.items()},
            }  # fmt: skip
        try:
            part = self.path.with_suffix(f".{os.getpid()}.part")
            part.write_text(json.dumps(data))
            part.replace(self.path)
        except OSError:
            self.path = None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        pass
    return True


def read_matches() -> list[dict[str, Any]]:
    """Every match file, newest first, with its state: playing, done or stopped."""
    out = []
    for path in LIVE.glob("*.json") if LIVE.is_dir() else ():
        try:
            match = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        match["alive"] = _alive(match["pid"])
        match["state"] = "done" if match["finished"] else "playing" if match["alive"] else "stopped"
        out.append(match)
    return sorted(out, key=lambda m: -m["updated_at"])
