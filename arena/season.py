"""Seasons: a fresh start on fresh secret deals.

    uv run python -m arena.season now      # the season being played
    uv run python -m arena.season start    # end this season and start the next
    uv run python -m arena.season seed     # this season's deal seed, from LADDER_SEED (the ladder only)

Within a season every result stays until its bot or the engine changes. Over months that leaves
ratings resting on old games, all dealt from one seed. A new season keeps the final standings of
the old one as they ended, then makes every bot earn its rating again on deals nobody has played:
results carry their season in their engine tag (``/s2``), so everything that asks "is this result
still current" already treats last season's as stale. Season 1 has no tag.

Fixed benchmarks keep their games against each other across seasons. Those set the scale, and a
benchmark never changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bots import ROOT

SEASON_FILE = Path("ladder") / "season.json"
ENDED_DIR = Path("ladder") / "seasons"
KEPT = ("name", "author", "anchor", "elo", "elo_margin", "king_elo", "score_elo", "games")


def _state(root: Path) -> dict[str, Any]:
    path = root / SEASON_FILE
    return json.loads(path.read_text()) if path.is_file() else {"season": 1, "started": None}


def current(root: Path = ROOT) -> int:
    return int(_state(root)["season"])


def started(root: Path = ROOT) -> str | None:
    """The day this season began, or nothing for season 1, which began with the arena."""
    return _state(root)["started"]


def tag(season: int) -> str:
    """What a season adds to a result's engine tag."""
    return f"/s{season}" if season > 1 else ""


def of(engine: str) -> int:
    """The season a result with this engine tag was played in."""
    found = re.search(r"/s(\d+)$", engine)
    return int(found.group(1)) if found else 1


def deal_seed(secret: int, season: int) -> int:
    """The seed a season deals from. Season 1 used the secret itself; each later season gets its
    own, which can't be worked back to the secret or forward to the next season's."""
    if season <= 1:
        return secret
    return int.from_bytes(hashlib.sha256(f"{secret}|season|{season}".encode()).digest()[:4], "big")


def ended(root: Path = ROOT) -> list[dict[str, Any]]:
    """The final standings of every finished season, latest first."""
    files = sorted((root / ENDED_DIR).glob("*.json"), key=lambda p: -int(p.stem))
    return [json.loads(p.read_text()) for p in files]


def start(root: Path = ROOT, today: str | None = None) -> int:
    """Keep this season's standings as they are now, then begin the next. Returns its number."""
    from .leaderboard import build  # the leaderboard reads seasons too

    today = today or datetime.now(UTC).date().isoformat()
    season = current(root)
    ranked = [row for row in build(root, official_only=True)["rows"] if row["elo"] is not None and not row.get("provisional")]
    final = {"season": season, "started": started(root), "ended": today, "rows": [{k: row.get(k) for k in KEPT} for row in ranked]}
    (root / ENDED_DIR).mkdir(parents=True, exist_ok=True)
    (root / ENDED_DIR / f"{season}.json").write_text(json.dumps(final, indent=1) + "\n")
    (root / SEASON_FILE).write_text(json.dumps({"season": season + 1, "started": today}, indent=1) + "\n")
    return season + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.season", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("now", "start", "seed"))
    args = parser.parse_args(argv)
    if args.command == "start":
        print(start())
    elif args.command == "seed":
        print(deal_seed(int(os.environ.get("LADDER_SEED") or 0), current()))
    else:
        print(current())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
