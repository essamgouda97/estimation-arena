"""The weekly full-game ladder: the top bots of the nightly ladder play each other in whole games.

    uv run python -m arena.full pairs                  # this week's pairs, one a:b per line
    uv run python -m arena.full play a b --seed S      # one batch of whole games for a pair, saved
    uv run python -m arena.full build                  # its own standings in site/full/

The nightly ladder plays micro games, a game's first 5 rounds, so every bot gets rated every
night. Once a week the top ``CONTENDERS`` play all 18 rounds against each other, which also tests
the fixed-trump rounds that micro games leave out. Each pair plays on its own runner.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from estimation.arena.ratings import DEFAULT_PRIOR, PairRecord, bradley_terry

from . import duels, leaderboard
from .bots import ROOT, SUB_PREFIX
from .direct import current, is_clear, load_matches, record, runoff, table
from .guide import STYLE as GUIDE_STYLE
from .run import FULL_GAME, RESULTS_DIR, engine_version

FULL_DIR = Path("results") / "full" / "direct"
CONTENDERS = 5
MAX_SETS = 100
SCALE = 1500  # with no benchmark to pin the scale, the bots' average rating


def contenders(root: Path = ROOT, top: int = CONTENDERS) -> list[str]:
    """The best rated bots of the nightly ladder, best first. Benchmarks and early results are out."""
    rows = leaderboard.build(root, official_only=True)["rows"]
    rated = [r for r in rows if not r["anchor"] and not r.get("provisional") and r["elo"] is not None]
    return [r["key"][len(SUB_PREFIX) :] for r in rated[:top]]


def _results(root: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((root / RESULTS_DIR.relative_to(ROOT)).glob("*.json"))]


def matches(root: Path = ROOT) -> list[dict[str, Any]]:
    """Whole-game matches whose bots are unchanged since they played."""
    return current(load_matches(root, FULL_DIR), _results(root), engine=engine_version(FULL_GAME))


def due(root: Path = ROOT) -> list[tuple[str, str]]:
    """Pairs of contenders with no match yet, then pairs still too close to call."""
    top = contenders(root)
    done = {tuple(sorted((m["a"], m["b"]))): m for m in matches(root)}
    unplayed, close = [], []
    for a, b in combinations(top, 2):
        match = done.get(tuple(sorted((a, b))))
        if match is None:
            unplayed.append((a, b))
            continue
        wins, draws, losses = record(match)
        if not is_clear(wins, losses) and wins + draws + losses < MAX_SETS:
            close.append((a, b))
    return unplayed + close


def play(a: str, b: str, seed: int, workers: int = 4, sets: int = duels.BATCH_SETS, root: Path = ROOT) -> Path:
    duels.play(a, b, seed, workers, sets, root, rounds=FULL_GAME, folder=FULL_DIR)
    first, second = sorted((a, b))
    return root / FULL_DIR / f"{first}.vs.{second}.json"


def _ratings(records: list[PairRecord]) -> dict[str, float]:
    """Bradley-Terry ratings, then shifted so the bots average ``SCALE``."""
    keys = sorted({k for p in records for k in (p.a, p.b)})
    if not keys:
        return {}
    fitted = bradley_terry(records, keys[0], 0.0, DEFAULT_PRIOR)
    if not fitted:
        return {}
    mean = sum(fitted.values()) / len(fitted)
    return {k: v - mean + SCALE for k, v in fitted.items()}


def _records(found: list[dict[str, Any]], field: str) -> list[PairRecord]:
    return [PairRecord(p["a"], p["b"], p["a_wins"], p["draws"], p["b_wins"]) for m in found for p in m[field]]


def build(root: Path = ROOT) -> dict[str, Any]:
    found = matches(root)
    top = contenders(root)
    weighted = _records(found, "position_pairs")
    ratings = {m: _ratings(_records(found, f)) for m, f in (("elo", "position_pairs"), ("king_elo", "king_pairs"), ("score_elo", "pairs"))}
    ranked = {r["key"]: r for r in leaderboard.build(root, official_only=True)["rows"]}
    keys = sorted({k for m in found for k in (SUB_PREFIX + m["a"], SUB_PREFIX + m["b"])}, key=lambda k: -ratings["elo"].get(k, -1e9))
    rows = []
    for key in keys:
        mine = [m for m in found if key in (SUB_PREFIX + m["a"], SUB_PREFIX + m["b"])]
        sets = sum(sum(record(m)) for m in mine)
        rows.append({"key": key, "code_name": key[len(SUB_PREFIX) :], "name": ranked[key]["name"] if key in ranked else key[len(SUB_PREFIX) :],
                     "author": ranked[key]["author"] if key in ranked else "",
                     "anchor": False, "contender": key[len(SUB_PREFIX) :] in top, "sets": sets,
                     **{m: r.get(key) for m, r in ratings.items()}})  # fmt: skip
    return {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "contenders": [SUB_PREFIX + name for name in top],
        "rows": rows,
        "direct": table(found),
        "runoff": runoff(keys, weighted),
        "scale": SCALE,
    }


def render(board: dict[str, Any]) -> str:
    e = html.escape
    lines = []
    for k, r in enumerate(board["rows"], 1):
        cells = "".join(f'<td class="num">{r[f]:,.0f}</td>' if r.get(f) is not None else '<td class="num unrated">unrated</td>'
                        for f in ("elo", "king_elo", "score_elo"))  # fmt: skip
        note = "" if r["contender"] else ' <span class="tag">not in the top 5 now</span>'
        lines.append(f'<tr class="sub bot"><td class="num rank">{k}</td><td class="who-cell"><strong>{e(r["name"])}</strong>{note}<br>'
                     f'<span class="desc who">Code name: {e(r["code_name"])}. {r["sets"]:,} sets of whole games.</span></td>'
                     f'<td class="author">{e(r.get("author", ""))}</td>{cells}</tr>')  # fmt: skip
    built = board["built_at"].replace("T", " ")[:16] + " UTC"
    empty = "" if lines else '<p class="desc">No whole games yet. The first weekly run fills this page.</p>'
    table_html = (f'<table><thead><tr><th class="num">#</th><th>Bot</th><th class="author">Author</th><th class="num">Weighted rank</th><th class="num">King only</th>'
                  f'<th class="num">Score delta</th></tr></thead><tbody>{chr(10).join(lines)}</tbody></table>') if lines else ""  # fmt: skip
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estimation Arena: whole games</title><link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{leaderboard._favicon()}"><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700&family=Instrument+Sans:wght@400;600&display=swap" rel="stylesheet"><style>{leaderboard.STYLE}{GUIDE_STYLE}:root{{--guide-line:var(--line)}}</style></head>
<body><main>
<h1 class="brand-title">{leaderboard._banner()}</h1>
<h2>Whole games</h2>
<p class="sub">Once a week the top {CONTENDERS} bots of the <a href="../">main standings</a> play each other in whole games: all 18 rounds, the last 5 with a fixed trump. The main standings use the first 5 rounds only, so this page shows who holds up over a full game. Updated {built}.</p>
<p class="desc">Ratings work like the main page, but there are no benchmark bots here, so the bots average {board["scale"]:,}. Compare ratings only within this page.</p>
{empty}{table_html}
{leaderboard.head_to_head(board, top=len(board["rows"]))}
</main></body></html>
"""


def write(board: dict[str, Any], out: Path) -> None:
    folder = out / "full"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "leaderboard.json").write_text(json.dumps(board, indent=1) + "\n")
    (folder / "index.html").write_text(render(board))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.full", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pairs", help="print this week's pairs, one a:b per line")
    one = commands.add_parser("play", help="play one batch of whole games for a pair and print its file")
    one.add_argument("a")
    one.add_argument("b")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--workers", type=int, default=4)
    page = commands.add_parser("build", help="write site/full/")
    page.add_argument("--out", type=Path, default=leaderboard.SITE_DIR)
    args = parser.parse_args(argv)
    if args.command == "pairs":
        for a, b in due():
            print(f"{a}:{b}")
        return 0
    if args.command == "play":
        print(play(args.a, args.b, args.seed, args.workers).relative_to(ROOT))
        return 0
    write(build(), args.out)
    print(f"wrote {args.out / 'full' / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
