"""Build the leaderboard from ``results/*.json`` and ``ladder/pairs.json``.

    uv run python -m arena.leaderboard

Every metric of arena.positions gets its own Elo: one Bradley-Terry fit over every pair's set
results under that metric, RandomBot fixed at 400. ``elo`` is weighted rank, the default ranking;
``king_elo`` is King only; ``score_elo`` is score delta. The page switches between them.
Writes ``site/leaderboard.json`` and ``site/index.html``, a single self-contained page.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from estimation.arena.duplicate import slot_at
from estimation.arena.ratings import ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR, PairRecord, bradley_terry

from .bots import ROOT, SUB_PREFIX, load_manifest
from .direct import current as current_matches, game_file, is_clear, load_matches, runoff, table
from .guide import GUIDE, STYLE as GUIDE_STYLE
from .anchor_positions import OUT as POSITIONS_FILE
from .import_ladder import LADDER_FILE
from .positions import NAMES as POSITION_NAMES, decode_scores, set_outcomes
from .plain import about_bot, author_name, base_key, benchmark_name, plain_name
from .run import ANCHORS_FILE, RESULTS_DIR

SITE_DIR = ROOT / "site"
EARLY_DIR = "early"


METRICS = (("weighted", "elo", "Weighted rank"), ("king", "king_elo", "King only"), ("delta", "score_elo", "Score delta"))
RANK_SCRIPT = """
document.querySelectorAll('[data-rank]').forEach(function (button) {
  button.addEventListener('click', function () {
    var m = button.dataset.rank, body = document.querySelector('tbody');
    document.body.dataset.metric = m;
    document.querySelectorAll('[data-rank]').forEach(function (b) { b.setAttribute('aria-pressed', b === button); });
    var early = function (row) { return row.classList.contains('early') ? 1 : 0; };
    var rows = Array.from(body.querySelectorAll('tr.bot')).sort(function (a, b) { return early(a) - early(b) || b.dataset[m] - a.dataset[m]; });
    var place = 0;
    rows.forEach(function (row) {
      var how = row.nextElementSibling && row.nextElementSibling.classList.contains('howrow') ? row.nextElementSibling : null;
      row.querySelector('.rank').textContent = row.dataset[m] > -1e8 && !early(row) ? ++place : '';
      body.appendChild(row);
      if (how) body.appendChild(how);
    });
    var divider = body.querySelector('tr.unranked'), firstEarly = body.querySelector('tr.early');
    if (divider && firstEarly) body.insertBefore(divider, firstEarly);
  });
});
"""


def _pairs(records: list[dict[str, Any]]) -> list[PairRecord]:
    return [
        PairRecord(str(p["a"]), str(p["b"]), int(p["a_wins"]), int(p["draws"]), int(p["b_wins"]))
        for p in records
    ]


def _mean_shares(rows: list[list[float]]) -> list[float] | None:
    return [sum(r[p] for r in rows) / len(rows) for p in range(4)] if rows else None


def _add_new(pairs: list[PairRecord], extra: list[PairRecord]) -> None:
    """Add a live match's sets, unless finished results already rate that pairing."""
    rated = {frozenset((p.a, p.b)) for p in pairs}
    pairs += [p for p in extra if frozenset((p.a, p.b)) not in rated]


def _author(row: dict[str, Any]) -> str:
    """Who made a bot. A test version has no result of its own, so it takes its bot's author."""
    if row["author"] or not row["key"].startswith(SUB_PREFIX):
        return author_name(row["author"])
    try:
        return author_name(load_manifest(base_key(row["key"])[len(SUB_PREFIX) :]).author)
    except (ValueError, OSError):
        return ""


def _overlay_row(key: str, played: list[dict[str, Any]], ratings: tuple[dict, dict, dict]) -> dict[str, Any]:
    return {
        "name": played[0]["name"],
        "author": "",
        "kind": "provisional",
        "description": "",
        "key": key,
        "method": [],
        "elo": ratings[0].get(key),
        "king_elo": ratings[1].get(key),
        "score_elo": ratings[2].get(key),
        "positions": _mean_shares([m["shares"] for m in played]),
        "games": sum(m["done"] * 4 for m in played),
        "versus": [],
        "anchor": False,
        "provisional": True,
    }


def build(
    root: Path = ROOT,
    live: list[dict[str, Any]] | None = None,
    experiments: Path | None = None,
    official_only: bool = False,
) -> dict[str, Any]:
    """Standings from the finished results under ``root``, plus any ``live`` matches
    (arena.live.read_matches), which count for pairings no finished result covers, plus results
    under ``experiments`` for bots ``root`` has not rated, plus ``results/early``, committed results
    of bots the ladder can't play. Those three are marked provisional, can't move an official rating
    and are listed after the ranked bots. A test version of a bot already on the board gets no row:
    the board shows one row per bot. ``official_only`` leaves the early results out."""
    anchors: list[dict[str, str]] = json.loads((root / ANCHORS_FILE.relative_to(ROOT)).read_text())
    ladder_file, positions_file = root / LADDER_FILE.relative_to(ROOT), root / POSITIONS_FILE.relative_to(ROOT)
    ladder = json.loads(ladder_file.read_text()) if ladder_file.is_file() else {"pairs": []}
    results = [json.loads(p.read_text()) for p in sorted((root / RESULTS_DIR.relative_to(ROOT)).glob("*.json"))]
    official = {r["bot"] for r in results}
    early_dir = root / RESULTS_DIR.relative_to(ROOT) / EARLY_DIR
    early = [] if official_only else [json.loads(p.read_text()) for p in sorted(early_dir.glob("*.json"))]
    results += [{**r, "provisional": True} for r in early if r["bot"] not in official]
    official |= {r["bot"] for r in early}
    if experiments and experiments.resolve() != root.resolve():
        for path in sorted((experiments / RESULTS_DIR.relative_to(ROOT)).glob("*.json")):
            result = json.loads(path.read_text())
            if result["bot"] not in official:
                results.append({**result, "provisional": True})
    direct = current_matches(load_matches(root), [r for r in results if not r.get("provisional")])
    counted = [m for m in live or [] if m.get("pairs") and m["state"] != "stopped"
               and (m["kind"] == "entry" or (m["kind"] == "rating" and m["alive"]))]  # fmt: skip
    pairs = _pairs(ladder["pairs"])
    for r in [*results, *direct]:
        pairs += _pairs(r["pairs"])
    _add_new(pairs, _pairs([q for m in counted for q in m["pairs"]["delta"]]))
    score_ratings = bradley_terry(pairs, ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR) if pairs else {}
    matches = json.loads(positions_file.read_text())["matches"] if positions_file.is_file() else []
    by_position = [q for m in matches for q in _pairs(m["pairs"])]
    for r in [*results, *direct]:
        by_position += _pairs(r.get("position_pairs", []))
    official_records = list(by_position)
    _add_new(by_position, _pairs([q for m in counted for q in m["pairs"]["weighted"]]))
    ratings = bradley_terry(by_position, ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR) if by_position else {}
    by_king = [q for m in matches for q in _pairs(m.get("king_pairs", []))]
    for r in [*results, *direct]:
        by_king += _pairs(r.get("king_pairs", []))
    _add_new(by_king, _pairs([q for m in counted for q in m["pairs"]["king"]]))
    king_ratings = bradley_terry(by_king, ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR) if by_king else {}
    anchor_places: dict[str, list[dict[str, Any]]] = {}
    for m in matches:
        for spec, place in m["summary"].items():
            anchor_places.setdefault(spec, []).append(place)

    rows: list[dict[str, Any]] = []
    for r in results:
        key = SUB_PREFIX + r["bot"]
        try:
            current = load_manifest(r["bot"]).to_dict()
        except ValueError:
            current = r["manifest"]
        timings = [v["timing"] for v in r["versus"]]
        decisions = sum(t["count"] for t in timings)
        rows.append(
            {
                "name": r["bot"],
                "author": r["manifest"].get("author", ""),
                "kind": r["manifest"].get("kind", ""),
                "description": current.get("description", r["manifest"].get("description", "")),
                "key": key,
                "provisional": r.get("provisional", False),
                "method": current.get("method", []),
                "elo": ratings.get(key),
                "king_elo": king_ratings.get(key),
                "score_elo": score_ratings.get(key),
                "positions": _mean_shares([v["positions"] for v in r["versus"] if "positions" in v]),
                "games": sum(v["games"] for v in r["versus"]),
                "sets": r["sets"],
                "played_at": r["played_at"],
                "engine": r.get("engine", ""),
                "p95_ms": max((t["p95_s"] for t in timings), default=0.0) * 1000,
                "over_budget": (sum(t["over_budget"] for t in timings) / decisions) if decisions else 0.0,
                "versus": [
                    {
                        "anchor": v["anchor"],
                        "rsa": v["rsa"]["mean"],
                        "rsa_low": v["rsa"]["low"],
                        "rsa_high": v["rsa"]["high"],
                        "first_place": v["first_place"]["mean"],
                        "positions": v.get("positions"),
                        "position_reward": v.get("position_reward", {}).get("mean"),
                        "delta": v.get("delta", {}).get("mean", 1.5 * v["rsa"]["mean"]),
                    }
                    for v in r["versus"]
                ],
                "anchor": False,
            }
        )
    for a in anchors:
        rows.append(
            {
                "name": a["name"],
                "author": a.get("author", "Yosif Elessawi"),
                "kind": "anchor",
                "description": a["spec"],
                "key": a["spec"],
                "method": a.get("method", []),
                "elo": ratings.get(a["spec"]),
                "king_elo": king_ratings.get(a["spec"]),
                "score_elo": score_ratings.get(a["spec"]),
                "positions": _mean_shares([q["shares"] for q in anchor_places.get(a["spec"], [])]),
                "games": None,
                "versus": [],
                "anchor": True,
            }
        )
    known = {row["key"] for row in rows}
    fresh: dict[str, list[dict[str, Any]]] = {}
    for m in counted:
        if m["key"] not in known and base_key(m["key"]) not in known:
            fresh.setdefault(m["key"], []).append(m)
    for key, played in fresh.items():
        rows.append(_overlay_row(key, played, (ratings, king_ratings, score_ratings)))
    for row in rows:
        row["code_name"] = row["name"]
        row["name"] = plain_name(row["key"], row["name"])
        row["author"] = _author(row)
        about = about_bot(row["key"])
        row["description"] = about.get("about", row["description"])
        row["method"] = about.get("method", row["method"])
    if live or experiments or early:
        # Provisional bots must not move official ratings, so every board keeps the ladder's numbers.
        official_rows = {row["key"]: row for row in build(root, official_only=True)["rows"]}
        for row in rows:
            if row["key"] in official_rows:
                for field in ("elo", "king_elo", "score_elo"):
                    row[field] = official_rows[row["key"]][field]
    rows.sort(key=lambda row: (bool(row.get("provisional")), row["elo"] is None, -(row["elo"] or 0)))
    rated = [row["key"] for row in rows if row["elo"] is not None and not row.get("provisional")]
    return {
        "matchups": matchups(results, direct, rows),
        "direct": table(direct),
        "runoff": runoff(rated, official_records),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "anchor": ANCHOR,
        "anchor_rating": ANCHOR_RATING,
        "ladder": {k: ladder.get(k) for k in ("source", "sets", "imported_at")},
        "submissions": sum(not r.get("provisional") for r in results),
        "rows": rows,
    }


STYLE = """
:root{color-scheme:dark;--bg:#17235e;--fg:#eef0ff;--muted:#aab4e6;--line:#3a4a9a;--bar:#f2c14e;--tint:#1d2a6b;--good:#8fe0a8;--bad:#f3a6a8}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.45 "Instrument Sans",system-ui,sans-serif}a{color:var(--fg)}h2,h3{font-family:"Bricolage Grotesque",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
table{width:100%;border-collapse:collapse}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:13px}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.anchor td{color:var(--muted);background:var(--tint)}tr:has(+ tr.howrow) td{border-bottom:none}tr.howrow td{white-space:normal;padding-top:0}.places{display:flex;gap:2px;width:210px;height:12px;vertical-align:middle}.places span{display:block;height:100%}.places span:first-child{border-radius:3px 0 0 3px}.places span:last-child{border-radius:0 3px 3px 0}.p0{background:#f2c14e}.p1{background:#f8e3a8}.p2{background:#f3a6a8}.p3{background:#e5484d}.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}.king{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted)}.metrics{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}.metrics button{font:inherit;font-size:14px;padding:6px 12px;border:1px solid var(--muted);border-radius:6px;background:transparent;color:inherit;cursor:pointer}.metrics button[aria-pressed=true]{background:#f2c14e;border-color:#f2c14e;color:#1c2147;font-weight:600}.banner{display:block;width:100%;height:auto;margin:0 0 16px}.unrated{color:var(--muted);font-size:13px}td.m-weighted,td.m-king,td.m-delta{color:var(--muted)}body[data-metric=weighted] td.m-weighted,body[data-metric=king] td.m-king,body[data-metric=delta] td.m-delta{color:inherit;font-weight:700}span.m-king,span.m-delta{display:none}body[data-metric=king] span.m-weighted,body[data-metric=delta] span.m-weighted{display:none}body[data-metric=king] span.m-king,body[data-metric=delta] span.m-delta{display:inline}
.brand-title{margin:0}.tag{font-size:12px;padding:1px 6px;border:1px solid var(--line);border-radius:10px;color:var(--muted)}.desc{color:var(--muted);font-size:13px;white-space:normal}
.direct{margin-top:32px}table.cross td,table.cross th{white-space:nowrap}table.cross .who-cell{white-space:normal;min-width:130px;max-width:190px;position:sticky;left:0;background:var(--bg);z-index:1}table.cross .rank{color:var(--muted)}table.cross .self{background:var(--tint)}table.cross .close,table.cross .unrated{color:var(--muted)}.pos{color:var(--good)}.neg{color:var(--bad)}details{margin-top:24px}summary{cursor:pointer;color:var(--muted)}
.who-cell{white-space:normal;min-width:230px}td.author{white-space:nowrap;color:var(--muted)}tr.unranked td{white-space:normal;color:var(--muted);padding-top:28px;font-size:13px}tr.unranked strong{color:var(--fg);font-size:15px;margin-right:6px}td.opt{white-space:normal}.places{flex-shrink:0}
@media(max-width:700px){th.opt,td.opt,.who,th.author,td.author{display:none}.who-cell{min-width:0}body:not([data-metric=weighted]) .num.m-weighted,body:not([data-metric=king]) .num.m-king,body:not([data-metric=delta]) .num.m-delta{display:none}}
details.how{margin:6px 0 2px}details.how summary{font-size:13px}.scroll{overflow-x:auto;margin:10px 0}
svg text{fill:var(--fg);font:12px system-ui,sans-serif}
.flow .step{fill:var(--tint);stroke:var(--line)}.i-n{fill:var(--muted)!important;font-size:11px!important}
.i-line{fill:none;stroke:var(--muted);stroke-width:1.4}.i-t{font-weight:600;font-size:13px!important}.i-s{fill:var(--muted)!important;font-size:10px!important}
.i-bid{fill:#8e7cf022;stroke:#8e7cf0}.i-wire{stroke:#f2c14e55;stroke-width:.8}.i-net{fill:#f2c14e}
.i-back{fill:#eb683433;stroke:#eb6834}.i-card{fill:var(--bg);stroke:var(--muted)}.i-win{fill:#1baf7a}.i-loss{fill:#e0525299}
.i-mean{fill:var(--muted)}.i-best{fill:#f2c14e!important}.i-tick{fill:none;stroke:#1baf7a;stroke-width:2.5}.i-data{fill:var(--bg);stroke:#8e7cf0}
.arrow{fill:none;stroke:var(--muted);stroke-width:1.5}
.lbl2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:12px 0 4px}
.scroll{max-width:calc(100vw - 72px)}
@media(max-width:700px){.bar{max-width:48px}th,td{padding:8px 6px}}
"""




def _lines(label: str, width: int = 19) -> list[str]:
    words, lines, cur = label.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return [*lines, cur][:4]


def _count(label: str) -> str:
    """The ``xN`` in a step label (``imagine hidden hands x30``), drawn on the deal icon."""
    for word in label.split():
        if word[:1] in "x×" and word[1:].isdigit():
            return "×" + word[1:]
    return ""


# Each icon is drawn in a 130x56 box at (x, y): a small picture of what the step does.
def _icon(kind: str, x: float, y: float, label: str) -> str:
    cx = x + 65
    if kind == "rule":  # a decision diamond with two branches
        return (f'<path class="i-line" d="M{cx} {y + 2}l16 14l-16 14l-16 -14z"/>'
                f'<text class="i-t" x="{cx}" y="{y + 16}" text-anchor="middle" dominant-baseline="central">?</text>'
                f'<path class="i-line" d="M{cx - 16} {y + 16}h-22v22M{cx + 16} {y + 16}h22v22"/>'
                f'<text class="i-s" x="{cx - 38}" y="{y + 50}" text-anchor="middle">yes</text>'
                f'<text class="i-s" x="{cx + 38}" y="{y + 50}" text-anchor="middle">no</text>')
    if kind == "bid":  # a speech bubble with a number
        return (f'<path class="i-bid" d="M{cx - 26} {y + 6}h52a6 6 0 0 1 6 6v20a6 6 0 0 1 -6 6h-34l-10 10v-10h-8a6 6 0 0 1 -6 -6v-20a6 6 0 0 1 6 -6z"/>'
                f'<text class="i-t" x="{cx}" y="{y + 23}" text-anchor="middle" dominant-baseline="central">3?</text>')
    if kind == "net":  # input dots, a hidden layer, output bars (one tall)
        out = []
        ins = [y + 8 + 10 * k for k in range(5)]
        hid = [y + 13 + 10 * k for k in range(4)]
        for a in ins:
            for b in hid:
                out.append(f'<path class="i-wire" d="M{x + 22} {a}L{x + 58} {b}"/>')
        for b in hid:
            out.append(f'<path class="i-wire" d="M{x + 58} {b}L{x + 84} {y + 28}"/>')
        out += [f'<circle class="i-net" cx="{x + 22}" cy="{a}" r="3.5"/>' for a in ins]
        out += [f'<circle class="i-net" cx="{x + 58}" cy="{b}" r="3.5"/>' for b in hid]
        for k, h in enumerate((10, 34, 16, 6)):
            out.append(f'<rect class="i-net{" i-best" if k == 1 else ""}" x="{x + 88 + k * 9}" y="{y + 48 - h}" width="6" height="{h}" rx="1"/>')
        return "".join(out)
    if kind == "deal":  # three fanned hands of card backs, times N
        out = []
        for k in range(3):
            hx = x + 16 + k * 30
            for j in range(3):
                out.append(f'<rect class="i-back" x="{hx + j * 6}" y="{y + 10 + j * 2}" width="16" height="24" rx="2"/>')
        n = _count(label)
        if n:
            out.append(f'<text class="i-t" x="{x + 116}" y="{y + 24}" text-anchor="middle" dominant-baseline="central">{n}</text>')
        out.append(f'<text class="i-s" x="{cx - 14}" y="{y + 48}" text-anchor="middle">left · across · right</text>')
        return "".join(out)
    if kind == "rollout":  # rows = cards, columns = imagined deals, cells = result, then the mean
        out = []
        pattern = ("+-++-", "--+--", "+++-+")
        for r, row in enumerate(pattern):
            ry = y + 6 + r * 14
            out.append(f'<rect class="i-card" x="{x + 12}" y="{ry}" width="12" height="14" rx="2"/>')
            for c, s in enumerate(row):
                out.append(f'<rect class="{"i-win" if s == "+" else "i-loss"}" x="{x + 28 + c * 14}" y="{ry + 1}" width="12" height="12" rx="2"/>')
            w = 6 + 8 * row.count("+")
            out.append(f'<rect class="i-mean{" i-best" if r == 2 else ""}" x="{x + 102}" y="{ry + 2}" width="{w * 0.55:.0f}" height="8" rx="2"/>')
        out.append(f'<text class="i-s" x="{x + 56}" y="{y + 58}" text-anchor="middle">deal 1 … N</text>')
        return "".join(out)
    if kind == "pick":  # bars, the highest one ticked
        out = []
        for k, h in enumerate((18, 30, 40, 24)):
            out.append(f'<rect class="i-mean{" i-best" if k == 2 else ""}" x="{cx - 34 + k * 18}" y="{y + 46 - h}" width="12" height="{h}" rx="2"/>')
        out.append(f'<path class="i-tick" d="M{cx + 2} {y + 1}l4 4l8 -8"/>')
        return "".join(out)
    if kind == "play":  # four seats round a table
        out = [f'<circle class="i-card" cx="{cx}" cy="{y + 26}" r="13"/>']
        for dx, dy in ((0, -19), (19, 0), (0, 19), (-19, 0)):
            out.append(f'<circle class="i-net" cx="{cx + dx}" cy="{y + 26 + dy}" r="5"/>')
        return "".join(out)
    if kind == "reward":  # the four places, the best one marked
        out = []
        for k, h in enumerate((40, 26, 16, 8)):
            out.append(f'<rect class="{"i-mean i-best" if k == 0 else "i-mean"}" x="{cx - 34 + k * 18}" y="{y + 46 - h}" width="12" height="{h}" rx="2"/>')
        return "".join(out)
    if kind == "update":  # a net with an arrow into it
        out = [f'<circle class="i-net" cx="{cx + 14 + 16 * (k % 2)}" cy="{y + 12 + 12 * k}" r="4"/>' for k in range(4)]
        out.append(f'<path class="i-tick" d="M{cx - 34} {y + 28}h28m-7 -6l7 6l-7 6"/>')
        return "".join(out)
    if kind == "data":  # stacked game records
        return "".join(f'<rect class="i-data" x="{cx - 28 + k * 6}" y="{y + 6 + k * 7}" width="44" height="22" rx="3"/>' for k in range(4))
    return ""


def method_svg(steps: list[list[str]], repeat: str = "") -> str:
    """The algorithm as a strip of steps, each a small picture of what it does and a short
    label, left to right. ``repeat`` draws a loop from the last step back to the first."""
    if not steps:
        return ""
    bw, gap, h = 140, 22, 132
    width = len(steps) * bw + (len(steps) - 1) * gap
    height = h + 34 if repeat else h
    parts = [f'<svg class="flow" viewBox="0 0 {width} {height}" width="{width}" role="img">']
    if repeat:
        first, last = bw / 2, width - bw / 2
        parts.append(f'<path class="loop" d="M{last} {h}v14a8 8 0 0 1 -8 8H{first + 8}a8 8 0 0 1 -8 -8v-10m-4 5l4 -6l4 6"/>')
        parts.append(f'<text class="loop-text" x="{first + 14}" y="{h + 16}">{html.escape(repeat)}</text>')
    for i, (label, kind) in enumerate(steps):
        x = i * (bw + gap)
        parts.append(f'<rect class="step" x="{x + 1}" y="1" width="{bw - 2}" height="{h - 2}" rx="10"/>')
        parts.append(f'<text class="i-n" x="{x + 12}" y="16">{i + 1}</text>')
        parts.append(_icon(kind, x + 5, 12, label))
        for j, line in enumerate(_lines(label)):
            parts.append(f'<text x="{x + bw / 2}" y="{86 + j * 14}" text-anchor="middle">{html.escape(line)}</text>')
        if i < len(steps) - 1:
            ax = x + bw + 2
            parts.append(f'<path class="arrow" d="M{ax} {h / 2}h{gap - 6}m-5 -4l5 4l-5 4"/>')
    parts.append("</svg>")
    return "".join(parts)


def how_it_plays(row: dict[str, Any]) -> str:
    if not row.get("method"):
        return ""
    return (f'<details class="how"><summary>how it plays</summary>'
            f'<div class="scroll">{method_svg(row["method"])}</div></details>')


def _banner() -> str:
    return (ROOT / "brand" / "banner.svg").read_text().replace("<svg ", '<svg class="banner" ', 1)


def _favicon() -> str:
    return base64.b64encode((ROOT / "brand" / "favicon.svg").read_bytes()).decode()


def matchups(results: list[dict[str, Any]], direct: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every pairing played, for the game browser: the record and, where kept, each set's games.

    Each game holds both bots' average seat scores, from ``a``'s side. Direct matches played before
    scores were kept have totals only.
    """
    out = []
    names = {row["key"]: row["name"] for row in rows}
    codes = {row["key"]: row["code_name"] for row in rows}

    def add(a: str, b: str, position_pairs: list[dict[str, Any]], scores: str | None, played: str, kind: str) -> None:
        wins = draws = losses = 0
        for pair in position_pairs:
            if {pair["a"], pair["b"]} != {a, b}:
                continue
            mine, theirs = (pair["a_wins"], pair["b_wins"]) if pair["a"] == a else (pair["b_wins"], pair["a_wins"])
            wins, draws, losses = wins + mine, draws + pair["draws"], losses + theirs
        sets = []
        if scores:
            line = (a, b, a, b)
            for result in decode_scores(scores):
                games = []
                for rotation, game in enumerate(result.scores):
                    mine = [game[s] for s in range(4) if line[slot_at(s, rotation)] == a]
                    theirs = [game[s] for s in range(4) if line[slot_at(s, rotation)] == b]
                    games.append([sum(mine) / 2, sum(theirs) / 2])
                sets.append({"outcome": set_outcomes([result], line, a), "games": games})
        out.append({"a": a, "b": b, "kind": kind, "a_wins": wins, "draws": draws, "b_wins": losses,
                    "sets": wins + draws + losses, "clear": is_clear(wins, losses), "played_at": played[:10],
                    "detail": sets, "a_name": names.get(a, a), "b_name": names.get(b, b),
                    "file": game_file(codes.get(a, a), codes.get(b, b))})  # fmt: skip

    for r in results:
        for v in r["versus"]:
            add(SUB_PREFIX + r["bot"], v["spec"], r.get("position_pairs", []), v.get("scores"), r.get("played_at", ""), "benchmark")
    for m in direct:
        played = m["batches"][-1]["played_at"] if m.get("batches") else ""
        add(SUB_PREFIX + m["a"], SUB_PREFIX + m["b"], m["position_pairs"], m.get("scores"), played, "direct")
    return out


def head_to_head(board: dict[str, Any], top: int = 8) -> str:
    """The top bots against each other, or nothing until direct games exist."""
    if not board["direct"]:
        return ""
    e = html.escape
    bots = [r for r in board["rows"] if not r["anchor"] and not r.get("provisional")][:top]
    played = {(m["a"], m["b"]): m for m in board["direct"]}
    head = "".join(f'<th class="num" title="{e(r["name"])}">{k + 1}</th>' for k, r in enumerate(bots))
    lines = []
    for k, row in enumerate(bots):
        cells = []
        for j, other in enumerate(bots):
            match = played.get((row["key"], other["key"]))
            flip = played.get((other["key"], row["key"]))
            if k == j:
                cells.append('<td class="num self"></td>')
            elif match or flip:
                mine, theirs = (match["a_wins"], match["b_wins"]) if match else (flip["b_wins"], flip["a_wins"])
                clear = (match or flip)["clear"]
                kind = ("pos" if mine > theirs else "neg") if clear else "close"
                draws = (match or flip)["draws"]
                cells.append(f'<td class="num {kind}">{mine}-{draws}-{theirs}</td>')
            else:
                cells.append('<td class="num unrated">-</td>')
        lines.append(f'<tr><td class="who-cell"><span class="rank">{k + 1}</span> {e(row["name"])}</td>{"".join(cells)}</tr>')
    names = {r["key"]: r["name"] for r in board["rows"]}
    leader = next((r for r in board["rows"] if r["elo"] is not None and not r.get("provisional")), None)
    winner = names.get(board["runoff"][0]) if board["runoff"] else None
    verdict = f"Runoff winner: <strong>{e(winner)}</strong>." if winner else ""
    if winner and leader and leader["key"] != board["runoff"][0]:
        verdict += f' The rating puts {e(leader["name"])} first, so treat the top spot as contested.'
    return f"""<section class="direct" aria-labelledby="direct-title">
<h2 id="direct-title">Head to head</h2>
<p class="desc">Sets won, drawn and lost by the bot on the left against each bot across the top. Coloured results are unlikely between equal bots; plain ones are too close to call, and a dash means not played yet. {verdict} The runoff drops the bot scoring least against those left, until one remains.</p>
<div class="scroll"><table class="cross"><thead><tr><th class="who-cell">Bot</th>{head}</tr></thead><tbody>
{chr(10).join(lines)}
</tbody></table></div></section>"""


COLUMNS = 8  # rank, bot, author, three ratings, finishing places, benchmarks
UNRANKED_NOTE = "These bots have a rating from a few games but no place: they are being tested, or can't play on the nightly ladder."


def render(board: dict[str, Any]) -> str:
    rows = board["rows"]
    anchors = [r["name"] for r in rows if r["anchor"]]
    lines = []
    rank = 0
    divided = False
    ranked = sum(r["elo"] is not None and not r.get("provisional") for r in rows)
    for r in rows:
        e = html.escape
        early = r.get("provisional")
        rank += 0 if early else 1  # one global ranking of anchors and submissions; early results sit outside it
        elo = r["elo"]
        cells = "".join(
            f'<td class="num m-{m}">{r[k]:,.0f}</td>' if r.get(k) is not None else f'<td class="num m-{m} unrated">unrated</td>'
            for m, k, _ in METRICS
        )
        sort_keys = " ".join(f'data-{m}="{r[k] if r.get(k) is not None else -1e9}"' for m, k, _ in METRICS)
        places = r.get("positions")
        place_bar = ""
        if places:
            title = " · ".join(f"{n} {x:.0%}" for n, x in zip(POSITION_NAMES, places))
            parts = "".join(f'<span class="p{k}" style="width:{x * 100:.1f}%"></span>' for k, x in enumerate(places))
            place_bar = f'<div class="places" title="{title}" role="img" aria-label="{title}">{parts}</div><span class="king">King {places[0]:.0%}</span>'
        if r["anchor"]:
            name = f'<a href="games/#bot={quote(r["key"])}">{e(r["name"])}</a> <span class="tag">benchmark</span>'
            meta = f'<span class="desc who">{e(r["description"])} Code name: {e(r["code_name"])}.</span>'
            detail = ""
        else:
            name = f'<strong><a href="games/#bot={quote(r["key"])}">{e(r["name"])}</a></strong>'
            if early:
                name += f' <span class="tag">{r["games"]:,} games so far</span>'
            meta = f'<span class="desc who">{e(r["description"])} Code name: {e(r["code_name"])}.</span>'
            vs = " · ".join(
                f'{e(benchmark_name(v["anchor"]))} <span class="m-weighted {"pos" if (v["position_reward"] or 0) >= 0 else "neg"}">{(v["position_reward"] or 0):+.2f}</span>'
                f'<span class="m-king {"pos" if v["first_place"] >= 0.25 else "neg"}">{v["first_place"]:.0%}</span>'
                f'<span class="m-delta {"pos" if v["delta"] >= 0 else "neg"}">{v["delta"]:+.0f}</span>'
                for v in r["versus"]
            )
            detail = (
                f'<span class="desc">{vs}</span><br><span class="desc">{r["games"]:,} games, '
                f'last played {e(r["played_at"][:10])}</span>'
            )
        if early and not divided:
            divided = True
            lines.append(f'<tr class="unranked"><td></td><td colspan="{COLUMNS - 1}"><strong>Not ranked yet.</strong> {UNRANKED_NOTE}</td></tr>')
        lines.append(
            f'<tr class="{"anchor" if r["anchor"] else "sub"} bot{" early" if early else ""}" {sort_keys}>'
            f'<td class="num rank">{rank if elo is not None and not early else ""}</td>'
            f'<td class="who-cell">{name}<br>{meta}</td><td class="author">{e(r["author"])}</td>'
            f'{cells}<td class="opt">{place_bar}</td><td class="opt">{detail}</td></tr>'
        )
        how = how_it_plays(r)
        if how:
            cls = "anchor" if r["anchor"] else "sub"
            lines.append(f'<tr class="{cls} howrow"><td></td><td colspan="{COLUMNS - 1}">{how}</td></tr>')
    ladder = board["ladder"]
    built = board["built_at"].replace("T", " ")[:16] + " UTC"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estimation Arena</title><link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{_favicon()}"><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700&family=Instrument+Sans:wght@400;600&display=swap" rel="stylesheet"><style>{STYLE}{GUIDE_STYLE}:root{{--guide-line:var(--line)}}</style></head>
<body data-metric="weighted"><main>
<h1 class="brand-title">{_banner()}</h1>
<p class="sub">{ranked} bots ranked three ways from the same games, including {len(anchors)} fixed benchmarks. Updated {built}.</p>
<div class="metrics" role="group" aria-label="Rank by">{"".join(f'<button type="button" data-rank="{m}"{" aria-pressed=true" if m == "weighted" else " aria-pressed=false"}>{label}</button>' for m, _, label in METRICS)}</div>
<p class="desc"><span class="key p0"></span>King<span class="key p1"></span>Sub-King<span class="key p2"></span>Sub-Kooz<span class="key p3"></span>Kooz</p>
<p class="desc"><a href="games/">Games</a> · <a href="full/">Whole games</a> · <a href="#guide">What the numbers mean</a> · <a href="diagrams/how-it-works.html">How it works</a> · <a href="https://github.com/essamgouda97/estimation-arena">Code on GitHub</a></p>
<p class="desc">Open “how it plays” under any bot to see how it decides.</p>
<table>
<thead><tr><th class="num">#</th><th>Bot</th><th class="author">Author</th>{"".join(f'<th class="num m-{m}">{label}</th>' for m, _, label in METRICS)}<th class="opt">Finishing places</th><th class="opt">Against each benchmark</th></tr></thead>
<tbody>
{chr(10).join(lines)}
</tbody></table>
{head_to_head(board)}
{GUIDE}
</main><script>{RANK_SCRIPT}</script></body></html>
"""


def browser_data(board: dict[str, Any], games_dir: Path) -> dict[str, Any]:
    """What the game browser reads: the bots, their pairings and which games can be watched."""
    fields = ("key", "name", "code_name", "anchor", "provisional", "elo", "king_elo", "score_elo", "positions", "description")
    return {
        "built_at": board["built_at"],
        "bots": [{k: row.get(k) for k in fields} for row in board["rows"]],
        "matchups": board["matchups"],
        "games": sorted(p.name for p in games_dir.glob("*.vs.*.json")),
        "runoff": board["runoff"],
    }


def write_browser(board: dict[str, Any], out: Path) -> None:
    folder = out / "games"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text((Path(__file__).parent / "static" / "games.html").read_text())
    (folder / "data.json").write_text(json.dumps(browser_data(board, folder), separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.leaderboard")
    parser.add_argument("--out", type=Path, default=SITE_DIR, help="output folder (site/)")
    args = parser.parse_args(argv)
    board = build()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaderboard.json").write_text(json.dumps(board, indent=1) + "\n")
    (args.out / "index.html").write_text(render(board))
    write_browser(board, args.out)
    for r in board["rows"]:
        elo = f"{r['elo']:7.0f}" if r["elo"] is not None else "unrated"
        print(f"{elo}  {r['name']}{'  (anchor)' if r['anchor'] else ''}")
    print(f"wrote {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
