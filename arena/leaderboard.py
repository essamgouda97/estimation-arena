"""Build the leaderboard from ``results/*.json`` and ``ladder/pairs.json``.

    uv run python -m arena.leaderboard

Every metric of arena.positions gets its own Elo: one Bradley-Terry fit over every pair's set
results under that metric, RandomBot fixed at 400. ``elo`` is weighted rank, the default ranking;
``king_elo`` is King only; ``score_elo`` is score delta. The page switches between them.
Writes ``site/leaderboard.json`` and ``site/index.html``, a single self-contained page.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from estimation.arena.ratings import ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR, PairRecord, bradley_terry

from .bots import ROOT, SUB_PREFIX, load_manifest
from .anchor_positions import OUT as POSITIONS_FILE
from .import_ladder import LADDER_FILE
from .positions import NAMES as POSITION_NAMES
from .run import ANCHORS_FILE, RESULTS_DIR

SITE_DIR = ROOT / "site"


METRICS = (("weighted", "elo", "Weighted rank"), ("king", "king_elo", "King only"), ("delta", "score_elo", "Score delta"))
RANK_SCRIPT = """
document.querySelectorAll('[data-rank]').forEach(function (button) {
  button.addEventListener('click', function () {
    var m = button.dataset.rank, body = document.querySelector('tbody');
    document.body.dataset.metric = m;
    document.querySelectorAll('[data-rank]').forEach(function (b) { b.setAttribute('aria-pressed', b === button); });
    var rows = Array.from(body.querySelectorAll('tr.bot')).sort(function (a, b) { return b.dataset[m] - a.dataset[m]; });
    rows.forEach(function (row, k) {
      var how = row.nextElementSibling && row.nextElementSibling.classList.contains('howrow') ? row.nextElementSibling : null;
      row.querySelector('.rank').textContent = row.dataset[m] > -1e8 ? k + 1 : '';
      body.appendChild(row);
      if (how) body.appendChild(how);
    });
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


def build(root: Path = ROOT, live: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Standings from the finished results under ``root``, plus any ``live`` matches
    (arena.live.read_matches), which count for pairings no finished result covers."""
    anchors: list[dict[str, str]] = json.loads((root / ANCHORS_FILE.relative_to(ROOT)).read_text())
    ladder_file, positions_file = root / LADDER_FILE.relative_to(ROOT), root / POSITIONS_FILE.relative_to(ROOT)
    ladder = json.loads(ladder_file.read_text()) if ladder_file.is_file() else {"pairs": []}
    results = [json.loads(p.read_text()) for p in sorted((root / RESULTS_DIR.relative_to(ROOT)).glob("*.json"))]
    counted = [m for m in live or [] if m.get("pairs") and m["state"] != "stopped"
               and (m["kind"] == "entry" or (m["kind"] == "rating" and m["alive"]))]  # fmt: skip
    pairs = _pairs(ladder["pairs"])
    for r in results:
        pairs += _pairs(r["pairs"])
    _add_new(pairs, _pairs([q for m in counted for q in m["pairs"]["delta"]]))
    score_ratings = bradley_terry(pairs, ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR) if pairs else {}
    matches = json.loads(positions_file.read_text())["matches"] if positions_file.is_file() else []
    by_position = [q for m in matches for q in _pairs(m["pairs"])]
    for r in results:
        by_position += _pairs(r.get("position_pairs", []))
    _add_new(by_position, _pairs([q for m in counted for q in m["pairs"]["weighted"]]))
    ratings = bradley_terry(by_position, ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR) if by_position else {}
    by_king = [q for m in matches for q in _pairs(m.get("king_pairs", []))]
    for r in results:
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
                "method": current.get("method", []),
                "elo": ratings.get(key),
                "king_elo": king_ratings.get(key),
                "score_elo": score_ratings.get(key),
                "positions": _mean_shares([v["positions"] for v in r["versus"] if "positions" in v]),
                "games": sum(v["games"] for v in r["versus"]),
                "sets": r["sets"],
                "seed": r["seed"],
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
        if m["key"] not in known:
            fresh.setdefault(m["key"], []).append(m)
    for key, played in fresh.items():
        rows.append(_overlay_row(key, played, (ratings, king_ratings, score_ratings)))
    rows.sort(key=lambda row: (row["elo"] is None, -(row["elo"] or 0)))
    return {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "anchor": ANCHOR,
        "anchor_rating": ANCHOR_RATING,
        "ladder": {k: ladder.get(k) for k in ("source", "sets", "seed", "imported_at")},
        "submissions": len(results),
        "rows": rows,
    }


STYLE = """
:root{--bg:#fff;--fg:#1b1b1f;--muted:#6b6f76;--line:#e3e5e8;--bar:#3b6ea5;--tint:#f4f6f8;--good:#2e7d32;--bad:#c62828}
@media(prefers-color-scheme:dark){:root{--bg:#121316;--fg:#e8e9ec;--muted:#9aa0a6;--line:#2b2e33;--bar:#7aa7d9;--tint:#1b1d21;--good:#81c784;--bad:#ef9a9a}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
table{width:100%;border-collapse:collapse}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:13px}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.anchor td{color:var(--muted);background:var(--tint)}tr:has(+ tr.howrow) td{border-bottom:none}tr.howrow td{white-space:normal;padding-top:0}.places{display:flex;gap:2px;width:210px;height:12px;vertical-align:middle}.places span{display:block;height:100%}.places span:first-child{border-radius:3px 0 0 3px}.places span:last-child{border-radius:0 3px 3px 0}.p0{background:#2a78d6}.p1{background:#8fbbef}.p2{background:#f0a8a0}.p3{background:#d03b3b}.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}.king{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted)}.metrics{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}.metrics button{font:inherit;font-size:14px;padding:6px 12px;border:1px solid var(--muted);border-radius:6px;background:transparent;color:inherit;cursor:pointer}.metrics button[aria-pressed=true]{background:#2a78d6;border-color:#2a78d6;color:#fff}.unrated{color:var(--muted);font-size:13px}td.m-weighted,td.m-king,td.m-delta{color:var(--muted)}body[data-metric=weighted] td.m-weighted,body[data-metric=king] td.m-king,body[data-metric=delta] td.m-delta{color:inherit;font-weight:700}span.m-king,span.m-delta{display:none}body[data-metric=king] span.m-weighted,body[data-metric=delta] span.m-weighted{display:none}body[data-metric=king] span.m-king,body[data-metric=delta] span.m-delta{display:inline}
.tag{font-size:12px;padding:1px 6px;border:1px solid var(--line);border-radius:10px;color:var(--muted)}.desc{color:var(--muted);font-size:13px;white-space:normal}
.pos{color:var(--good)}.neg{color:var(--bad)}details{margin-top:24px}summary{cursor:pointer;color:var(--muted)}
.who-cell{white-space:normal;min-width:230px}td.opt{white-space:normal}.places{flex-shrink:0}
@media(max-width:700px){th.opt,td.opt,.who{display:none}.who-cell{min-width:0}body:not([data-metric=weighted]) .num.m-weighted,body:not([data-metric=king]) .num.m-king,body:not([data-metric=delta]) .num.m-delta{display:none}}
details.how{margin:6px 0 2px}details.how summary{font-size:13px}.scroll{overflow-x:auto;margin:10px 0}
svg text{fill:var(--fg);font:12px system-ui,sans-serif}
.flow .step{fill:var(--tint);stroke:var(--line)}.i-n{fill:var(--muted)!important;font-size:11px!important}
.i-line{fill:none;stroke:var(--muted);stroke-width:1.4}.i-t{font-weight:600;font-size:13px!important}.i-s{fill:var(--muted)!important;font-size:10px!important}
.i-bid{fill:#8e7cf022;stroke:#8e7cf0}.i-wire{stroke:#3987e555;stroke-width:.8}.i-net{fill:#3987e5}
.i-back{fill:#eb683433;stroke:#eb6834}.i-card{fill:var(--bg);stroke:var(--muted)}.i-win{fill:#1baf7a}.i-loss{fill:#e0525299}
.i-mean{fill:var(--muted)}.i-best{fill:#eda100!important}.i-tick{fill:none;stroke:#1baf7a;stroke-width:2.5}.i-data{fill:var(--bg);stroke:#8e7cf0}
.arrow{fill:none;stroke:var(--muted);stroke-width:1.5}
.lbl2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:12px 0 4px}
.move{border:1px solid var(--line);border-radius:10px;padding:10px 12px;max-width:520px}
.sit{margin:2px 0 6px;font-size:13px}.lbl{color:var(--muted);font-size:12px;margin-right:6px}.seat{color:var(--muted);font-size:11px;margin:0 3px 0 8px}
.pc{display:inline-block;min-width:34px;padding:1px 5px;border:1px solid var(--line);border-radius:5px;background:var(--bg);font-weight:600;text-align:center;font-variant-numeric:tabular-nums}
.pc.red{color:#e05252}.pc.on{border:2px solid #eda100}.pc.dim{opacity:.5;font-weight:400}
table.opts{width:auto;margin:4px 0 8px}table.opts td,table.opts th{border:none;padding:3px 10px 3px 0;font-size:13px;color:var(--fg)!important;background:none!important}
.track{display:inline-block;width:90px;height:9px;border-radius:3px;background:var(--line);margin-right:6px;vertical-align:middle;overflow:hidden}
.scroll{max-width:calc(100vw - 72px)}.move{max-width:min(520px,calc(100vw - 72px))}
@media(max-width:700px){.bar{max-width:48px}th,td{padding:8px 6px}.track{width:54px}.sit{line-height:1.9}}
.fill{display:block;height:100%}.f-net{background:#3987e5}.f-search{background:#eb6834}.tick{color:#1baf7a;font-weight:600}
"""


SUITS, RANKS = "♣♦♥♠", "23456789TJQKA"
SEATS = {1: "left", 2: "across", 3: "right"}


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
        parts.append(f'<text class="loop-text" x="{width / 2}" y="{h + 16}" text-anchor="middle">{html.escape(repeat)}</text>')
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


def _card(c: int, cls: str = "") -> str:
    red = " red" if c // 13 in (1, 2) else ""
    return f'<span class="pc{red} {cls}">{RANKS[c % 13].replace("T", "10")}{SUITS[c // 13]}</span>'


def _pick_frame(frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most telling decision: one the bot explained, where the search overruled the net if
    any, with the most legal cards."""
    if not frames:
        return None

    def telling(f: dict[str, Any]) -> tuple[bool, bool, int]:
        net, search = f.get("scores", {}).get("net"), f.get("scores", {}).get("search")
        overruled = bool(net and search) and max(net, key=net.get) != max(search, key=search.get)
        return bool(f.get("scores")), overruled, len(f["legal"])  # search overruling the net first

    return max(frames, key=telling)


def move_html(frame: dict[str, Any]) -> str:
    """One real decision: the situation, then one row per legal card with the net's
    probability and the search's average score, the played card ticked."""
    trump = "no trump" if frame["trump"] == 4 else f'trump {SUITS[frame["trump"]]}'
    est = frame["estimate"]
    goal = f'won {frame["won"]} of {est} tricks bid' if est is not None else f'won {frame["won"]}'
    table = " ".join(f'<span class="seat">{SEATS.get(seat, seat)}</span>{_card(c)}' for seat, c in frame["trick"])
    table = table or '<span class="desc">empty: it leads</span>'
    legal, chosen = frame["legal"], frame["chosen"]
    rest = [c for c in sorted(frame["hand"], key=lambda c: (c // 13, c % 13)) if c not in legal]
    scores = {name: {int(k): v for k, v in vals.items()} for name, vals in frame.get("scores", {}).items()}
    net, search = scores.get("net", {}), scores.get("search", {})
    lo, hi = (min(search.values()), max(search.values())) if search else (0.0, 1.0)
    head = "<tr><th>card</th>" + ("<th>net</th>" if net else "") + ("<th>search avg</th>" if search else "") + "<th></th></tr>"
    rows = []
    order = sorted(legal, key=lambda c: (-search.get(c, -1e9), -net.get(c, 0.0), c))
    for c in order:
        cells = f"<td>{_card(c, 'on' if c == chosen else '')}</td>"
        if net:
            p = net.get(c, 0.0)
            cells += f'<td><span class="track"><span class="fill f-net" style="width:{100 * p:.0f}%"></span></span>{100 * p:.0f}%</td>'
        if search:
            if c in search:
                v = search[c]
                w = 8 + 92 * (v - lo) / ((hi - lo) or 1)
                cells += f'<td><span class="track"><span class="fill f-search" style="width:{w:.0f}%"></span></span>{v:+.1f}</td>'
            else:
                cells += '<td class="desc">not searched</td>'
        cells += f'<td class="tick">{"✓ played" if c == chosen else ""}</td>'
        rows.append(f"<tr>{cells}</tr>")
    rest_html = " ".join(_card(c, "dim") for c in rest)
    return (f'<div class="move"><div class="sit"><b>{trump}</b> · {goal}</div>'
            f'<div class="sit"><span class="lbl">table</span> {table}</div>'
            f'<table class="opts">{head}{"".join(rows)}</table>'
            + (f'<div class="sit"><span class="lbl">rest of hand</span> {rest_html}</div>' if rest else "")
            + "</div>")


def how_it_plays(row: dict[str, Any], samples: dict[str, Any]) -> str:
    frame = _pick_frame(samples.get(row["key"], {}).get("frames", []))
    if not row.get("method") and frame is None:
        return ""
    move = ""
    if frame is not None:
        move = f'<div class="lbl2">one real decision</div>{move_html(frame)}'
    return (f'<details class="how"><summary>how it plays</summary>'
            f'<div class="lbl2">algorithm</div><div class="scroll">{method_svg(row.get("method", []))}</div>'
            f"{move}</details>")


def render(board: dict[str, Any]) -> str:
    rows = board["rows"]
    anchors = [r["name"] for r in rows if r["anchor"]]
    samples_file = SITE_DIR / "samples.json"
    samples = json.loads(samples_file.read_text()) if samples_file.is_file() else {}
    lines = []
    rank = 0
    for r in rows:
        e = html.escape
        rank += 1  # one global ranking: anchors and submissions alike
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
            name = f'{e(r["name"])} <span class="tag">anchor</span>'
            meta = f'<span class="desc who">{e(r["author"])}: {e(r["description"])}</span>'
            detail = ""
        else:
            name = f'<strong>{e(r["name"])}</strong> <span class="tag">{e(r["kind"])}</span>'
            meta = f'<span class="desc who">{e(r["author"])}: {e(r["description"])}</span>'
            vs = " · ".join(
                f'{e(v["anchor"])} <span class="m-weighted {"pos" if (v["position_reward"] or 0) >= 0 else "neg"}">{(v["position_reward"] or 0):+.2f}</span>'
                f'<span class="m-king {"pos" if v["first_place"] >= 0.25 else "neg"}">{v["first_place"]:.0%}</span>'
                f'<span class="m-delta {"pos" if v["delta"] >= 0 else "neg"}">{v["delta"]:+.0f}</span>'
                for v in r["versus"]
            )
            detail = (
                f'<span class="desc">{vs}</span><br><span class="desc">{r["games"]} games, '
                f'seed {r["seed"]}, p95 {r["p95_ms"]:.0f} ms, over budget {r["over_budget"]:.1%}, '
                f'{e(r["played_at"][:10])}</span>'
            )
        lines.append(
            f'<tr class="{"anchor" if r["anchor"] else "sub"} bot" {sort_keys}>'
            f'<td class="num rank">{rank if elo is not None else ""}</td>'
            f'<td class="who-cell">{name}<br>{meta}</td>'
            f'{cells}<td class="opt">{place_bar}</td><td class="opt">{detail}</td></tr>'
        )
        how = how_it_plays(r, samples)
        if how:
            cls = "anchor" if r["anchor"] else "sub"
            lines.append(f'<tr class="{cls} howrow"><td></td><td colspan="6">{how}</td></tr>')
    ladder = board["ladder"]
    built = board["built_at"].replace("T", " ")[:16] + " UTC"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estimation Arena</title><style>{STYLE}</style></head>
<body data-metric="weighted"><main>
<h1>Estimation Arena</h1>
<p class="sub">{board["submissions"]} submissions and {len(anchors)} anchors, rated three ways from the same games (Random = {board["anchor_rating"]:.0f}). Built {built}.</p>
<div class="metrics" role="group" aria-label="Rank by">{"".join(f'<button type="button" data-rank="{m}"{" aria-pressed=true" if m == "weighted" else " aria-pressed=false"}>{label}</button>' for m, _, label in METRICS)}</div>
<p class="desc"><span class="key p0"></span>King +1<span class="key p1"></span>Sub-King +0.2<span class="key p2"></span>Sub-Kooz −0.2<span class="key p3"></span>Kooz −1</p>
<p class="desc">Open “how it plays” under any bot: its algorithm, then one real decision from the same deal (seat 0 vs three Heuristic bots).</p>
<table>
<thead><tr><th class="num">#</th><th>Bot</th>{"".join(f'<th class="num m-{m}">{label}</th>' for m, _, label in METRICS)}<th class="opt">Final positions</th><th class="opt">Against each anchor</th></tr></thead>
<tbody>
{chr(10).join(lines)}
</tbody></table>
<details><summary>How the numbers are made</summary>
<p class="desc">A submission plays each anchor in duplicate sets: every deal is played four times with the seats rotated, so luck cancels. A game pays by final position: King +1, Sub-King +0.2, Sub-Kooz −0.2, Kooz −1, tied seats sharing their positions. Every game also counts two other ways: King only pays 1 for King and 0 for anything else, and score delta is our seats' mean score minus the opponent's. Each way decides its own set winner, and each rating is one Bradley-Terry fit over every pair's set results under it, anchors included. Elo counts sets won, so a single exploding score cannot swing it. Against each anchor shows the average per game: reward, King share or delta. The bar shows each position's share, averaged over the bot's opponents. Over budget counts card decisions slower than 0.25 s.</p>
</details>
</main><script>{RANK_SCRIPT}</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.leaderboard")
    parser.add_argument("--out", type=Path, default=SITE_DIR, help="output folder (site/)")
    args = parser.parse_args(argv)
    board = build()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaderboard.json").write_text(json.dumps(board, indent=1) + "\n")
    (args.out / "index.html").write_text(render(board))
    for r in board["rows"]:
        elo = f"{r['elo']:7.0f}" if r["elo"] is not None else "unrated"
        print(f"{elo}  {r['name']}{'  (anchor)' if r['anchor'] else ''}")
    print(f"wrote {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
