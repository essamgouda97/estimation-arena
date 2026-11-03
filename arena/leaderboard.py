"""Build the leaderboard from ``results/*.json`` and ``ladder/pairs.json``.

    uv run python -m arena.leaderboard

One Bradley-Terry fit over every pairwise set result (the imported ladder plus every
submission's runs), RandomBot fixed at 400, gives every bot one Elo on the same scale.
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
from .import_ladder import LADDER_FILE
from .run import ANCHORS_FILE, RESULTS_DIR

SITE_DIR = ROOT / "site"


def _pairs(records: list[dict[str, Any]]) -> list[PairRecord]:
    return [
        PairRecord(str(p["a"]), str(p["b"]), int(p["a_wins"]), int(p["draws"]), int(p["b_wins"]))
        for p in records
    ]


def build() -> dict[str, Any]:
    anchors: list[dict[str, str]] = json.loads(ANCHORS_FILE.read_text())
    ladder = json.loads(LADDER_FILE.read_text()) if LADDER_FILE.is_file() else {"pairs": []}
    results = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    pairs = _pairs(ladder["pairs"])
    for r in results:
        pairs += _pairs(r["pairs"])
    ratings = bradley_terry(pairs, ANCHOR, ANCHOR_RATING, DEFAULT_PRIOR) if pairs else {}

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
                "games": None,
                "versus": [],
                "anchor": True,
            }
        )
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
main{max-width:1040px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
table{width:100%;border-collapse:collapse}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:13px}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.anchor td{color:var(--muted);background:var(--tint)}tr:has(+ tr.howrow) td{border-bottom:none}tr.howrow td{white-space:normal;padding-top:0}.bar{display:inline-block;height:10px;background:var(--bar);border-radius:3px;vertical-align:middle;margin-right:8px}
.tag{font-size:12px;padding:1px 6px;border:1px solid var(--line);border-radius:10px;color:var(--muted)}.desc{color:var(--muted);font-size:13px;white-space:normal}
.pos{color:var(--good)}.neg{color:var(--bad)}details{margin-top:24px}summary{cursor:pointer;color:var(--muted)}
@media(max-width:700px){th.opt,td.opt{display:none}}
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
    if kind == "data":  # stacked game records
        return "".join(f'<rect class="i-data" x="{cx - 28 + k * 6}" y="{y + 6 + k * 7}" width="44" height="22" rx="3"/>' for k in range(4))
    return ""


def method_svg(steps: list[list[str]]) -> str:
    """The algorithm as a strip of steps, each a small picture of what it does and a short
    label, left to right."""
    if not steps:
        return ""
    bw, gap, h = 140, 22, 132
    width = len(steps) * bw + (len(steps) - 1) * gap
    parts = [f'<svg class="flow" viewBox="0 0 {width} {h}" width="{width}" role="img">']
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
    rated = [r["elo"] for r in rows if r["elo"] is not None]
    top = max(rated) if rated else 1.0
    anchors = [r["name"] for r in rows if r["anchor"]]
    samples_file = SITE_DIR / "samples.json"
    samples = json.loads(samples_file.read_text()) if samples_file.is_file() else {}
    lines = []
    rank = 0
    for r in rows:
        e = html.escape
        rank += 1  # one global ranking: anchors and submissions alike
        elo = r["elo"]
        bar = f'<span class="bar" style="width:{max(2, int(140 * (elo / top)))}px"></span>' if elo else ""
        elo_text = f"{elo:,.0f}" if elo is not None else "unrated"
        if r["anchor"]:
            name = f'{e(r["name"])} <span class="tag">anchor</span>'
            meta = f'{e(r["author"])}<br><span class="desc">{e(r["description"])}</span>'
            detail = ""
        else:
            name = f'<strong>{e(r["name"])}</strong> <span class="tag">{e(r["kind"])}</span>'
            meta = f'{e(r["author"])}<br><span class="desc">{e(r["description"])}</span>'
            vs = " · ".join(
                f'{e(v["anchor"])} <span class="{"pos" if v["rsa"] >= 0 else "neg"}">{v["rsa"]:+.0f}</span>'
                for v in r["versus"]
            )
            detail = (
                f'<span class="desc">{vs}</span><br><span class="desc">{r["games"]} games, '
                f'seed {r["seed"]}, p95 {r["p95_ms"]:.0f} ms, over budget {r["over_budget"]:.1%}, '
                f'{e(r["played_at"][:10])}</span>'
            )
        lines.append(
            f'<tr class="{"anchor" if r["anchor"] else "sub"}">'
            f'<td class="num">{rank if elo is not None else ""}</td>'
            f"<td>{name}</td><td class=\"opt\">{meta}</td>"
            f'<td class="num">{bar}{elo_text}</td><td class="opt">{detail}</td></tr>'
        )
        how = how_it_plays(r, samples)
        if how:
            cls = "anchor" if r["anchor"] else "sub"
            lines.append(f'<tr class="{cls} howrow"><td></td><td colspan="4">{how}</td></tr>')
    ladder = board["ladder"]
    built = board["built_at"].replace("T", " ")[:16] + " UTC"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estimation Arena</title><style>{STYLE}</style></head>
<body><main>
<h1>Estimation Arena</h1>
<p class="sub">{board["submissions"]} submissions and {len(anchors)} anchors, one Elo scale (Random = {board["anchor_rating"]:.0f}). Built {built}.</p>
<p class="desc">Open “how it plays” under any bot: its algorithm, then one real decision from the same deal (seat 0 vs three Heuristic bots).</p>
<table>
<thead><tr><th class="num">#</th><th>Bot</th><th class="opt">Author</th><th class="num">Elo</th><th class="opt">Score advantage vs anchors (RSA)</th></tr></thead>
<tbody>
{chr(10).join(lines)}
</tbody></table>
<details><summary>How the numbers are made</summary>
<p class="desc">A submission plays each anchor in duplicate sets: every deal is played four times with the seats rotated, so luck cancels. RSA is the mean score advantage per game over the anchor. Elo is one Bradley-Terry fit over every pairwise set result, including the engine ladder <code>{html.escape(str(ladder.get("source")))}</code> ({ladder.get("sets")} sets), which is what places the anchors. Over budget counts card decisions slower than 0.25 s.</p>
</details>
</main></body></html>
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
