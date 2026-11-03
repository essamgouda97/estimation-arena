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

from .bots import ROOT, SUB_PREFIX
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
        timings = [v["timing"] for v in r["versus"]]
        decisions = sum(t["count"] for t in timings)
        rows.append(
            {
                "name": r["bot"],
                "author": r["manifest"].get("author", ""),
                "kind": r["manifest"].get("kind", ""),
                "description": r["manifest"].get("description", ""),
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
tr.anchor td{color:var(--muted);background:var(--tint)}.bar{display:inline-block;height:10px;background:var(--bar);border-radius:3px;vertical-align:middle;margin-right:8px}
.tag{font-size:12px;padding:1px 6px;border:1px solid var(--line);border-radius:10px;color:var(--muted)}.desc{color:var(--muted);font-size:13px;white-space:normal}
.pos{color:var(--good)}.neg{color:var(--bad)}details{margin-top:24px}summary{cursor:pointer;color:var(--muted)}
@media(max-width:700px){th.opt,td.opt{display:none}}
"""


def render(board: dict[str, Any]) -> str:
    rows = board["rows"]
    rated = [r["elo"] for r in rows if r["elo"] is not None]
    top = max(rated) if rated else 1.0
    anchors = [r["name"] for r in rows if r["anchor"]]
    lines = []
    rank = 0
    for r in rows:
        e = html.escape
        if not r["anchor"]:
            rank += 1
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
            f'<td class="num">{"" if r["anchor"] else rank}</td>'
            f"<td>{name}</td><td class=\"opt\">{meta}</td>"
            f'<td class="num">{bar}{elo_text}</td><td class="opt">{detail}</td></tr>'
        )
    ladder = board["ladder"]
    built = board["built_at"].replace("T", " ")[:16] + " UTC"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estimation Arena</title><style>{STYLE}</style></head>
<body><main>
<h1>Estimation Arena</h1>
<p class="sub">{board["submissions"]} submissions and {len(anchors)} anchors, one Elo scale (Random = {board["anchor_rating"]:.0f}). Built {built}.</p>
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
