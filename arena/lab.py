"""The lab: read-only data for watching a training run (jaxenv.selfplay + jaxenv.lab_capture)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

LAB = Path(os.environ.get("ESTIMATION_LAB", Path.home() / "workspace" / "estimation-data" / "bitter"))
RUN_RE = re.compile(r"^[a-z0-9-]{1,40}$")
FIELDS = ("iteration", "decisions", "per_s", "entropy", "kl", "dash", "bid", "with_", "pass_", "super_", "skipped",
          "all_lose", "caller_made", "rounds_per_game", "final_mse", "king_vs_random", "king_vs_start",
          "pos_vs_random", "pos_vs_start", "overflow")  # fmt: skip
CHECK_HEAD = re.compile(r"== (\d+) min, iter (\d+) vs (\w+)")
CHECK_ROW = re.compile(r"sets\s+(\d+)-(\d+)-(\d+)\s+first\s+([\d.]+)%\s+RSA\s+([-+\d.]+) \[([-+\d.]+), ([-+\d.]+)\].*position ([-+\d.]+) \[([-+\d.]+), ([-+\d.]+)\]\s+K/SK/SKo/Ko (\d+)%/(\d+)%/(\d+)%/(\d+)%")
CHECK_KING = re.compile(r"king [\d.]+% \[([-\d.]+)%, ([-\d.]+)%\]")


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:  # the line being written right now
            pass
    return out


def _checks(run: str) -> list[dict]:
    digits = re.sub(r"\D", "", run)
    checks = []
    for path in [LAB / f"eval{digits}.log", *sorted(LAB.glob(f"eval{digits}-*.log"))]:
        if not path.is_file():
            continue
        head = None
        for line in path.read_text().splitlines():
            h = CHECK_HEAD.search(line)
            if h:
                head = h
                continue
            m = CHECK_ROW.search(line)
            if m and head:
                k = CHECK_KING.search(line)
                checks.append({
                    "minutes": int(head[1]), "iteration": int(head[2]), "opponent": head[3],
                    "won": int(m[1]), "drawn": int(m[2]), "lost": int(m[3]), "king": float(m[4]) / 100,
                    "king_low": float(k[1]) / 100 if k else None, "king_high": float(k[2]) / 100 if k else None,
                    # delta = 1.5 x RSA in a two-against-two lineup (see tests/test_positions.py)
                    "delta": 1.5 * float(m[5]), "delta_low": 1.5 * float(m[6]), "delta_high": 1.5 * float(m[7]),
                    "reward": float(m[8]), "low": float(m[9]), "high": float(m[10]),
                    "shares": [int(m[k]) / 100 for k in (11, 12, 13, 14)],
                })  # fmt: skip
    return sorted(checks, key=lambda c: (c["iteration"], c["opponent"]))


async def runs(_: Request) -> Response:
    names = sorted(p.parent.name for p in LAB.glob("*/log.jsonl")) if LAB.is_dir() else []
    return JSONResponse(names)


async def summary(request: Request) -> Response:
    run = request.path_params["run"]
    if not RUN_RE.match(run) or not (LAB / run / "log.jsonl").is_file():
        return JSONResponse({"error": "no such run"}, status_code=404)
    rows = [r for r in _rows(LAB / run / "log.jsonl")]
    step = max(1, len(rows) // 500)
    keep = rows[::step] + rows[-1:]
    trimmed = [{k: r[k] for k in FIELDS if k in r} for r in keep]
    return JSONResponse({"run": run, "rows": trimmed, "versions": _rows(LAB / run / "lab" / "index.jsonl"), "checks": _checks(run)})


async def snapshot(request: Request) -> Response:
    run, iteration = request.path_params["run"], request.path_params["iteration"]
    path = LAB / run / "lab" / f"snap-{iteration:05d}.json"
    if not RUN_RE.match(run) or not path.is_file():
        return JSONResponse({"error": "no such version"}, status_code=404)
    return FileResponse(path, media_type="application/json")
