"""The live board's data: standings and matches, rebuilt only when a file changed."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from .guide import GUIDE, STYLE as GUIDE_STYLE
from .leaderboard import browser_data, build, method_svg
from . import ladder_live
from .live import LIVE, public, read_matches
from .plain import about_bot, base_key, plain_name
from .training import training_of
from .bots import ROOT, SUB_PREFIX

# This checkout's results are the official ones, the same as the public page. Where matches run
# on the home server, a bot not merged yet can also have results: those show as provisional.
RESULTS_ROOT = Path(os.environ.get("ESTIMATION_RESULTS_ROOT", Path.home() / "workspace" / "estimation-arena"))
RECENT = 8
GAME_NAME = re.compile(r"[a-z0-9-]+\.vs\.[a-z0-9-]+\.json")
KEPT_NAME = re.compile(r"[a-z0-9-]+\.vs\.[a-z0-9-]+\.\d+\.json")  # a watch game the home server kept
_cache: dict[str, Any] = {"signature": None, "standings": [], "direct": [], "runoff": []}
_estimate: dict[str, Any] = {"at": None, "value": {}}
ROW_FIELDS = ("name", "author", "code_name", "key", "kind", "anchor", "provisional", "elo", "king_elo", "score_elo", "elo_margin", "king_elo_margin", "score_elo_margin", "close_to", "positions", "games", "played_at")
MATCH_FIELDS = ("id", "kind", "key", "name", "opponent", "sets", "done", "state", "strip", "shares", "reward", "king", "delta", "started_at", "updated_at")


def _signature(root: Path) -> tuple:
    paths = [*root.glob("results/*.json"), *root.glob("results/direct/*.json"), *root.glob("results/early/*.json"), *root.glob("site/games/*.vs.*.json"), root / "ladder" / "positions.json", root / "ladder" / "season.json", *LIVE.glob("*.json")]
    stamps = []
    for path in paths:
        try:
            stamps.append((path.name, path.stat().st_mtime_ns))
        except OSError:
            pass
    return tuple(sorted(stamps))


def payload() -> dict[str, Any]:
    feed = ladder_live.read_feed()
    local = read_matches()
    here = {(m["key"], m["opponent"]) for m in local}  # a pairing played here too is shown once
    remote = [m for m in ladder_live.remote_matches(feed) if (m["key"], m["opponent"]) not in here]
    matches = sorted(local + remote, key=lambda m: -m["updated_at"])
    ladder_at = feed["updated_at"] if feed else None
    signature = (_signature(ROOT), _signature(RESULTS_ROOT), tuple(m["state"] for m in matches), ladder_at)
    if signature != _cache["signature"]:
        board = build(ROOT, matches, RESULTS_ROOT)
        _cache["standings"] = [{k: row.get(k) for k in ROW_FIELDS} for row in board["rows"]]
        _cache["direct"], _cache["runoff"] = board["direct"], board["runoff"]
        _cache["season"] = {k: board["season"][k] for k in ("number", "started")}
        _cache["browser"] = browser_data(board, ROOT / "site" / "games")
        _cache["signature"] = signature
    names = {row["key"]: row["name"] for row in _cache["standings"]}
    view = [{**{k: m.get(k) for k in MATCH_FIELDS}, "name": plain_name(m["key"], m["name"]), "opponent": names.get(m["opponent"], plain_name(m["opponent"], public(m["opponent"])[1]))} for m in matches]
    return {
        "standings": _cache["standings"],
        "direct": _cache["direct"],
        "runoff": _cache["runoff"],
        "season": _cache.get("season"),
        "playing": [m for m in view if m["state"] == "playing"],
        "recent": [m for m in view if m["state"] != "playing"][:RECENT],
    }


def _manifest(key: str) -> dict[str, Any]:
    """A submission's manifest, from the merged bots or the ones still being tested."""
    name = key[len(SUB_PREFIX) :].split(":")[0]
    for root in (ROOT, RESULTS_ROOT):
        path = root / "bots" / name / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text())
    return {}


def explain(key: str) -> dict[str, Any] | None:
    """How one bot on the board plays: its steps and training, as HTML we generate."""
    code_names = {row["key"]: row["code_name"] for row in _cache["standings"]}
    listed = key if key in code_names else base_key(key)  # a test version is explained by its bot's row
    if listed not in code_names:
        return None
    if key.startswith(SUB_PREFIX):
        about = _manifest(key)
    else:
        anchors = json.loads((ROOT / "anchors.json").read_text())
        about = next((a for a in anchors if a["spec"] == key), {})
    plain = about_bot(key)
    trained = training_of(key) or {}
    return {
        "description": plain.get("about", about.get("description", "")),
        "code_name": code_names[listed],
        "method": method_svg(plain.get("method", about.get("method", []))),
        "training": {
            "summary": trained.get("summary", ""),
            "stages": [{"title": s["title"], "svg": method_svg(s["steps"], s.get("repeat"))} for s in trained.get("stages", [])],
            "curves": trained.get("curves", ""),
            "lab": trained.get("lab", False),
        },
    }


def api(_: Request) -> Response:
    return JSONResponse(payload())


def ladder(_: Request) -> Response:
    """The ladder as it plays: its plan and matches, the bot names, and the game being watched."""
    feed = ladder_live.read_feed()
    if _cache["signature"] is None:
        payload()
    names = {row["key"]: row["name"] for row in _cache["standings"]}
    name = lambda key: names.get(key, plain_name(key, public(key)[1]))  # noqa: E731
    if not feed:
        return JSONResponse({"run": None, "watch": None, "names": {}})
    keys = {m["key"] for m in feed["matches"]} | {m["opponent"] for m in feed["matches"]}
    keys |= {SUB_PREFIX + p["bot"] for p in feed["plan"]} | {p["anchor"] for p in feed["plan"]}
    watch = ladder_live.read_watch()
    if watch:
        keys |= {watch["a"], watch["b"]}
    run = {**feed, "url": ladder_live.safe_url(feed.get("url", "")), "running": ladder_live.is_running(feed),
           "matches": ladder_live.remote_matches(feed)}  # fmt: skip
    return JSONResponse({"run": run, "watch": watch, "names": {k: name(k) for k in keys}, "estimate": estimate(feed)})


def estimate(feed: dict[str, Any]) -> dict[str, Any]:
    """Where each bot on the ladder would land if it stopped now: the standings refitted with the
    ladder's sets counted, even for a bot that already has a saved result. Rebuilt only when the
    ladder reports, since a build takes seconds."""
    if _estimate["at"] != feed["updated_at"]:
        rows = build(ROOT, ladder_live.remote_matches(feed), keep_official=False)["rows"]
        ranked = [r for r in rows if r["elo"] is not None and not r.get("provisional")]
        mine = {SUB_PREFIX + p["bot"] for p in feed["plan"]}
        _estimate["value"] = {"rows": [{"key": r["key"], "name": r["name"], "anchor": r["anchor"], "elo": r["elo"]} for r in ranked],
                              "bots": sorted(mine & {r["key"] for r in ranked})}  # fmt: skip
        _estimate["at"] = feed["updated_at"]
    return _estimate["value"]


def games_data(_: Request) -> Response:
    payload()
    return JSONResponse(_cache["browser"])


def game_file(request: Request) -> Response:
    """A watchable game: the name is checked, so a request can't name any other file."""
    name = request.path_params["name"]
    path = ladder_live.DATA / ladder_live.KEPT / name if KEPT_NAME.fullmatch(name) else ROOT / "site" / "games" / name
    if not (GAME_NAME.fullmatch(name) or KEPT_NAME.fullmatch(name)) or not path.is_file():
        return JSONResponse({"error": "no such game"}, status_code=404)
    return FileResponse(path)


def kept_games(request: Request) -> Response:
    """The games the home server kept of one pairing, newest first."""
    pairing = request.query_params.get("file", "")
    if not GAME_NAME.fullmatch(pairing):
        return JSONResponse({"error": "no such pairing"}, status_code=404)
    return JSONResponse({"games": ladder_live.kept_games(pairing)})


def guide(_: Request) -> Response:
    return JSONResponse({"html": GUIDE, "css": GUIDE_STYLE})


def bot(request: Request) -> Response:
    if _cache["signature"] is None:
        payload()
    data = explain(request.query_params.get("key", ""))
    return JSONResponse(data) if data else JSONResponse({"error": "no such bot on the board"}, status_code=404)
