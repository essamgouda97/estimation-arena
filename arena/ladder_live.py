"""The nightly ladder as it plays, on the home server's live board.

The ladder runs on GitHub's machines, which can't reach the home server. So the ladder pushes a
small progress file to its own branch every 30 seconds, and the home server fetches it:

    GitHub ladder --(push ladder.json to branch ladder-live)--> GitHub
    home server  --(git fetch every 20 s)--> its copy of ladder.json --> /ladder and the standings

Rated games use the secret seed, so the file carries only what the public results show anyway:
which pairing is playing, each set's outcome and the running ratings. No cards, no seed. To show
cards, the home server plays the same pairing at the same time on public deals: games to watch,
which never count towards a rating.

    uv run python -m arena.ladder_live publish --live DIR --plan FILE   # on the ladder, every 30 s
    uv run python -m arena.ladder_live follow                           # on the home server
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

BRANCH = "ladder-live"
FEED = "ladder.json"
REPO = os.environ.get("ESTIMATION_REPO", "https://github.com/essamgouda97/estimation-arena")
DATA = Path(os.environ.get("ESTIMATION_LADDER", Path.home() / "workspace" / "estimation-data" / "ladder"))
STALE_S = 300  # a ladder silent this long has stopped, even if it never said so
PUBLISH_EVERY_S = 30
FETCH_EVERY_S = 20
WATCH_PAUSE_S = 20  # how long a finished watch game stays up before the next one starts

# What a match file may say in public. Everything else (the seed, the process id, a file name
# made from the seed) stays on the ladder's machine.
PUBLIC_FIELDS = ("kind", "key", "name", "opponent", "sets", "done", "finished", "strip", "shares",
                 "reward", "king", "delta", "pairs", "started_at", "updated_at")  # fmt: skip


# On the ladder ------------------------------------------------------------------------------


def public_match(match: dict[str, Any]) -> dict[str, Any]:
    """A match file with only its public fields, and an id that doesn't depend on the seed."""
    out = {k: match[k] for k in PUBLIC_FIELDS if k in match}
    out["id"] = hashlib.sha1(f"{match['key']}|{match['opponent']}|ladder".encode()).hexdigest()[:16]
    return out


def read_plan(path: Path) -> list[dict[str, str]]:
    """The ladder's plan, one ``bot<TAB>benchmark`` line per pairing, in playing order."""
    lines = path.read_text().splitlines() if path.is_file() else []
    return [{"bot": bot, "anchor": spec} for bot, _, spec in (line.partition("\t") for line in lines) if spec]


def bundle(live: Path, plan: Path, run: dict[str, Any], finished: bool = False) -> dict[str, Any]:
    matches = []
    for path in sorted(live.glob("*.json")):
        try:
            matches.append(public_match(json.loads(path.read_text())))
        except (OSError, ValueError, KeyError):
            continue
    return {**run, "plan": read_plan(plan), "matches": matches, "finished": finished, "updated_at": time.time()}


def _git(*args: str, data: str | None = None) -> str:
    return subprocess.run(["git", *args], input=data, capture_output=True, text=True, check=True).stdout.strip()


def push(feed: dict[str, Any], remote: str = "origin") -> None:
    """Replace the branch with one commit holding ``ladder.json``. Plumbing only, so the ladder's
    own checkout, index and branch are never touched while it saves results."""
    blob = _git("hash-object", "-w", "--stdin", data=json.dumps(feed, separators=(",", ":")))
    tree = _git("mktree", data=f"100644 blob {blob}\t{FEED}\n")
    commit = _git("commit-tree", tree, "-m", "ladder progress")
    _git("push", "--quiet", "--force", remote, f"{commit}:refs/heads/{BRANCH}")


def publish(live: Path, plan: Path, run: dict[str, Any], every: float = 0, finished: bool = False) -> None:
    """Push the progress once, or every ``every`` seconds until stopped. A failed push only
    waits for the next one: the live view must never stop the ladder."""
    while True:
        try:
            push(bundle(live, plan, run, finished))
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"ladder-live: push failed: {error}", flush=True)
        if not every:
            return
        time.sleep(every)


# On the home server ---------------------------------------------------------------------------


def fetch(cache: Path, repo: str = REPO) -> dict[str, Any] | None:
    """The latest progress file from GitHub, or None when there is none or GitHub can't be reached."""
    if not (cache / "HEAD").is_file():
        cache.mkdir(parents=True, exist_ok=True)
        _git("init", "--quiet", "--bare", str(cache))
    try:
        _git("-C", str(cache), "fetch", "--quiet", "--depth", "1", repo, BRANCH)
        return json.loads(_git("-C", str(cache), "show", f"FETCH_HEAD:{FEED}"))
    except (subprocess.CalledProcessError, ValueError):
        return None


def read_feed(data: Path | None = None) -> dict[str, Any] | None:
    try:
        return json.loads(((data or DATA) / FEED).read_text())
    except (OSError, ValueError):
        return None


def is_running(feed: dict[str, Any] | None, now: float | None = None) -> bool:
    return bool(feed) and not feed["finished"] and (now or time.time()) - feed["updated_at"] < STALE_S


def remote_matches(feed: dict[str, Any] | None, now: float | None = None) -> list[dict[str, Any]]:
    """The ladder's matches in the shape of ``arena.live.read_matches``. While the ladder runs
    they all count, finished ones too, until the results they save reach this checkout."""
    if not feed:
        return []
    running = is_running(feed, now)
    return [{**m, "alive": running, "remote": True,
             "state": "done" if m["finished"] else "playing" if running else "stopped"}
            for m in feed["matches"]]  # fmt: skip


def current_pairing(feed: dict[str, Any] | None, now: float | None = None) -> tuple[str, str] | None:
    """The bot and benchmark the ladder is playing right now."""
    playing = [m for m in remote_matches(feed, now) if m["state"] == "playing"]
    return (playing[0]["key"], playing[0]["opponent"]) if playing else None


def watchable(key: str, root: Path | None = None) -> bool:
    """A benchmark, or a bot merged in this checkout. The progress file comes from a branch, so
    it never chooses what code or files the home server loads: only names already here."""
    from .bots import ROOT, SUB_PREFIX

    root = root or ROOT
    if key in {a["spec"] for a in json.loads((root / "anchors.json").read_text())}:
        return True
    name = key[len(SUB_PREFIX) :]
    return key.startswith(SUB_PREFIX) and bool(re.fullmatch(r"[a-z0-9-]+", name)) and (root / "bots" / name / "manifest.json").is_file()


def safe_url(url: str) -> str:
    return url if url.startswith("https://github.com/") else ""


def read_watch(data: Path | None = None) -> dict[str, Any] | None:
    try:
        return json.loads(((data or DATA) / "watch.json").read_text())
    except (OSError, ValueError):
        return None


def _write(path: Path, value: Any) -> None:
    part = path.with_suffix(f".{os.getpid()}.part")
    part.write_text(json.dumps(value, separators=(",", ":")))
    part.replace(path)


def watch_game(pairing: tuple[str, str], number: int, data: Path, every: float = 0.25,
               record: Callable[..., dict[str, Any]] | None = None, stop: Callable[[], bool] = lambda: False) -> None:  # fmt: skip
    """Play one public-deal game of ``pairing`` and keep ``watch.json`` up to date as it goes."""
    from .replay import game_seed, record_game
    from .run import GAME_ROUNDS

    a, b = pairing
    seed = game_seed(a, b) + number
    head = {"a": a, "b": b, "number": number, "started_at": time.time()}
    last = [0.0]

    def moved(game: dict[str, Any]) -> None:
        if stop():
            raise SystemExit(0)
        if time.time() - last[0] >= every:
            last[0] = time.time()
            _write(data / "watch.json", {**head, "game": game, "finished": False, "updated_at": last[0]})

    game = (record or record_game)([a, b, a, b], seed, rounds=GAME_ROUNDS, on_move=moved)
    _write(data / "watch.json", {**head, "game": game, "finished": True, "updated_at": time.time()})


def follow(data: Path | None = None) -> None:
    """Fetch the ladder's progress every 20 s and, while it plays, play its pairing on public deals."""
    data = data or DATA
    data.mkdir(parents=True, exist_ok=True)
    parent = os.getppid()

    def orphaned() -> bool:  # the app that started this has gone: stop with it
        return os.getppid() != parent

    def fetching() -> None:
        while not orphaned():
            try:
                feed = fetch(data / "cache.git")
                if feed:
                    _write(data / FEED, feed)
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"ladder-live: fetch failed: {error}", flush=True)
            time.sleep(FETCH_EVERY_S)

    threading.Thread(target=fetching, daemon=True).start()
    played: dict[tuple[str, str], int] = {}
    while not orphaned():
        pairing = current_pairing(read_feed(data))
        if pairing is None or not all(map(watchable, pairing)):
            time.sleep(FETCH_EVERY_S)
            continue
        number = played.get(pairing, 0)
        played[pairing] = number + 1
        try:
            watch_game(pairing, number, data, stop=orphaned)
        except Exception as error:  # noqa: BLE001  a bot that can't play here must not stop the follower
            print(f"ladder-live: can't play {pairing} here: {error}", flush=True)
            _write(data / "watch.json", {"a": pairing[0], "b": pairing[1], "number": number, "error": str(error),
                                         "finished": True, "updated_at": time.time()})  # fmt: skip
            time.sleep(STALE_S)
            continue
        time.sleep(WATCH_PAUSE_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.ladder_live", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    out = commands.add_parser("publish", help="push the ladder's progress to the ladder-live branch")
    out.add_argument("--live", type=Path, required=True, help="the folder arena.live writes match files to")
    out.add_argument("--plan", type=Path, required=True, help="bot<TAB>benchmark lines, in playing order")
    out.add_argument("--every", type=float, default=0, help="keep pushing every this many seconds")
    out.add_argument("--finished", action="store_true", help="say the ladder has finished")
    commands.add_parser("follow", help="on the home server: fetch the progress and play watch games")
    args = parser.parse_args(argv)
    if args.command == "follow":
        follow()
        return 0
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    run = {"run_id": run_id, "url": f"{server}/{repo}/actions/runs/{run_id}" if run_id else "",
           "started_at": float(os.environ.get("LADDER_STARTED_AT") or time.time()),
           "sets": int(os.environ.get("LADDER_SETS") or 0)}  # fmt: skip
    publish(args.live, args.plan, run, args.every, args.finished)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
