"""The nightly ladder on the live board: what leaves GitHub, and how the home server reads it."""

import json
import subprocess
import time

from starlette.testclient import TestClient

from arena import board, ladder_live
from arena.replay import record_game
from arena.serve import app

MATCH = {"id": "made-from-the-seed", "pid": 4242, "seed": 987654321, "kind": "rating", "key": "sub:guess-search",
         "name": "guess-search", "opponent": "heuristic", "sets": 40, "done": 3, "finished": False, "strip": "WDL",
         "started_at": 1.0, "updated_at": 2.0, "pairs": {"delta": [], "weighted": [], "king": []}}  # fmt: skip


def _feed(updated_at, finished=False, matches=(MATCH,)):
    return {"run_id": "1", "url": "", "started_at": 0.0, "sets": 40, "finished": finished, "updated_at": updated_at,
            "plan": [{"bot": "guess-search", "anchor": "heuristic"}], "matches": [ladder_live.public_match(m) for m in matches]}  # fmt: skip


def test_nothing_secret_leaves_the_ladder():
    out = ladder_live.public_match(MATCH)
    assert "seed" not in out and "pid" not in out and "987654321" not in json.dumps(out)
    assert out["id"] == ladder_live.public_match({**MATCH, "seed": 1})["id"] != MATCH["id"]


def test_the_bundle_carries_the_plan_and_the_matches(tmp_path):
    (tmp_path / "live").mkdir()
    (tmp_path / "live" / "a.json").write_text(json.dumps(MATCH))
    (tmp_path / "plan.tsv").write_text("guess-search\theuristic\nguess-search\trandom\n")
    feed = ladder_live.bundle(tmp_path / "live", tmp_path / "plan.tsv", {"run_id": "7"})
    assert feed["plan"] == [{"bot": "guess-search", "anchor": "heuristic"}, {"bot": "guess-search", "anchor": "random"}]
    assert [m["done"] for m in feed["matches"]] == [3] and feed["run_id"] == "7" and not feed["finished"]


def test_a_running_ladder_plays_and_a_silent_one_has_stopped():
    now = time.time()
    running = ladder_live.remote_matches(_feed(now - 10), now)
    assert [(m["state"], m["alive"]) for m in running] == [("playing", True)]
    assert ladder_live.current_pairing(_feed(now - 10), now) == ("sub:guess-search", "heuristic")
    silent = ladder_live.remote_matches(_feed(now - ladder_live.STALE_S - 1), now)
    assert [(m["state"], m["alive"]) for m in silent] == [("stopped", False)]
    assert ladder_live.current_pairing(_feed(now, finished=True), now) is None
    done = ladder_live.remote_matches(_feed(now - 10, matches=({**MATCH, "finished": True},)), now)
    assert [(m["state"], m["alive"]) for m in done] == [("done", True)]  # counts until its results arrive


def test_progress_goes_up_a_branch_and_comes_back(tmp_path, monkeypatch):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    for k, v in (("user.name", "t"), ("user.email", "t@example.com")):
        subprocess.run(["git", "-C", str(work), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(work), "remote", "add", "origin", str(remote)], check=True)
    monkeypatch.chdir(work)
    ladder_live.push({"done": 1})
    ladder_live.push({"done": 2})  # replaces the branch: always one commit
    assert ladder_live.fetch(tmp_path / "cache.git", str(remote)) == {"done": 2}
    count = subprocess.run(["git", "-C", str(remote), "rev-list", "--count", ladder_live.BRANCH], capture_output=True, text=True)
    assert count.stdout.strip() == "1"
    assert ladder_live.fetch(tmp_path / "other.git", str(tmp_path / "missing")) is None


def test_a_watch_game_is_written_as_it_plays(tmp_path):
    seen = []
    game = record_game(["random", "heuristic", "random", "heuristic"], 5, rounds=1, on_move=lambda g: seen.append(len(g["rounds"][0]["moves"])))
    assert seen == list(range(1, len(game["rounds"][0]["moves"]) + 1))

    def fake(seats, seed, rounds, on_move):
        on_move({"rounds": [{"moves": [[0, 1]]}]})
        return {"rounds": [], "final": [1, 2, 3, 4], "finished": True}

    ladder_live.watch_game(("sub:guess-search", "heuristic"), 2, tmp_path, every=0, record=fake)
    watch = ladder_live.read_watch(tmp_path)
    assert watch["finished"] and watch["number"] == 2 and watch["a"] == "sub:guess-search"


def test_the_board_counts_the_ladder_while_it_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder_live, "DATA", tmp_path)
    (tmp_path / ladder_live.FEED).write_text(json.dumps(_feed(time.time())))
    board._cache["signature"] = None
    client = TestClient(app())
    data = client.get("/api/ladder").json()
    assert data["run"]["running"] and data["run"]["matches"][0]["state"] == "playing"
    assert data["names"]["heuristic"] and data["watch"] is None
    playing = client.get("/api/board").json()["playing"]
    assert any(m["key"] == "sub:guess-search" for m in playing)
    assert client.get("/ladder").status_code == 200
    board._cache["signature"] = None


def test_the_progress_file_never_picks_what_the_server_runs():
    assert ladder_live.watchable("heuristic") and ladder_live.watchable("sub:guess-search")
    for key in ("sub:no-such-bot", "sub:guess-search:net=/etc/passwd", "sub:../bots/x", "rm -rf"):
        assert not ladder_live.watchable(key)
    assert ladder_live.safe_url("javascript:alert(1)") == ""
    assert ladder_live.safe_url("https://github.com/a/b/actions/runs/1") == "https://github.com/a/b/actions/runs/1"
