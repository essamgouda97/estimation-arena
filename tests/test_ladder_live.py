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
    assert "sub:guess-search" in data["estimate"]["bots"]
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


def test_the_estimate_lets_tonights_sets_move_a_rated_bot():
    from arena.bots import ROOT
    from arena.leaderboard import build

    win = [{"a": "sub:lowcard-wire", "b": "sub:guess-search", "a_wins": 40, "draws": 0, "b_wins": 0}]
    match = {**MATCH, "key": "sub:lowcard-wire", "opponent": "sub:guess-search", "alive": True, "state": "playing",
             "pairs": {"weighted": win, "king": win, "delta": win}}  # fmt: skip
    elo = lambda rows: next(r["elo"] for r in rows if r["key"] == "sub:lowcard-wire")  # noqa: E731
    saved = elo(build(ROOT, [match])["rows"])
    assert saved == elo(build(ROOT)["rows"])  # the board keeps the saved rating
    assert elo(build(ROOT, [match], keep_official=False)["rows"]) > saved + 100


def test_finished_watch_games_are_kept_for_the_game_browser(tmp_path, monkeypatch):
    game = {"seats": ["sub:guess-search", "heuristic"] * 2, "seed": 7, "rounds": [], "final": [10, -4, 6, 2], "finished": True}
    for at in range(1000, 1000 + ladder_live.KEEP_PER_PAIRING + 3):
        ladder_live.keep_game(tmp_path, "guess-search.vs.heuristic.json", game, at)
    ladder_live.keep_game(tmp_path, "guess-search.vs.heuristic-two.json", game, 5)
    kept = ladder_live.kept_games("guess-search.vs.heuristic.json", tmp_path)
    assert len(kept) == ladder_live.KEEP_PER_PAIRING  # the oldest ones are dropped
    assert kept[0]["played_at"] == 1042 and kept[0]["final"] == [10, -4, 6, 2]
    assert len(ladder_live.kept_games("guess-search.vs.heuristic-two.json", tmp_path)) == 1

    monkeypatch.setattr(ladder_live, "DATA", tmp_path)
    client = TestClient(app())
    listed = client.get("/games/kept.json", params={"file": "guess-search.vs.heuristic.json"}).json()["games"]
    assert listed[0]["file"] == "guess-search.vs.heuristic.1042.json"
    assert client.get("/games/" + listed[0]["file"]).json()["seed"] == 7
    assert client.get("/games/kept.json", params={"file": "../../etc/passwd"}).status_code == 404
    assert client.get("/games/guess-search.vs.heuristic.9.json").status_code == 404


def test_the_games_page_compares_two_bots():
    page = (ladder_live.Path(__file__).parent.parent / "arena" / "static" / "games.html").read_text()
    assert "async function versus(" in page and "kept.json?file=" in page


def test_a_game_can_be_asked_for_and_is_played_first(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder_live, "DATA", tmp_path)
    client = TestClient(app())
    pair = {"a": "sub:guess-search", "b": "heuristic"}
    assert client.post("/games/play", json=pair).json() == {"waiting": 1}
    assert client.post("/games/play", json={"a": "heuristic", "b": "sub:guess-search"}).json() == {"waiting": 1}  # asked once
    assert client.post("/games/play", json={"a": "random", "b": "heuristic"}).json() == {"waiting": 2}
    for bad in ({"a": "heuristic", "b": "heuristic"}, {"a": "sub:../x", "b": "heuristic"}, {"a": "heuristic"}):
        assert client.post("/games/play", json=bad).status_code == 400
    state = client.get("/games/kept.json", params={"file": "guess-search.vs.heuristic.json", **pair}).json()
    assert state["can_play"] and state["waiting"] == 1 and not state["playing"]
    assert ladder_live.take_request(tmp_path) == ("sub:guess-search", "heuristic")
    assert ladder_live.place_in_line("random", "heuristic") == 1
    for _ in range(ladder_live.MAX_REQUESTS):
        ladder_live.take_request(tmp_path)
    for other in ("pimc", "search", "random", "hybrid", "sub:rl-pimc"):
        assert client.post("/games/play", json={"a": "heuristic", "b": other}).status_code == 200
    assert client.post("/games/play", json=pair).status_code == 429  # the line is short
