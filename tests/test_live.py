import json

from estimation.arena.duplicate import SetResult, slot_at

from arena import board, leaderboard, live
from arena.bots import ROOT


def _result(k, mine, theirs):
    """Slots 0 and 2 are mine, wherever each rotation seats them."""
    by_slot = (mine, theirs, mine - 1, theirs - 1)
    games = tuple(tuple(by_slot[slot_at(seat, rotation)] for seat in range(4)) for rotation in range(4))
    return SetResult(k, 0, games, (18,) * 4)


def test_public_hides_paths():
    assert live.public("sub:bitter-net:net=/data/secret/run2-final.npz") == ("sub:bitter-net:run2-final", "run2-final")
    assert live.public("sub:table-pressure") == ("sub:table-pressure", "table-pressure")
    assert live.public("hybrid") == ("hybrid", "hybrid")
    assert live.public("sub:b:top_k=2") == ("sub:b:top_k=2", "b top_k=2")


def test_writes_nothing_without_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE", tmp_path / "absent")
    match = live.Match(("sub:a", "hybrid", "sub:a", "hybrid"), 2, 1)
    match.update([_result(0, 100, 50)], finished=True)
    assert not (tmp_path / "absent").exists()


def test_a_match_reports_its_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE", tmp_path)
    lineup = ("sub:a:net=/x/n.npz", "hybrid", "sub:a:net=/x/n.npz", "hybrid")
    match = live.Match(lineup, 3, 1, kind="entry")
    match.update([_result(0, 100, 50), _result(1, 10, 90)])
    [m] = live.read_matches()
    assert (m["done"], m["strip"], m["state"], m["key"]) == (2, "WL", "playing", "sub:a:n")
    assert "/x/" not in json.dumps(m)
    assert m["pairs"]["weighted"][0]["a"] == "sub:a:n"
    match.update([_result(0, 100, 50), _result(1, 10, 90), _result(2, 100, 90)], finished=True)
    assert live.read_matches()[0]["state"] == "done"


def test_dead_process_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({"pid": 2**22 + 12345, "finished": False, "updated_at": 1}))
    assert live.read_matches()[0]["state"] == "stopped"


def _entry(key, opponent, wins, losses, state="done", kind="entry", alive=False):
    pair = [{"a": key, "b": opponent, "a_wins": wins, "draws": 0, "b_wins": losses}]
    return {"key": key, "name": "fresh", "kind": kind, "state": state, "alive": alive, "done": wins + losses,
            "shares": [0.5, 0.3, 0.1, 0.1], "pairs": {"weighted": pair, "king": pair, "delta": pair}}


def test_entry_gets_a_provisional_row_and_rating():
    rows = leaderboard.build(ROOT, [_entry("sub:n:fresh", "heuristic", 50, 10)])["rows"]
    row = next(r for r in rows if r["key"] == "sub:n:fresh")
    base = next(r for r in rows if r["key"] == "heuristic")
    assert row["provisional"] and row["elo"] > base["elo"] and row["games"] == 240


def test_stopped_match_is_not_rated():
    rows = leaderboard.build(ROOT, [_entry("sub:n:fresh", "heuristic", 50, 10, state="stopped", alive=False)])["rows"]
    assert all(r["key"] != "sub:n:fresh" for r in rows)


def test_rated_pairing_is_not_counted_twice():
    plain = {r["key"]: r["elo"] for r in leaderboard.build(ROOT)["rows"]}
    bot = next(k for k in plain if k.startswith("sub:") and plain[k] is not None)
    rows = leaderboard.build(ROOT, [_entry(bot, "random", 0, 100, state="playing", kind="rating", alive=True)])["rows"]
    assert {r["key"]: r["elo"] for r in rows}[bot] == plain[bot]


def test_payload_hides_internals(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE", tmp_path)
    monkeypatch.setattr(board, "LIVE", tmp_path)
    monkeypatch.setattr(board, "RESULTS_ROOT", ROOT)
    board._cache["signature"] = None
    match = live.Match(("sub:a:net=/x/n.npz", "heuristic", "sub:a:net=/x/n.npz", "heuristic"), 3, 1, kind="entry")
    match.update([_result(0, 100, 50)])
    data = board.payload()
    assert data["playing"][0]["opponent"] == "Basic rules"
    assert data["playing"][0]["done"] == 1
    assert any(r["key"] == "sub:a:n" for r in data["standings"])
    text = json.dumps(data)
    assert "/x/" not in text and "pid" not in data["playing"][0]


def test_explains_only_bots_on_the_board(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE", tmp_path)
    monkeypatch.setattr(board, "LIVE", tmp_path)
    monkeypatch.setattr(board, "RESULTS_ROOT", ROOT)
    board._cache["signature"] = None
    board.payload()
    assert board.explain("../../etc/passwd") is None
    hybrid = board.explain("hybrid")
    assert "<svg" in hybrid["method"] and "move" not in hybrid


def test_board_shows_steps_for_a_bot_in_testing(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE", tmp_path)
    monkeypatch.setattr(board, "LIVE", tmp_path)
    monkeypatch.setattr(board, "RESULTS_ROOT", ROOT)
    board._cache["signature"] = None
    match = live.Match(("sub:table-pressure:pressure=1", "heuristic", "sub:table-pressure:pressure=1", "heuristic"), 3, 1, kind="entry")
    match.update([_result(0, 100, 50)])
    board.payload()
    assert board.explain("sub:table-pressure:pressure=1")["description"]


def test_every_bot_on_the_board_says_how_it_was_trained():
    from arena.training import TRAINING, training_of

    board_rows = leaderboard.build(ROOT)["rows"]
    missing = [r["key"] for r in board_rows if training_of(r["key"]) is None]
    assert not missing
    assert training_of("sub:bitter-search:rollout=net") is TRAINING["bitter-search"]
    assert "loop" in leaderboard.method_svg([["a", "play"], ["b", "update"]], "5 times")
    assert "loop" not in leaderboard.method_svg([["a", "play"]])


def test_tensorboard_forwarding_drops_identity(monkeypatch):
    from arena import tensorboard
    from starlette.requests import Request

    scope = {"type": "http", "method": "GET", "path": "/tb/x", "query_string": b"", "path_params": {"path": "x"},
             "headers": [(b"remote-user", b"egouda"), (b"cookie", b"secret"), (b"x-home-estimation-key", b"k"), (b"accept", b"*/*")]}
    assert tensorboard._headers(Request(scope)) == {"accept": "*/*"}


def test_the_guide_defines_every_metric_on_both_pages():
    from arena.guide import GUIDE

    for word in ("Weighted rank", "King only", "Score delta", "Elo", "early result", "Benchmark"):
        assert word in GUIDE
    assert GUIDE in leaderboard.render(leaderboard.build(ROOT))


def test_tensorboard_page_is_not_rewritten_by_the_cdn():
    from email.message import Message

    from arena import tensorboard

    page = Message()
    page["Content-Type"] = "text/html; charset=utf-8"
    page["Cache-Control"] = "no-cache, must-revalidate"
    assert tensorboard._reply_headers(page)["Cache-Control"] == "no-cache, must-revalidate, no-transform"
    script = Message()
    script["Content-Type"] = "text/javascript"
    script["Cache-Control"] = "private, max-age=86400"
    assert tensorboard._reply_headers(script)["Cache-Control"] == "private, max-age=86400"


def test_reader_text_has_no_inside_terms():
    from arena.guide import GUIDE
    from arena.plain import BOTS
    from arena.training import TRAINING

    text = json.dumps([BOTS, TRAINING]) + GUIDE
    for word in ("PPO", "PIMC", "GPU", "Heuristic", "self-play", "Rung", "RL v2", "bitter", " net ", "critic"):
        assert word not in text.replace("bitter-net", "").replace("bitter-search", "").replace("bitter-run", ""), word


def test_the_lab_page_has_no_inside_terms():
    page = (ROOT / "arena/static/lab.html").read_text()
    script = (ROOT / "arena/static/lab.js").read_text()
    for word in ("Heuristic", "PIMC", "TensorBoard", "Iteration", "self-play", "GPU", "engine"):
        assert word not in page, word
    for word in ('"Heuristic', "Iteration", "self-play", "GPU", "engines", ">vs PIMC"):
        assert word not in script, word


def test_page_files_stay_out_of_shared_caches():
    import asyncio

    from arena.serve import STATIC, PrivateStatic

    scope = {"type": "http", "method": "GET", "path": "/lab.css", "headers": []}
    response = asyncio.run(PrivateStatic(directory=STATIC).get_response("lab.css", scope))
    assert response.headers["Cache-Control"] == "private, no-cache"


def _experiment(folder, bot):
    pair = {"a": f"sub:{bot}", "b": "random", "a_wins": 3, "draws": 0, "b_wins": 1}
    (folder / "results").mkdir(parents=True, exist_ok=True)
    (folder / "results" / f"{bot}.json").write_text(json.dumps(
        {"bot": bot, "manifest": {"name": bot}, "sets": 4, "played_at": "t", "versus": [],
         "pairs": [pair], "position_pairs": [pair], "king_pairs": [pair]}
    ))  # fmt: skip


def test_an_unmerged_bot_from_the_server_is_provisional(tmp_path):
    _experiment(tmp_path, "my-experiment")
    rows = {r["key"]: r for r in leaderboard.build(ROOT, None, tmp_path)["rows"]}
    assert rows["sub:my-experiment"]["provisional"] is True
    assert rows["sub:lowcard-wire"]["provisional"] is False


def test_official_results_beat_the_servers_copy(tmp_path):
    _experiment(tmp_path, "lowcard-wire")
    official = {r["key"]: r for r in leaderboard.build(ROOT)["rows"]}["sub:lowcard-wire"]
    both = {r["key"]: r for r in leaderboard.build(ROOT, None, tmp_path)["rows"]}["sub:lowcard-wire"]
    assert (both["elo"], both["provisional"]) == (official["elo"], False)


def test_without_experiments_it_matches_the_public_page():
    rows = leaderboard.build(ROOT)["rows"]
    assert [r["key"] for r in rows] == [r["key"] for r in leaderboard.build(ROOT, None, ROOT)["rows"]]
    early = {"sub:" + q.stem for q in (ROOT / "results" / "early").glob("*.json")}
    assert {r["key"] for r in rows if r.get("provisional")} == early


def test_experiments_do_not_move_official_ratings(tmp_path):
    _experiment(tmp_path, "my-experiment")
    fields = ("key", "elo", "king_elo", "score_elo")
    official = {r["key"]: [r[f] for f in fields] for r in leaderboard.build(ROOT)["rows"] if not r.get("provisional")}
    live_match = [_entry("sub:n:fresh", "heuristic", 50, 10)]
    both = {r["key"]: [r[f] for f in fields] for r in leaderboard.build(ROOT, live_match, tmp_path)["rows"]}
    assert {key: both[key] for key in official} == official
    assert "sub:my-experiment" in both and "sub:n:fresh" in both


def test_every_bot_panel_opens(monkeypatch):
    monkeypatch.setattr(board, "RESULTS_ROOT", ROOT)
    monkeypatch.setattr(board, "_cache", {"signature": None, "standings": []})
    for row in board.payload()["standings"]:
        assert board.explain(row["key"]), row["key"]
