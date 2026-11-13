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
    assert data["playing"][0]["opponent"] == "Heuristic"
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
    assert "<svg" in hybrid["method"] and hybrid["move"]
    samples = tmp_path / "samples"
    samples.mkdir()
    frame = {"trump": 1, "estimate": 2, "won": 0, "hand": [1, 2], "trick": [], "legal": [1, 2], "chosen": 2,
             "scores": {"search": {"1": -3.0, "2": 4.0}}}
    (samples / "x.json").write_text(json.dumps({"key": "hybrid", "frames": [frame]}))
    assert "+4.0" in board.explain("hybrid")["move"]


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
