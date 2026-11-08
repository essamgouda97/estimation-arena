import json
from pathlib import Path

from estimation.arena.ratings import PairRecord

from arena import direct, duels, leaderboard
from arena.bots import load_manifest
from arena.run import bot_hash, engine_version

ROOT = Path(__file__).resolve().parents[1]


def _pair(a, b, a_wins, draws=0, b_wins=0):
    return {"a": f"sub:{a}", "b": f"sub:{b}", "a_wins": a_wins, "draws": draws, "b_wins": b_wins}


def _match(a="a", b="b", a_wins=15, b_wins=5, hash_a="h", hash_b="h", engine="e"):
    pair = _pair(a, b, a_wins, 0, b_wins)
    return {"a": a, "b": b, "hash_a": hash_a, "hash_b": hash_b, "engine": engine, "batches": [],
            "pairs": [pair], "position_pairs": [pair], "king_pairs": [pair]}


def _rated(bot, bot_hash="h", engine="e"):
    return {"bot": bot, "bot_hash": bot_hash, "engine": engine}


def test_seeds_differ_by_pair_and_batch_but_not_by_order():
    assert duels.pair_seed(1, "a", "b", 0) == duels.pair_seed(1, "b", "a", 0)
    seeds = {duels.pair_seed(1, "a", "b", 0), duels.pair_seed(1, "a", "c", 0), duels.pair_seed(1, "a", "b", 1)}
    assert len(seeds) == 3


def test_a_changed_bot_or_engine_voids_its_matches():
    match = _match()
    assert direct.current([match], [_rated("a"), _rated("b")]) == [match]
    assert direct.current([match], [_rated("a", "new"), _rated("b")]) == []
    assert direct.current([match], [_rated("a"), _rated("b", engine="new")]) == []
    assert direct.current([match], [_rated("a")]) == []


def test_clear_needs_a_real_margin():
    assert direct.is_clear(14, 6) is False
    assert direct.is_clear(15, 5) is True
    assert direct.is_clear(20, 0) is True
    assert direct.is_clear(0, 0) is False


def test_record_reads_either_orientation():
    match = _match(a_wins=3, b_wins=1)
    match["position_pairs"].append(_pair("b", "a", 2, 1, 4))
    assert direct.record(match) == (7, 1, 3)


def test_runoff_is_not_won_by_beating_weak_bots():
    # "farmer" crushes the weak bots but loses to "true", which only edges them.
    records = [
        PairRecord("sub:farmer", "sub:w1", 20, 0, 0), PairRecord("sub:farmer", "sub:w2", 20, 0, 0),
        PairRecord("sub:true", "sub:w1", 11, 0, 9), PairRecord("sub:true", "sub:w2", 11, 0, 9),
        PairRecord("sub:true", "sub:farmer", 15, 0, 5), PairRecord("sub:w1", "sub:w2", 10, 0, 10),
    ]
    keys = ["sub:farmer", "sub:true", "sub:w1", "sub:w2"]
    assert direct.runoff(keys, records)[0] == "sub:true"


def test_runoff_lists_everyone_once():
    keys = ["sub:a", "sub:b", "sub:c"]
    assert sorted(direct.runoff(keys, [])) == sorted(keys)


def test_a_real_direct_match_changes_the_board_and_stays_secret(tmp_path, monkeypatch):
    import shutil

    for name in ("anchors.json", "ladder"):
        source = ROOT / name
        shutil.copytree(source, tmp_path / name) if source.is_dir() else shutil.copy(source, tmp_path / name)
    (tmp_path / "results").mkdir()
    for bot in ("lowcard-wire", "highcard-wire"):
        result = json.loads((ROOT / "results" / f"{bot}.json").read_text())
        result["bot_hash"] = bot_hash(load_manifest(bot))
        result["engine"] = engine_version()
        (tmp_path / "results" / f"{bot}.json").write_text(json.dumps(result))
    assert duels.due(tmp_path) in ([("lowcard-wire", "highcard-wire")], [("highcard-wire", "lowcard-wire")])
    match = duels.play("lowcard-wire", "highcard-wire", 7, workers=1, sets=1, root=tmp_path)
    assert "seed" not in match
    assert len(duels.due(tmp_path)) == 1  # one set can't be clear, so this top pair plays on
    monkeypatch.setattr(duels, "MAX_SETS", 1)
    assert duels.due(tmp_path) == []  # until it reaches the limit
    board = leaderboard.build(tmp_path)
    assert len(board["direct"]) == 1 and board["direct"][0]["sets"] == 1
    assert set(board["runoff"]) >= {"sub:lowcard-wire", "sub:highcard-wire"}
    assert "Head to head" in leaderboard.render(board)


def test_a_board_without_direct_games_hides_the_section():
    assert leaderboard.head_to_head({"direct": [], "rows": [], "runoff": []}) == ""


def test_the_app_notices_a_new_direct_match(tmp_path):
    from arena import board

    (tmp_path / "results" / "direct").mkdir(parents=True)
    before = board._signature(tmp_path)
    (tmp_path / "results" / "direct" / "a.vs.b.json").write_text("{}")
    assert board._signature(tmp_path) != before


def test_game_files_are_safe_and_order_free():
    assert direct.game_file("hybrid-v4", "RL v2") == direct.game_file("RL v2", "hybrid-v4") == "hybrid-v4.vs.rl-v2.json"


def test_a_watchable_game_has_every_card_and_move(tmp_path):
    from arena import replay

    game = replay.record_game(["sub:lowcard-wire", "sub:highcard-wire", "sub:lowcard-wire", "sub:highcard-wire"], 5, rounds=2)
    assert len(game["rounds"]) == 2
    first = game["rounds"][0]
    assert sorted(c for hand in first["hands"] for c in hand) == list(range(52))
    played = [a - 68 for _, a in first["moves"] if a >= 68]
    assert not played or sorted(played) == list(range(52))
    assert first["after"] != first["before"] or not played


def test_public_game_seeds_are_per_pair_and_order_free():
    from arena import replay

    assert replay.game_seed("a", "b") == replay.game_seed("b", "a") != replay.game_seed("a", "c")


def test_the_browser_lists_every_pairing_with_its_set_margins():
    board = leaderboard.build(ROOT)
    data = leaderboard.browser_data(board, ROOT / "site" / "games")
    hybrid = [m for m in data["matchups"] if "sub:hybrid-v4" in (m["a"], m["b"])]
    assert len(hybrid) >= 6
    with_sets = next(m for m in hybrid if m["kind"] == "benchmark")
    assert with_sets["sets"] == len(with_sets["detail"]) or not with_sets["detail"]
    assert all(len(s["games"]) == 4 for s in with_sets["detail"])


def test_the_app_notices_a_new_watchable_game(tmp_path):
    from arena import board

    (tmp_path / "site" / "games").mkdir(parents=True)
    before = board._signature(tmp_path)
    (tmp_path / "site" / "games" / "a.vs.b.json").write_text("{}")
    assert board._signature(tmp_path) != before


def _early_board(tmp_path):
    import shutil

    for name in ("anchors.json", "ladder"):
        source = ROOT / name
        shutil.copytree(source, tmp_path / name) if source.is_dir() else shutil.copy(source, tmp_path / name)
    shutil.copytree(ROOT / "results", tmp_path / "results")
    pair = {"a": "sub:llm-claude-api", "b": "heuristic", "a_wins": 1, "draws": 0, "b_wins": 0}
    entry = {"anchor": "Heuristic", "spec": "heuristic", "sets": 1, "games": 4, "mean_score": 1.0, "positions": [0.5, 0.25, 0.15, 0.1],
             "rsa": {"mean": 1.0, "low": 0.0, "high": 2.0, "sets": 1}, "first_place": {"mean": 0.5, "low": 0.0, "high": 1.0, "sets": 1},
             "position_reward": {"mean": 0.3}, "delta": {"mean": 10.0}, "timing": {"count": 1, "p95_s": 1.0, "over_budget": 0}}  # fmt: skip
    early = {"bot": "llm-claude-api", "bot_hash": "h", "engine": "e", "sets": 1, "played_at": "2026-11-07T00:00:00+00:00",
             "manifest": {"name": "llm-claude-api", "author": "Essam", "kind": "wire"}, "versus": [entry],
             "pairs": [pair], "position_pairs": [pair], "king_pairs": [pair]}  # fmt: skip
    (tmp_path / "results" / "early").mkdir(exist_ok=True)
    (tmp_path / "results" / "early" / "llm-claude-api.json").write_text(json.dumps(early))
    return leaderboard.build(tmp_path)


def test_an_early_result_shows_without_moving_official_ratings(tmp_path):
    board = _early_board(tmp_path)
    plain = leaderboard.build(ROOT, official_only=True)
    fields = ("key", "elo", "king_elo", "score_elo")
    shown = {r["key"]: r for r in board["rows"]}
    assert all([shown[o["key"]][f] for f in fields] == [o[f] for f in fields] for o in plain["rows"])
    assert shown["sub:llm-claude-api"]["provisional"] is True
    assert shown["sub:llm-claude-api"]["elo"] is not None
    assert board["runoff"] == plain["runoff"]


def test_an_early_bot_is_on_the_page_without_a_rank_and_has_a_browser_page(tmp_path):
    board = _early_board(tmp_path)
    page = leaderboard.render(board)
    assert "early result, 4 games" in page and 'bot early"' in page
    assert any(m["a"] == "sub:llm-claude-api" and m["b"] == "heuristic" for m in board["matchups"])
    assert next(r for r in board["rows"] if r["key"] == "sub:llm-claude-api")["name"] == "Claude Opus 5.5"


def test_the_ladder_never_plays_an_early_bot(tmp_path):
    _early_board(tmp_path)
    assert not any("llm-claude-api" in pair for pair in duels.due(tmp_path))


def test_recorded_game_scores_match_the_arena():
    from estimation.arena.duplicate import bot_seed, deal_seed
    from estimation.arena.match import play_bola
    from estimation.engine.config import DEFAULT_CONFIG

    from arena.bots import maker
    from arena.replay import record_game

    seats = ["heuristic", "random", "heuristic", "random"]
    bots = lambda: [maker(s, bot_seed(2000, 0, 0, k), DEFAULT_CONFIG) for k, s in enumerate(seats)]  # noqa: E731
    real = play_bola(deal_seed(2000, 0), bots(), DEFAULT_CONFIG)
    assert record_game(seats, deal_seed(2000, 0), bots=bots())["final"] == list(real.scores)
