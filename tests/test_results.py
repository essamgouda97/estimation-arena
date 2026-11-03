from arena.run import keep_unplayed


def _result(versus, pairs, bot_hash="h1", engine="0.1.0", played_at="t"):
    return {"bot": "b", "bot_hash": bot_hash, "engine": engine, "played_at": played_at,
            "versus": versus, "pairs": pairs}


OLD = _result(
    [{"anchor": "Random", "spec": "random"}, {"anchor": "Hybrid", "spec": "hybrid"}],
    [{"a": "sub:b", "b": "random"}, {"a": "sub:b", "b": "hybrid"}, {"a": "hybrid", "b": "sub:b"}],
    played_at="old",
)


def test_keeps_games_against_anchors_not_played_now():
    new = _result([{"anchor": "Random", "spec": "random"}], [{"a": "sub:b", "b": "random"}])
    assert keep_unplayed(new, OLD) == ["Hybrid"]
    assert [v["spec"] for v in new["versus"]] == ["random", "hybrid"]
    assert new["versus"][1]["carried_from"] == "old"
    assert len(new["pairs"]) == 3


def test_drops_them_when_the_bot_changed():
    new = _result([{"anchor": "Random", "spec": "random"}], [], bot_hash="h2")
    assert keep_unplayed(new, OLD) == []
    assert len(new["versus"]) == 1
