from estimation.arena.ratings import PairRecord, bradley_terry

from arena.margins import Margins


def _fit(records):
    return Margins(records, bradley_terry(records))


def test_more_sets_make_a_rating_surer():
    few = [PairRecord("random", "a", 4, 0, 6), PairRecord("a", "b", 6, 0, 4), PairRecord("random", "b", 3, 0, 7)]
    many = [PairRecord(p.a, p.b, p.a_wins * 10, 0, p.b_wins * 10) for p in few]
    field = ["a", "b", "random"]
    assert _fit(many).of("a", field) < _fit(few).of("a", field) / 2


def test_a_narrow_lead_is_too_close_and_a_wide_one_is_not():
    close = [PairRecord("random", "a", 2, 0, 18), PairRecord("random", "b", 2, 0, 18), PairRecord("a", "b", 11, 0, 9)]
    apart = [PairRecord("random", "a", 2, 0, 18), PairRecord("random", "b", 2, 0, 18), PairRecord("a", "b", 36, 0, 4)]
    assert not _fit(close).separates("a", "b")
    assert _fit(apart).separates("a", "b")


def test_doubt_about_weak_bots_does_not_blur_the_top():
    """a and b are both far above Random on little evidence, but have met each other often."""
    records = [PairRecord("random", "a", 0, 0, 5), PairRecord("random", "b", 0, 0, 5), PairRecord("a", "b", 140, 0, 60)]
    margins = _fit(records)
    assert margins.separates("a", "b")
    assert margins.of("a", ["a", "b"]) < margins.of("a", ["random"])


def test_an_unrated_bot_has_no_margin():
    margins = _fit([PairRecord("random", "a", 4, 0, 6)])
    assert margins.of("stranger", ["a", "random"]) is None
    assert not margins.separates("a", "stranger")
