"""Micro games: the ladder plays a game's first 5 rounds, and results say which length they are."""

from estimation.arena.duplicate import bot_seed, deal_seed
from estimation.arena.match import play_bola
from estimation.arena.timing import Timing
from estimation.engine.config import DEFAULT_CONFIG

from arena import direct
from arena.bots import maker
from arena.replay import record_game
from arena.run import FULL_GAME, GAME_ROUNDS, base_engine, engine_version, play_game, play_set

SEATS = ["heuristic", "random", "heuristic", "random"]


def _bots():
    return [maker(s, bot_seed(2000, 0, 0, k), DEFAULT_CONFIG) for k, s in enumerate(SEATS)]


def test_a_micro_game_stops_after_five_rounds():
    game = play_game(deal_seed(2000, 0), _bots(), DEFAULT_CONFIG, [Timing() for _ in SEATS])
    assert GAME_ROUNDS == 5 and len(game.history) == 5 and not game.is_over
    assert all(r <= 13 for r in (game.round_no,))  # still in the open rounds


def test_a_micro_set_plays_five_rounds_in_every_rotation():
    result, timings = play_set(SEATS, 2000, 0)
    assert result.rounds == (5, 5, 5, 5)
    assert all(t.count for t in timings.values())


def test_micro_scores_match_the_watchable_game():
    game = play_game(deal_seed(2000, 0), _bots(), DEFAULT_CONFIG, [Timing() for _ in SEATS])
    watched = record_game(SEATS, deal_seed(2000, 0), rounds=GAME_ROUNDS, bots=_bots())
    assert len(watched["rounds"]) == GAME_ROUNDS
    assert watched["final"] == list(game.scores)


def test_a_full_game_matches_the_engine():
    game = play_game(deal_seed(2000, 0), _bots(), DEFAULT_CONFIG, [Timing() for _ in SEATS], FULL_GAME)
    real = play_bola(deal_seed(2000, 0), _bots(), DEFAULT_CONFIG)
    assert game.is_over and game.scores == real.scores and len(game.history) == real.rounds


def test_the_engine_key_names_the_game_length():
    micro, full = engine_version(), engine_version(FULL_GAME)
    assert micro == f"{full}/r5" and "/" not in full
    assert base_engine(micro) == base_engine(full) == full


def test_full_game_matches_count_only_on_the_full_ladder():
    full, micro = engine_version(FULL_GAME), engine_version()
    match = {"a": "a", "b": "b", "hash_a": "h", "hash_b": "h", "engine": full}
    results = [{"bot": b, "bot_hash": "h", "engine": micro} for b in ("a", "b")]
    assert direct.current([match], results) == []  # the nightly ladder ignores it
    assert direct.current([match], results, engine=full) == [match]
    assert direct.current([{**match, "engine": micro}], results, engine=full) == []
