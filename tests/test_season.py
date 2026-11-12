import json
import shutil
from pathlib import Path

from arena import leaderboard, season
from arena.run import engine_version

ROOT = Path(__file__).resolve().parents[1]


def _copy(tmp_path: Path) -> Path:
    for name in ("anchors.json", "ladder", "results", "bots"):
        source = ROOT / name
        shutil.copytree(source, tmp_path / name) if source.is_dir() else shutil.copy(source, tmp_path / name)
    return tmp_path


def test_a_season_has_its_own_deals_and_never_gives_away_the_secret():
    assert season.deal_seed(123, 1) == 123  # season 1 was played on the secret itself
    seeds = {season.deal_seed(123, n) for n in (2, 3, 4)}
    assert len(seeds) == 3 and 123 not in seeds
    assert season.deal_seed(123, 2) != season.deal_seed(124, 2)


def test_a_result_says_which_season_it_was_played_in():
    assert season.of("0.1.0+abc/r5") == 1 and season.of("0.1.0+abc") == 1
    assert season.of("0.1.0+abc/r5" + season.tag(3)) == 3


def test_a_new_season_keeps_the_final_standings_and_clears_the_board(tmp_path: Path):
    root = _copy(tmp_path)
    before = leaderboard.build(root, official_only=True)
    ranked = [r["name"] for r in before["rows"] if r["elo"] is not None and not r["anchor"]]
    assert ranked and before["season"]["number"] == season.current(root)
    number = season.start(root, today="2026-11-12")
    assert number == before["season"]["number"] + 1 == season.current(root)
    assert engine_version(root=root).endswith(season.tag(number))
    after = leaderboard.build(root, official_only=True)
    assert not [r for r in after["rows"] if not r["anchor"] and r["elo"] is not None]  # every bot starts again
    assert [r for r in after["rows"] if r["anchor"] and r["elo"] is not None]  # benchmarks keep the scale
    final = after["season"]["ended"][0]
    assert final["ended"] == "2026-11-12"
    assert [r["name"] for r in final["rows"] if not r["anchor"]] == ranked
    page = leaderboard.render(after)
    assert f"Season {number}, since 2026-11-12" in page and "final standings" in page
    assert "seed" not in json.dumps(final)


def test_ratings_come_with_a_margin():
    board = leaderboard.build(ROOT, official_only=True)
    ranked = [r for r in board["rows"] if r["elo"] is not None and not r.get("provisional")]
    assert all(r["elo_margin"] is not None and r["elo_margin"] > 0 for r in ranked if r["key"] != board["anchor"])
    assert ranked[0]["close_to"] is None  # nobody is above the leader
    assert "±" in leaderboard.render(board)
