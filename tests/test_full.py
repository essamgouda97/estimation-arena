"""The weekly full-game ladder: who plays, which pairs, and its own page."""

import json
import shutil
from pathlib import Path

from arena import full
from arena.bots import load_manifest
from arena.run import FULL_GAME, bot_hash, engine_version

ROOT = Path(__file__).resolve().parents[1]
BOTS = ("lowcard-wire", "highcard-wire")


def _root(tmp_path):
    for name in ("anchors.json", "ladder"):
        source = ROOT / name
        shutil.copytree(source, tmp_path / name) if source.is_dir() else shutil.copy(source, tmp_path / name)
    (tmp_path / "results").mkdir()
    for bot in BOTS:
        result = json.loads((ROOT / "results" / f"{bot}.json").read_text())
        result |= {"bot_hash": bot_hash(load_manifest(bot)), "engine": engine_version()}
        (tmp_path / "results" / f"{bot}.json").write_text(json.dumps(result))
    return tmp_path


def _save_match(root, a_wins, b_wins, engine=None):
    a, b = sorted(BOTS)
    pair = {"a": f"sub:{a}", "b": f"sub:{b}", "a_wins": a_wins, "draws": 0, "b_wins": b_wins}
    match = {"a": a, "b": b, "hash_a": bot_hash(load_manifest(a)), "hash_b": bot_hash(load_manifest(b)),
             "engine": engine or engine_version(FULL_GAME), "batches": [], "pairs": [pair], "position_pairs": [pair], "king_pairs": [pair]}  # fmt: skip
    path = root / full.FULL_DIR / f"{a}.vs.{b}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(match))


def test_contenders_are_rated_bots_only(tmp_path):
    root = _root(tmp_path)
    assert sorted(full.contenders(root)) == sorted(BOTS)
    assert len(full.contenders(root, top=1)) == 1


def test_a_new_pair_plays_and_a_settled_pair_rests(tmp_path):
    root = _root(tmp_path)
    assert [tuple(sorted(p)) for p in full.due(root)] == [tuple(sorted(BOTS))]
    _save_match(root, 18, 2)
    assert full.due(root) == []  # clear
    _save_match(root, 10, 10)
    assert len(full.due(root)) == 1  # too close: another batch next week


def test_micro_matches_do_not_count_as_whole_games(tmp_path):
    root = _root(tmp_path)
    _save_match(root, 18, 2, engine=engine_version())
    assert full.matches(root) == [] and len(full.due(root)) == 1


def test_the_page_rates_the_pair_around_the_scale(tmp_path):
    root = _root(tmp_path)
    assert full.build(root)["rows"] == []
    assert "No whole games yet" in full.render(full.build(root))
    _save_match(root, 15, 5)
    board = full.build(root)
    elos = [r["elo"] for r in board["rows"]]
    assert len(elos) == 2 and abs(sum(elos) / 2 - full.SCALE) < 1e-6 and elos[0] > elos[1]
    assert board["direct"][0]["sets"] == 20 and set(board["runoff"]) == {f"sub:{b}" for b in BOTS}
    page = full.render(board)
    assert "Whole games" in page and "Head to head" in page
    assert {r["author"] for r in board["rows"]} == {"Essam Gouda"} and '<td class="author">Essam Gouda</td>' in page
    full.write(board, tmp_path / "site")
    assert (tmp_path / "site" / "full" / "index.html").is_file()
