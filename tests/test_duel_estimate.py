"""The ladder starts a direct match only if its estimated time fits what is left of the job."""

import json

from arena.duels import MARGIN, estimate, match_file
from arena.run import RESULTS_DIR
from arena.bots import ROOT


def _result(root, name, versus):
    folder = root / RESULTS_DIR.relative_to(ROOT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps({"versus": versus}))


def test_estimate_adds_both_bots_cost_against_random(tmp_path):
    _result(tmp_path, "slow", [{"spec": "random", "sets": 20, "wall_s": 4000.0}, {"spec": "pimc", "sets": 20, "wall_s": 9000.0}])
    _result(tmp_path, "quick", [{"spec": "random", "sets": 20, "wall_s": 1000.0}])
    assert estimate("slow", "quick", root=tmp_path) == MARGIN * 5000.0


def test_an_earlier_longer_batch_wins(tmp_path):
    _result(tmp_path, "a", [{"spec": "random", "sets": 20, "wall_s": 100.0}])
    _result(tmp_path, "b", [{"spec": "random", "sets": 20, "wall_s": 100.0}])
    path = match_file(tmp_path, "a", "b")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"batches": [{"sets": 20, "wall_s": 3000.0}]}))
    assert estimate("b", "a", root=tmp_path) == MARGIN * 3000.0


def test_unknown_bots_cost_nothing(tmp_path):
    assert estimate("x", "y", root=tmp_path) == 0.0
