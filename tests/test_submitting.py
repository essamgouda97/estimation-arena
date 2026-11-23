import importlib.util
from pathlib import Path

import pytest

from arena import check
from arena.bots import Manifest
from arena.run import engine_version
from arena.training import TRAINING

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("label_pr", ROOT / "scripts" / "label-pr.py")
label_pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_pr)


def _files(*names, status="modified"):
    return [{"filename": name, "status": status} for name in names]


def test_a_new_bot_is_a_new_strat_and_not_an_update():
    files = _files("bots/new/manifest.json", "bots/new/bot.py", status="added")
    assert label_pr.labels_for(files) == {"new-strat"}


def test_editing_a_bot_is_an_update():
    assert label_pr.labels_for(_files("bots/old/bot.py")) == {"strat-update"}


def test_a_pull_request_can_be_several_things():
    files = _files("bots/new/manifest.json", status="added") + _files("bots/old/bot.py", "models.lock.json", "uv.lock")
    assert label_pr.labels_for(files, engine_changed=True, fork=True) == {
        "new-strat", "strat-update", "model-update", "harness", "engine-update", "fork",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("anchors.json", "anchors"),
        ("arena/run.py", "harness"),
        ("train/imitate.py", "training"),
        ("jaxenv/rlv2.py", "training"),
        (".github/workflows/smoke.yml", "ci"),
        ("docs/metrics.md", "docs"),
    ],
)
def test_each_area_has_its_label(name, label):
    assert label_pr.labels_for(_files(name)) == {label}


def test_every_label_is_defined():
    files = _files("bots/a/manifest.json", status="added") + _files(
        "bots/b/bot.py", "models.lock.json", "anchors.json", "arena/run.py", "train/x.py", ".github/x", "docs/x.md"
    )
    assert label_pr.labels_for(files, True, True) <= set(label_pr.LABELS)


def test_engine_source_pattern_reads_the_lock_line():
    line = 'source = { git = "https://github.com/YosifElessawi/estimation-engine#2dbf68d" }'
    assert label_pr.ENGINE_SOURCE.search(f"x\n{line}\ny").group() == line


@pytest.fixture(autouse=True)
def _known_bot(monkeypatch):
    monkeypatch.setitem(TRAINING, "b", {"summary": "", "stages": []})


def _manifest(tmp_path, **fields):
    folder = tmp_path / "bots" / "b"
    folder.mkdir(parents=True, exist_ok=True)
    defaults = {"author": "A", "description": "d", "method": (("step", "pick"),), "needs": ""}
    return Manifest(name="b", kind="wire", command=("x",), entry="", timeout_s=5, folder=folder, **{**defaults, **fields})


def test_check_passes_a_complete_bot(tmp_path):
    assert check.problems(_manifest(tmp_path), {}, tmp_path) == []


def test_check_asks_for_the_missing_fields(tmp_path):
    found = " ".join(check.problems(_manifest(tmp_path, author="", method=()), {}, tmp_path))
    assert "author" in found and "method" in found


def test_check_asks_how_the_bot_was_trained(tmp_path, monkeypatch):
    monkeypatch.delitem(TRAINING, "b")
    assert "arena/training.py" in check.problems(_manifest(tmp_path), {}, tmp_path)[0]


def test_check_refuses_a_big_file_outside_the_lock(tmp_path):
    manifest = _manifest(tmp_path)
    (manifest.folder / "net.npz").write_bytes(b"0" * (check.MAX_FILE_BYTES + 1))
    assert "too big" in check.problems(manifest, {}, tmp_path)[0]


def test_check_wants_needed_models_pinned(tmp_path):
    manifest = _manifest(tmp_path, needs="runs/m.pt")
    assert "not in models.lock.json" in check.problems(manifest, {}, tmp_path)[0]
    lock = {"runs/m.pt": {"revision": "main", "sha256": "x"}}
    assert "pin" in check.problems(manifest, lock, tmp_path)[0]
    lock = {"runs/m.pt": {"revision": "a" * 40, "sha256": "x"}}
    assert "missing" in check.problems(manifest, lock, tmp_path)[0]
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "m.pt").write_bytes(b"m")
    assert "sha256" in check.problems(manifest, lock, tmp_path)[0]


def test_check_plays_a_real_bot(capsys):
    assert check.main(["lowcard-wire"]) == 0
    assert "ready for a pull request" in capsys.readouterr().out


def test_check_names_an_unknown_bot(capsys):
    assert check.main(["no-such-bot"]) == 1
    assert "problem:" in capsys.readouterr().out


def test_engine_version_names_the_commit():
    # The package version stays 0.1.0 across engine commits, so the ladder needs the commit.
    assert "+" in engine_version()


def test_a_borrowed_file_is_part_of_the_bots_hash(tmp_path):
    from arena.run import bot_hash

    lender = tmp_path / "bots" / "lender"
    lender.mkdir(parents=True)
    alone, borrower = _manifest(tmp_path), _manifest(tmp_path, borrows=("../lender/net.npz",))
    (lender / "net.npz").write_bytes(b"old net")
    before = bot_hash(borrower)
    (lender / "net.npz").write_bytes(b"new net")
    assert bot_hash(borrower) != before  # the lender's new net replays the borrower
    assert bot_hash(alone) == bot_hash(_manifest(tmp_path))  # and nobody else
    assert bot_hash(alone) != before
