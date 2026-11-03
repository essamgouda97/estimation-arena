"""record -> imitate -> play, end to end on a tiny teacher (a few seconds)."""

import numpy as np

from arena.bots import load_manifest, make_submission
from estimation.engine import Bola, RulesConfig
from train.imitate import main as imitate
from train.record import record_chunk


def test_record_train_play(tmp_path, monkeypatch):
    # A smoke test of the pipeline, not of the GPU: torch can see a GPU it cannot run on
    # (the home server's Pascal card with a newer torch), so keep this on the CPU.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    games, rows = record_chunk(range(2), "pimc:samples=2", None, 1.0, tmp_path)
    assert games == 2 and rows > 500
    shard = np.load(next(tmp_path.glob("*.npz")))
    assert shard["features"].shape == (rows, shard["features"].shape[1])
    assert shard["legal"][np.arange(rows), shard["choice"]].all()  # the teacher plays legal cards

    net = tmp_path / "net.npz"
    imitate(["--data", str(tmp_path), "--out", str(net), "--epochs", "1", "--hidden", "32", "--val", "0.5"])

    bot = make_submission(load_manifest("pimc-clone"), 0, RulesConfig(), {"net": str(net)})
    bola = Bola(7)
    while not bola.is_over:
        seat = bola.current_player
        bola.step(bot.act(bola.observation(seat), bola.legal_actions()))  # raises if illegal


def test_suit_tables_relabel_a_decision_consistently(tmp_path):
    import torch

    from estimation.engine import Bola, Phase
    from estimation.engine.actions import PLAY_BASE
    from estimation.rl.encoder import encode
    from estimation.rl.symmetry import SUIT_PERMUTATIONS

    t = np.load("train/suit_tables.npz")
    bola = Bola(5)
    from estimation.bots import HeuristicBot

    h = HeuristicBot()
    while bola.phase is not Phase.PLAY or len(bola.legal_actions()) < 2:
        seat = bola.current_player
        bola.step(h.act(bola.observation(seat), bola.legal_actions()))
    obs = bola.observation(bola.current_player)
    legal = np.zeros(52, bool)
    legal[[a - PLAY_BASE for a in bola.legal_actions()]] = True
    x = encode(obs)
    for p in range(24):
        relabelled_legal = legal[t["card_source"][p]]
        assert relabelled_legal.sum() == legal.sum()
        for c in np.flatnonzero(legal):  # a legal card maps to a legal card
            assert relabelled_legal[t["card_destination"][p][c]]
        assert np.allclose(np.sort(x[t["feature_source"][p]]), np.sort(x))  # same values, moved
    assert len(SUIT_PERMUTATIONS) == 24
