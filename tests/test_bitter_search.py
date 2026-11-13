from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from arena.bots import ROOT
from estimation.arena.duplicate import play_set
from estimation.bots.hybrid.rollout import Start, _beats
from estimation.bots.knowledge import read_knowledge
from estimation.engine import Phase, RulesConfig
from jaxenv.encode_np import PLAY_BASE, SIZE, encode


def _bot_module():
    spec = spec_from_file_location("bitter_search_under_test", ROOT / "bots" / "bitter-search" / "bot.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search = _bot_module()


def _random_net(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    path = tmp_path / "net.npz"
    np.savez(path, w0=rng.normal(size=(SIZE, 32)).astype(np.float32), b0=np.zeros(32, np.float32),
             pw=rng.normal(size=(32, 120)).astype(np.float32), pb=np.zeros(120, np.float32))  # fmt: skip
    return str(path)


class Recorder:
    """Plays the net's top legal action and keeps what it saw."""

    def __init__(self, net, seen):
        self.net, self.seen = net, seen

    def act(self, obs, legal):
        logits = self.net(encode(obs, legal))
        action = max(legal, key=lambda a: logits[a])
        if obs.phase is Phase.PLAY:
            self.seen.append((obs, list(legal), action))
        return action


@pytest.fixture(scope="module")
def game(tmp_path_factory):
    """The card decisions of one duplicate set of net-against-itself, grouped by round."""
    net = search.load_net(_random_net(tmp_path_factory.mktemp("net")))
    seen = []
    play_set(("a", "b", "c", "d"), 7, 0, RulesConfig(), make=lambda name, seed, rules: Recorder(net, seen))
    rounds = []
    for row in seen:
        if not row[0].tricks and not row[0].trick:
            rounds.append([])
        rounds[-1].append(row)
    assert rounds
    return net, rounds


def test_the_rollout_view_equals_the_encoder_at_every_decision(game):
    _, rounds = game
    for decisions in rounds:
        templates = search.templates_of(decisions[0][0])
        for obs, legal, _ in decisions:
            know = read_knowledge(obs)
            cards = [c for _, c in obs.trick] + [0] * (4 - len(obs.trick))
            mask = sum(1 << (a - PLAY_BASE) for a in legal)
            x = np.empty((1, SIZE), np.float32)
            search.write_view(
                x, templates, np.array([obs.hand], np.uint64), np.array([mask], np.uint64),
                np.array([know.played_by], np.uint64), np.array([know.voids], np.uint8), np.array([obs.seat]),
                np.array([cards]), len(obs.trick), np.array([obs.leader]), np.array([obs.tricks_won]),
            )  # fmt: skip
            np.testing.assert_array_equal(x[0], encode(obs, legal))


def test_a_rollout_replays_the_real_round(game):
    net, rounds = game
    for decisions in rounds:
        first, _, action = decisions[0]
        dealt = [0, 0, 0, 0]
        for obs, _, _ in decisions:
            dealt[obs.seat] = dealt[obs.seat] or obs.hand
        start = Start(trick=[], leader=first.leader, won=first.tricks_won, tricks_done=0,
                      estimates=(0, 0, 0, 0), trump=first.trump)  # fmt: skip
        won = search.play_out(
            net, search.templates_of(first), start, np.array([dealt], np.uint64), np.zeros((1, 4), np.uint64),
            np.zeros((1, 4), np.uint8), np.array([action - PLAY_BASE]),
        )[0]  # fmt: skip
        last, _, final = decisions[-1]
        assert len(decisions) == 52
        plays = [c for _, c in last.trick] + [final - PLAY_BASE]
        best = 0
        for i, c in enumerate(plays):
            if _beats(c, plays[best], first.trump):
                best = i
        expected = list(last.tricks_won)
        expected[(last.leader + best) % 4] += 1
        assert won.tolist() == expected


def test_it_plays_a_set(tmp_path):
    if not (ROOT / "runs" / "m4-rules-v2" / "checkpoints" / "final.pt").exists():
        pytest.skip("needs runs/")
    from arena.bots import maker

    spec = f"sub:bitter-search:net={_random_net(tmp_path)}:samples=2:top_k=2"
    result, _ = play_set((spec, "random", spec, "random"), 1, 0, RulesConfig(), make=maker)
    assert len(result.scores) == 4
