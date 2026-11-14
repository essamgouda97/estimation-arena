"""Playing for the game: the position value of a round, and looking ahead on the bid."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from arena import lookahead
from arena.bots import ROOT
from arena.positions import position_reward
from estimation.arena.duplicate import play_set
from estimation.bots.hybrid.beliefs import PublicBidding
from estimation.engine import Phase, RulesConfig
from jaxenv.encode_np import SIZE, encode


def _bot_module():
    spec = spec_from_file_location("bitter_search_lookahead", ROOT / "bots" / "bitter-search" / "bot.py")
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


def test_with_no_rounds_left_the_value_is_the_position_reward():
    for scores in ((10, 40, -5, 0), (7, 7, 3, 3), (0, 0, 0, 0), (5, 9, 9, 9)):
        for seat in range(4):
            assert lookahead.position_value(scores, seat) == pytest.approx(position_reward(scores, seat))


def test_rounds_to_come_pull_the_value_toward_even():
    ahead = lookahead.position_value((60, 20, 10, 0), 0)
    assert ahead == 1.0
    assert 0 < lookahead.position_value((60, 20, 10, 0), 0, spread=40.0) < ahead
    values = [lookahead.position_value((60, 20, 10, 0), seat, spread=40.0) for seat in range(4)]
    assert sum(values) == pytest.approx(0.0, abs=1e-9)
    assert values == sorted(values, reverse=True)


def test_the_game_length_comes_from_the_arena(monkeypatch):
    monkeypatch.setenv(lookahead.GAME_ROUNDS_ENV, "5")
    assert [lookahead.rounds_after(r) for r in (1, 4, 5, 6)] == [4, 1, 0, 0]
    monkeypatch.setenv(lookahead.GAME_ROUNDS_ENV, "0")
    assert lookahead.rounds_after(1) == lookahead.WHOLE_GAME - 1
    monkeypatch.delenv(lookahead.GAME_ROUNDS_ENV)
    assert lookahead.rounds_after(18) == 0


class Recorder:
    """Plays the net's top legal action and keeps every decision it faced."""

    def __init__(self, net, seen):
        self.net, self.seen = net, seen

    def act(self, obs, legal):
        logits = self.net(encode(obs, legal))
        action = max(legal, key=lambda a: logits[a])
        self.seen.append((obs, list(legal), action))
        return action


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    """Every decision of one duplicate set of net-against-itself, grouped by round."""
    net = search.load_net(_random_net(tmp_path_factory.mktemp("net")))
    seen, grouped = [], []
    play_set(("a", "b", "c", "d"), 7, 0, RulesConfig(), make=lambda name, seed, rules: Recorder(net, seen))
    for row in seen:
        if not row[0].log:
            grouped.append([])
        grouped[-1].append(row)
    return net, grouped


def test_the_replayed_bidding_shows_each_seat_what_the_game_showed(rounds):
    _, grouped = rounds
    checked = 0
    for decisions in grouped:
        for obs, legal, _ in decisions:
            if obs.phase is Phase.PLAY:
                continue
            state = PublicBidding(obs, RulesConfig())
            for seat, action in obs.log:
                state.step(seat, action)
            assert state.cur == obs.seat and state.legal_actions() == legal
            np.testing.assert_array_equal(encode(replace(state.observation(obs.seat), hand=obs.hand), legal), encode(obs, legal))
            checked += 1
    assert checked > 50


def test_trying_a_call_replays_the_real_round(rounds):
    net, grouped = rounds
    played = 0
    for decisions in grouped:
        first, _, call = decisions[0]
        hands = [0, 0, 0, 0]
        for obs, _, _ in decisions:
            hands[obs.seat] = hands[obs.seat] or obs.hand
        cards = [row for row in decisions if row[0].phase is Phase.PLAY]
        if len(cards) != 52 or not all(hands):
            continue  # a skipped round
        state = PublicBidding(first, RulesConfig())
        state.step(first.seat, call)
        search.finish_bidding(net, state, hands)
        last = cards[-1][0]
        view = state.observation(last.seat)
        assert (view.estimates, view.trump, view.leader) == (last.estimates, last.trump, cards[0][0].leader)
        won = search.play_round(net, view, hands)
        assert sum(won) == 13 and all(w >= have for w, have in zip(won, last.tricks_won))
        assert sum(w - have for w, have in zip(won, last.tricks_won)) == 1
        played += 1
    assert played


def test_it_plays_a_set_for_position_with_bid_look_ahead(tmp_path):
    if not (ROOT / "runs" / "m4-rules-v2" / "checkpoints" / "final.pt").exists():
        pytest.skip("needs runs/")
    from arena.bots import maker

    spec = f"sub:bitter-search:net={_random_net(tmp_path)}:samples=2:top_k=2:objective=position:bid_search=true:bid_samples=2"
    result, _ = play_set((spec, "random", spec, "random"), 1, 0, RulesConfig(), make=maker)
    assert len(result.scores) == 4


def _standings_net(tmp_path: Path) -> str:
    """A policy file with a standings net that returns minus the largest lead over the seat."""
    path = tmp_path / "judge.npz"
    first = np.zeros((5, 2), np.float32)
    first[0] = (1.0, -1.0)  # relu(x) - relu(-x) == x, for the largest lead
    np.savez(path, w0=np.zeros((SIZE, 4), np.float32), b0=np.zeros(4, np.float32),
             pw=np.zeros((4, 120), np.float32), pb=np.zeros(120, np.float32),
             sw0=first, sb0=np.zeros(2, np.float32), sw1=np.array([[-1.0], [1.0]], np.float32), sb1=np.zeros(1, np.float32))  # fmt: skip
    return str(path)


def test_the_standings_net_reads_the_scores_from_a_seats_side(tmp_path, monkeypatch):
    monkeypatch.setenv(lookahead.GAME_ROUNDS_ENV, "5")
    judge = lookahead.standings_net(_standings_net(tmp_path))
    assert judge([100, 300, 0, 0], 0, 2, 1) == pytest.approx(-1.0)  # 200 behind the leader
    assert judge([100, 300, 0, 0], 1, 2, 1) == pytest.approx(1.0)  # 200 clear of the next seat
    assert lookahead.standings_view([0, 0, 0, 0], 0, 0.4, 4).tolist() == pytest.approx([0, 0, 0, 0.4, 0.2])


def test_a_policy_without_a_standings_net_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no standings net"):
        lookahead.standings_net(_random_net(tmp_path))


def test_play_and_training_read_the_standings_alike():
    jnp = pytest.importorskip("jax.numpy")
    from jaxenv import selfplay

    scores, carry = [120, -40, 260, 15], 2
    trained_on = np.asarray(selfplay.standings_view(jnp.array(scores), jnp.array(0.6), jnp.array(carry)))
    for seat in range(4):
        assert lookahead.standings_view(scores, seat, 0.6, carry) == pytest.approx(trained_on[seat], abs=1e-6)


def test_it_plays_a_set_by_the_standings_net(tmp_path):
    if not (ROOT / "runs" / "m4-rules-v2" / "checkpoints" / "final.pt").exists():
        pytest.skip("needs runs/")
    from arena.bots import maker

    spec = f"sub:bitter-search:net={_standings_net(tmp_path)}:samples=2:top_k=2:objective=standings"
    result, _ = play_set((spec, "random", spec, "random"), 1, 0, RulesConfig(), make=maker)
    assert len(result.scores) == 4
