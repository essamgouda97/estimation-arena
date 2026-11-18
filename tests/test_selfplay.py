"""The position reward and the per-seat TD(lambda) credit, checked by hand."""

import pytest

pytest.importorskip("optax")

import jax.numpy as jnp  # noqa: E402

from jaxenv.selfplay import position_rewards, td_lambda  # noqa: E402


@pytest.mark.parametrize(
    "scores, want",
    [
        ((50, 10, 0, -5), (1, 0.2, -0.2, -1)),
        ((-5, 0, 10, 50), (-1, -0.2, 0.2, 1)),
        ((50, 50, 0, -5), (0.6, 0.6, -0.2, -1)),  # share King and Sub-King
        ((50, 7, 7, -5), (1, 0, 0, -1)),
        ((7, 7, 7, -5), (1 / 3, 1 / 3, 1 / 3, -1)),
        ((7, 7, 7, 7), (0, 0, 0, 0)),
    ],
)
def test_position_rewards(scores, want):
    got = position_rewards(jnp.array(scores))
    assert jnp.allclose(got, jnp.array(want), atol=1e-6)
    assert abs(float(got.sum())) < 1e-6


def run(values, seats, finished=None, overflow=None, reward=None, lam=0.5):
    t = len(values)
    return td_lambda(
        jnp.array(values, jnp.float32)[:, None],
        jnp.array(seats)[:, None],
        jnp.array(finished or [False] * t)[:, None],
        jnp.array(overflow or [False] * t)[:, None],
        jnp.zeros((t, 1, 4)) if reward is None else jnp.array(reward, jnp.float32)[:, None],
        lam,
    )


def test_each_seat_chains_on_its_own_next_decision():
    # seat 0 at t=0,2; seat 1 at t=1,3. Each seat's last decision has no successor.
    adv, has = run([0.1, 0.2, 0.4, 0.8], [0, 1, 0, 1])
    assert has[:, 0].tolist() == [True, True, False, False]
    assert jnp.allclose(adv[:, 0], jnp.array([0.3, 0.6, 0.0, 0.0]), atol=1e-6)


def test_game_end_gives_each_seat_its_reward_and_lambda_carries_back():
    # seat 0 moves at t=0 and t=1, then the game ends with seat 0 winning.
    r = [[0, 0, 0, 0], [1, -1 / 3, -1 / 3, -1 / 3]]
    adv, has = run([0.2, 0.5], [0, 0], finished=[False, True], reward=r)
    assert has[:, 0].tolist() == [True, True]
    assert abs(float(adv[1, 0]) - 0.5) < 1e-6  # 1 - 0.5
    assert abs(float(adv[0, 0]) - (0.3 + 0.5 * 0.5)) < 1e-6  # (0.5 - 0.2) + lambda * 0.5


def test_a_new_game_cuts_the_chain():
    # t=0 is seat 0's last move of a game that finishes there; t=1 is seat 0 in the next game.
    r = [[-1 / 3, 1, -1 / 3, -1 / 3], [0, 0, 0, 0]]
    adv, has = run([0.1, 0.9], [0, 0], finished=[True, False], reward=r)
    assert has[:, 0].tolist() == [True, False]
    assert abs(float(adv[0, 0]) - (-1 / 3 - 0.1)) < 1e-6  # no value or credit from the next game


def test_an_overflowed_game_gives_no_reward_or_credit():
    r = [[0, 0, 0, 0], [1, 0, 0, 0]]
    adv, has = run([0.2, 0.5], [0, 0], overflow=[False, True], reward=r)
    assert has[1, 0].item() is False  # its final decision has no target at all
    assert has[0, 0].item() is True  # the earlier one bootstraps from it, with no reward
    assert abs(float(adv[0, 0]) - 0.3) < 1e-6


def test_a_batch_gets_the_same_rewards_as_one_game():
    batch = jnp.array([(50, 10, 0, -5), (7, 7, 7, -5)])
    assert jnp.allclose(position_rewards(batch)[1], position_rewards(batch[1]), atol=1e-6)
    assert jnp.allclose(position_rewards(batch)[0], position_rewards(batch[0]), atol=1e-6)


def test_objectives_are_zero_sum_and_pay_what_they_say():
    from jaxenv.selfplay import objective_rewards

    scores = jnp.array([[300, 10, 10, -320], [50, 50, 0, -5]])
    for name in ("weighted", "king", "delta"):
        assert jnp.allclose(objective_rewards(scores, name).sum(1), 0, atol=1e-5)
    king = objective_rewards(scores, "king")
    assert jnp.allclose(king[0], jnp.array([0.75, -0.25, -0.25, -0.25]))
    assert jnp.allclose(king[1], jnp.array([0.25, 0.25, -0.25, -0.25]))
    delta = objective_rewards(scores, "delta")
    assert delta[0, 0] > delta[0, 1] > delta[0, 3]


@pytest.mark.parametrize("objective", ["king", "delta"])
def test_training_runs_with_each_objective(tmp_path, objective):
    from jaxenv.selfplay import main

    args = ["--objective", objective, "--out", str(tmp_path), "--iterations", "1", "--batch", "16", "--steps", "8",
            "--minibatch", "64", "--hidden", "16", "--depth", "1"]  # fmt: skip
    assert main(args) in (0, None)
    assert (tmp_path / "log.jsonl").read_text().strip()


def test_each_length_gets_its_share_of_the_games():
    from jaxenv.selfplay import game_kinds, game_length

    kinds = game_kinds(8, rounds=("5", "mini", "whole"), shares=(0.5, 0.25, 0.25))
    assert kinds.limit.tolist() == [5, 5, 5, 5, 0, 0, 0, 0]
    assert kinds.open_rounds.tolist() == [13, 13, 13, 13, 5, 5, 13, 13]
    assert game_length(kinds).tolist() == [5, 5, 5, 5, 10, 10, 18, 18]
    assert game_kinds(4, rounds=(5,)).limit.tolist() == [5, 5, 5, 5]
    with pytest.raises(ValueError):
        game_kinds(8, rounds=(5, 0), shares=(1.0,))


def test_a_mini_game_is_five_open_rounds_then_the_five_fixed_trumps():
    import jax

    from jaxenv import game as G
    from jaxenv import rules as R

    g = G.new_game(G.deal(jax.random.key(0)), open_rounds=5)
    trumps = []
    for n in range(10):
        trumps.append(int(g.round.fixed_trump))
        done = g.round._replace(phase=jnp.asarray(R.DONE), trump_changed=jnp.asarray(False))  # a round, played
        g = G.advance(g._replace(round=done), G.deal(jax.random.key(n + 1)))
    assert trumps == [-1] * 5 + list(R.FIXED_ORDER)
    assert bool(g.over) and int(g.round_no) == 10


def test_training_counts_the_rounds_left_as_the_bots_do(monkeypatch):
    from arena import lookahead
    from jaxenv.selfplay import game_kinds, rounds_left

    for name, total in (("5", 5), ("whole", 0), ("mini", 10)):
        monkeypatch.setenv(lookahead.GAME_ROUNDS_ENV, str(total))
        kind = game_kinds(1, (name,))
        for round_no in (1, 4, 5, 6, 10, 13, 18, 21):
            assert int(rounds_left(jnp.asarray([round_no]), kind)[0]) == lookahead.rounds_after(round_no)


def test_a_policy_from_before_the_rounds_left_input_plays_as_it_did(tmp_path):
    import jax
    import numpy as np

    from jaxenv import encode as E
    from jaxenv.selfplay import VIEW, apply_policy, init_nets, load_policy, save_npz

    old = init_nets(jax.random.key(0), 16, 2)["policy"]
    old["body"][0]["w"] = old["body"][0]["w"][: E.SIZE]  # as saved before: the raw view alone
    save_npz(old, tmp_path / "old.npz", {})
    new = load_policy(tmp_path / "old.npz", init_nets(jax.random.key(1), 16, 2)["policy"])
    x = np.random.default_rng(0).random((3, VIEW)).astype(np.float32)
    assert jnp.allclose(apply_policy(new, x)[0], apply_policy(old, x[:, : E.SIZE])[0], atol=1e-5)
