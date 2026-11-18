"""Bitter-lesson self-play: one policy for every decision, trained to win the Bola, on the GPU.

    .venv-jax/bin/python -m jaxenv.selfplay --out runs/run2 --logdir runs/tb

- Policy: one MLP over jaxenv.encode (one seat's raw view) for Dash Calls, bids, With, estimates
  and cards. No hand evaluation, imitation or search; it starts from nothing.
- Reward: only at the end of the Bola, zero-sum, chosen by --objective (``objective_rewards``):
  weighted (King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1, ties shared), king (King or nothing)
  or delta (score lead, scaled and clipped). A game past int32's range is dropped, not scored.
- Credit: TD(lambda) along each seat's own decisions (``td_lambda``), so a move is judged by how
  it changed the seat's expected final position, without waiting for the end of the game.
- Critic, training only: the same view plus the three hidden hands, so deal luck stays out of
  the advantage. It is thrown away afterwards; the policy never sees hidden cards.
- Auxiliary head on the policy: predicts the round's relative score, which teaches the shared
  layers about tricks and contracts. It affects neither the reward nor the moves' scores.
- Opponents: half the games are self-play; in the rest two opposite seats play a past version.
- --benchmark: RL v2's weights. It then plays the other seats in --benchmark-share of the
  iterations, and the log reports the King rate against it.
- --rounds: the kind of game (``kind_of``). A number ends the game after that many rounds, as
  the ladder's micro games do (5); 0 or "whole" is the whole Bola; "mini" is 5 open rounds and
  then the five fixed trumps. Several kinds train one policy on a mix, --shares giving each
  kind's share of the games. The policy is told how many rounds follow the one it is playing
  (``view_of``), so it knows a micro game's fifth round is the last and a whole game's is not.
- --init starts from a trained policy; --warmup then trains only the critic and the
  auxiliary head for that many iterations, so a critic that knows nothing can't spoil it.
- Standings net, a by-product: from the scores after a round, the rounds left and the carry, each
  seat's expected final reward. It reads no cards, so a search can score a round's outcome with
  it. Saved in the same file (``sw0``...).
- --rivals: copies of other people's bots (train.rival). In --rival-share of the iterations one
  of them is the opponent in place of a past version, and the log reports the score against each.
- Evaluation: King rate and mean position reward of greedy seats against a random player and
  the starting policy (25% and 0 are an even share). Real strength is measured by the arena (bots/bitter-net).
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")  # before JAX touches the GPU

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from arena.positions import REWARDS  # noqa: E402

from . import encode as E  # noqa: E402
from . import game as G  # noqa: E402
from . import rules as R  # noqa: E402
from . import rlv2 as V  # noqa: E402

POOL_DEALS = 8  # deals per game per rollout
HIDDEN = 156  # the three other hands, relative to the acting seat


def _orth(scale):
    return jax.nn.initializers.orthogonal(scale)


def _mlp(key, size, hidden, depth):
    sizes = [size] + [hidden] * depth
    keys = jax.random.split(key, depth)
    return [{"w": _orth(jnp.sqrt(2.0))(k, (a, b)), "b": jnp.zeros(b)} for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def _head(key, hidden, out, scale):
    return {"w": _orth(scale)(key, (hidden, out)), "b": jnp.zeros(out)}


STANDINGS_SIZE, STANDINGS_HIDDEN = 5, 64
FULL_ROUNDS = 18  # a whole Bola without replays
VIEW = E.SIZE + 1  # the policy's input: the round as one seat sees it, then the rounds left


def standings_view(scores, left, carry):
    """[..., 4, 5]: for each seat, the other three's leads over it (largest first, in units of
    200 points), the share of the game still to play, and the carry."""
    lead = (scores[..., None, :] - scores[..., :, None]).astype(jnp.float32) / 200.0
    own = jnp.eye(4, dtype=bool)
    top, low = jnp.where(own, -jnp.inf, lead).max(-1), jnp.where(own, jnp.inf, lead).min(-1)
    middle = lead.sum(-1) - top - low  # a seat's lead over itself is 0; no sort: sorting tiny rows is slow on the GPU
    others = jnp.clip(jnp.stack([top, middle, low], -1), -5.0, 5.0)
    tail = jnp.stack([left.astype(jnp.float32), jnp.log2(jnp.maximum(carry, 1).astype(jnp.float32)) / 10.0], -1)
    return jnp.concatenate([others, jnp.broadcast_to(tail[..., None, :], (*others.shape[:-1], 2))], -1)


def apply_standings(net, view):
    return _out(net["value"], _body(net["body"], view))[..., 0]


def init_nets(key, hidden: int, depth: int):
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    return {
        "standings": {"body": _mlp(k6, STANDINGS_SIZE, STANDINGS_HIDDEN, 2), "value": _head(k7, STANDINGS_HIDDEN, 1, 1.0)},
        "policy": {"body": _mlp(k1, VIEW, hidden, depth), "policy": _head(k2, hidden, R.NUM_ACTIONS, 0.01),
                   "aux": _head(k3, hidden, 1, 1.0)},
        "critic": {"body": _mlp(k4, VIEW + HIDDEN, hidden, depth), "value": _head(k5, hidden, 1, 1.0)},
    }  # fmt: skip


def _body(layers, x):
    for layer in layers:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x


def _out(head, x):
    return x @ head["w"] + head["b"]


def apply_policy(net, x):
    h = _body(net["body"], x)
    return _out(net["policy"], h), _out(net["aux"], h)[..., 0]


def apply_critic(net, x, hidden_hands):
    h = _body(net["body"], jnp.concatenate([x, hidden_hands.astype(x.dtype)], -1))
    return _out(net["value"], h)[..., 0]


def position_rewards(scores):
    """[..., 4]: King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1 (arena.positions.REWARDS). Tied
    seats share the positions they occupy; the four always sum to 0."""
    above = (scores[..., None, :] > scores[..., :, None]).sum(-1)
    tied = (scores[..., None, :] == scores[..., :, None]).sum(-1)
    p = jnp.arange(4)[:, None] if scores.ndim == 1 else jnp.arange(4)[None, :, None]
    held = (above[..., None, :] <= p) & (p < (above + tied)[..., None, :])
    return (jnp.asarray(REWARDS)[..., None] * held).sum(-2) / tied


DELTA_SCALE = 200.0  # a score lead this large is worth the whole reward


def objective_rewards(scores, objective="weighted"):
    """[..., 4]: the end-of-Bola reward that training optimizes, summing to 0 across the seats."""
    if objective == "weighted":
        return position_rewards(scores)
    if objective == "king":
        top = scores == scores.max(-1, keepdims=True)
        return top / top.sum(-1, keepdims=True) - 0.25
    if objective == "delta":
        lead = scores - (scores.sum(-1, keepdims=True) - scores) / 3.0
        clipped = jnp.clip(lead / DELTA_SCALE, -1.0, 1.0)
        return clipped - clipped.mean(-1, keepdims=True)
    raise ValueError(f"unknown objective {objective}")


def round_rewards(r):
    """[B, 4]: each seat's round score minus the mean of the others, over 50, clipped (aux only)."""
    res = r.result.astype(jnp.float32)
    return jnp.clip((res - (res.sum(1, keepdims=True) - res) / 3.0) / 50.0, -5.0, 5.0)


def td_lambda(value, seat, finished, overflow, reward, lam):
    """Per-seat TD(lambda), for [T, B] decisions in time order; [T, B, 4] ``reward``.

    Each seat's decisions form their own chain. A finished game gives each seat its reward as
    the last target. A game past int32's range ends the chains unscored, so its decisions train
    only if a later decision of theirs bootstraps them. A seat's last decision in the rollout has
    no successor, so it is not trained on. Returns (advantage, has_target), both [T, B].
    """
    batch = value.shape[1]

    def back(carry, t):
        next_v, gae, has = carry
        next_v = jnp.where(t["finished"][:, None], t["reward"], next_v)
        gae = jnp.where((t["finished"] | t["overflow"])[:, None], 0.0, gae)
        has = jnp.where(t["finished"][:, None], True, jnp.where(t["overflow"][:, None], False, has))
        s = t["seat"][:, None]
        delta = jnp.take_along_axis(next_v, s, 1)[:, 0] - t["value"]
        adv = delta + lam * jnp.take_along_axis(gae, s, 1)[:, 0]
        adv = jnp.where(jnp.take_along_axis(has, s, 1)[:, 0], adv, 0.0)
        valid = jnp.take_along_axis(has, s, 1)[:, 0]
        next_v = jnp.put_along_axis(next_v, s, t["value"][:, None], 1, inplace=False)
        gae = jnp.put_along_axis(gae, s, adv[:, None], 1, inplace=False)
        has = jnp.put_along_axis(has, s, jnp.ones((batch, 1), bool), 1, inplace=False)
        return (next_v, gae, has), (adv, valid)

    init = (jnp.zeros((batch, 4)), jnp.zeros((batch, 4)), jnp.zeros((batch, 4), bool))
    xs = dict(value=value, seat=seat, finished=finished, overflow=overflow, reward=reward)
    _, (adv, valid) = jax.lax.scan(back, init, xs, reverse=True)
    return adv, valid


def _where_games(cond, a, b):
    return jax.tree.map(lambda x, y: jnp.where(cond.reshape(-1, *[1] * (x.ndim - 1)), x, y), a, b)


def _hidden_hands(r, seat):
    return r.hands[(seat + 1) % 4].ravel(), r.hands[(seat + 2) % 4].ravel(), r.hands[(seat + 3) % 4].ravel()


def hidden_of(rounds, seat):
    parts = jax.vmap(_hidden_hands)(rounds, seat)
    return jnp.concatenate(parts, -1)


class Kind(NamedTuple):
    """The kind of game a seat is in. Micro: stop after 5 of a Bola's rounds. Mini: 5 open
    rounds, then the five fixed trumps. Whole: the Bola, 13 open rounds and the fixed trumps."""

    limit: jax.Array  # stop after this many rounds (0: play until the fixed trumps are done)
    open_rounds: jax.Array  # open rounds before the fixed trumps


MINI_OPEN = 5
SPEED_ROUNDS = len(R.FIXED_ORDER)


def kind_of(name) -> tuple[int, int]:
    """(limit, open rounds) of "whole" (or 0), "mini", or a number of rounds to stop after."""
    if str(name) in ("0", "whole"):
        return 0, R.OPEN_ROUNDS
    if str(name) == "mini":
        return 0, MINI_OPEN
    return int(name), R.OPEN_ROUNDS


def game_length(kind: Kind):
    """Rounds in a game of this kind, fixed-trump replays aside."""
    return jnp.where(kind.limit > 0, kind.limit, kind.open_rounds + SPEED_ROUNDS)


def rounds_left(round_no, kind: Kind):
    """Rounds still to play once round ``round_no`` (from 1) is over, as arena.lookahead.rounds_after."""
    return jnp.maximum(game_length(kind) - round_no, 0)


def view_of(g, kind: Kind):
    """[B, VIEW]: what the policy sees. The acting seat's raw view of the round, then how many
    rounds follow it, out of a whole game's 18."""
    x = jax.vmap(E.encode)(g.round, g.round.cur)
    return jnp.concatenate([x, (rounds_left(g.round_no, kind) / FULL_ROUNDS)[:, None]], 1)


def legal_in(x):
    """The legal moves, which close the raw view inside a policy's input."""
    return x[:, E.SIZE - R.NUM_ACTIONS : E.SIZE] > 0.5


def choose(policy, opp, g, learner, key, kind: Kind, greedy=False, benchmark=None):
    """``kind``: each game's kind. ``benchmark``: RL v2's nets (jaxenv.rlv2), which then play
    the other seats instead of ``opp``."""
    seat = g.round.cur
    x = view_of(g, kind)
    mask = jax.vmap(G.game_legal)(g)
    logits, _ = apply_policy(policy, x)
    if benchmark is None:
        opp_logits, _ = apply_policy(opp, x)
    else:
        opp_logits = jax.nn.one_hot(jax.vmap(lambda one: V.act(benchmark, one))(g), R.NUM_ACTIONS) * 1e6
    mine = jnp.take_along_axis(learner, seat[:, None], 1)[:, 0]
    logits = jnp.where(mask, jnp.where(mine[:, None], logits, opp_logits), -1e9)
    action = jnp.argmax(logits, 1) if greedy else jax.random.categorical(key, logits)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
    return action, logp, x, mine, mask.sum(1)


def _advance(g, used, base, pool, action, kind: Kind):
    """Play ``action``. Returns the games (finished ones restarted), the deal counter, the round
    just played, whether it ended, the game before any restart, and whether that game is over:
    its fixed trumps are done, or its kind's limit of rounds was played."""
    r = jax.vmap(R.step)(g.round, action)
    ended = (r.phase == R.DONE) & ~g.over
    before = g.round_no
    after = jax.vmap(G.game_step)(g, action, pool[base + used % POOL_DEALS])
    used = used + (after.round_no != before)
    over = after.over | ((kind.limit > 0) & (after.round_no > kind.limit))
    stop = over | after.overflow
    fresh = jax.vmap(G.new_game)(pool[base + (used + 1) % POOL_DEALS], kind.open_rounds)
    return _where_games(stop, fresh, after), used + stop, r, ended, after, over


def game_kinds(batch: int, rounds=(0,), shares=()) -> Kind:
    """Each of ``batch`` game slots' kind: the kinds named in ``rounds`` (``kind_of``), each on
    its share of the slots (equal shares when none are given). A slot keeps its kind, so a
    kind's share of the slots is also its share of the decisions trained on."""
    shares = np.asarray(shares or [1.0] * len(rounds), np.float64)
    if len(shares) != len(rounds) or shares.min() <= 0:
        raise ValueError("--shares needs one positive share per --rounds")
    upto = np.round(np.cumsum(shares / shares.sum()) * batch).astype(int)
    which = np.searchsorted(upto, np.arange(batch), side="right").clip(max=len(rounds) - 1)
    limit, open_rounds = np.array([kind_of(name) for name in rounds])[which].T
    return Kind(jnp.asarray(limit, jnp.int32), jnp.asarray(open_rounds, jnp.int32))


def make_rollout(batch: int, steps: int, lam: float, objective: str = "weighted", benchmark=None, rounds=0):
    """``rounds``: every game's kind by name (``kind_of``), or each game's own (``game_kinds``)."""
    base = jnp.arange(batch) * POOL_DEALS
    kind = rounds if isinstance(rounds, Kind) else game_kinds(batch, (rounds,))
    length = game_length(kind)

    def rollout(nets, opp, g, key):
        k_pool, k_seats, k_mix, k_steps = jax.random.split(key, 4)
        pool = G.deal_pool(k_pool, batch * POOL_DEALS)
        pairs = jax.random.bernoulli(k_seats, 0.5, (batch,))
        two = jnp.stack([pairs, ~pairs, pairs, ~pairs], 1)
        learner = jax.random.bernoulli(k_mix, 0.5, (batch,))[:, None] | two  # half pure self-play

        def one(carry, k):
            g, used = carry
            seat = g.round.cur
            phase = g.round.phase
            pre_super = g.round.caller < 0
            action, logp, x, mine, n_legal = choose(nets["policy"], opp, g, learner, k, kind, benchmark=benchmark)
            hidden = hidden_of(g.round, seat)
            value = apply_critic(nets["critic"], x, hidden)
            g, used, r, ended, after, over = _advance(g, used, base, pool, action, kind)
            played = ended & ~r.skipped
            left = jnp.clip(length + 1 - after.round_no, 0, length) / length  # share of the game still to play
            made = jnp.take_along_axis(r.est == r.won, jnp.maximum(r.caller, 0)[:, None], 1)[:, 0]
            return (g, used), dict(
                x=x.astype(jnp.float16), hidden=hidden, action=action, logp=logp, value=value, seat=seat,
                train=mine & (n_legal > 1), ended=ended, round_rew=round_rewards(r),
                finished=over & ~after.overflow, overflow=after.overflow,
                standings=standings_view(after.scores, jnp.where(over, 0.0, left), after.carry),
                reward=objective_rewards(after.scores, objective), phase=phase, pre_super=pre_super,
                skipped=ended & r.skipped, played=played, made=played & made,
                all_lose=played & ~(r.est == r.won).any(1), game_rounds=after.round_no,
            )  # fmt: skip

        (g, _), tr = jax.lax.scan(one, (g, jnp.zeros(batch, jnp.int32)), jax.random.split(k_steps, steps))
        adv, has = td_lambda(tr["value"], tr["seat"], tr["finished"], tr["overflow"], tr["reward"], lam)

        def back(carry, t):  # the round's relative score (aux target) and the game's result
            ret, has_round, final, has_final = carry
            ret = jnp.where(t["ended"][:, None], t["round_rew"], ret)
            has_round = has_round | t["ended"]
            final = jnp.where(t["finished"][:, None], t["reward"], final)
            has_final = (has_final | t["finished"]) & ~t["overflow"]
            s = t["seat"][:, None]
            return (ret, has_round, final, has_final), (
                jnp.take_along_axis(ret, s, 1)[:, 0], has_round,
                jnp.take_along_axis(final, s, 1)[:, 0], has_final, final,
            )  # fmt: skip

        zeros = (jnp.zeros((batch, 4)), jnp.zeros(batch, bool), jnp.zeros((batch, 4)), jnp.zeros(batch, bool))
        keys = ("ended", "round_rew", "finished", "reward", "overflow", "seat")
        _, (aux_target, aux_ok, final, has_final, final_all) = jax.lax.scan(back, zeros, {k: tr[k] for k in keys}, reverse=True)

        a, ph = tr["action"], tr["phase"]
        rate = lambda hit, of: hit.sum() / jnp.maximum(of.sum(), 1)  # noqa: E731
        is_bid = (a >= R.FIRST_BID) & (a <= R.LAST_BID)
        n_fin = tr["finished"].sum()
        target = adv + tr["value"]
        valid = has & tr["train"]
        vw = valid.astype(jnp.float32)
        var = lambda z: ((z - (z * vw).sum() / vw.sum()) ** 2 * vw).sum() / vw.sum()  # noqa: E731
        stats = dict(
            dash=rate((ph == R.DASH) & (a == R.DASH_CALL), ph == R.DASH),
            bid=rate((ph == R.AUCTION) & is_bid, ph == R.AUCTION),
            with_=rate((ph == R.AUCTION) & (a == R.WITH), ph == R.AUCTION),
            pass_=rate((ph == R.AUCTION) & (a == R.PASS), ph == R.AUCTION),
            super_=rate((ph == R.ESTIMATION) & is_bid, (ph == R.ESTIMATION) & tr["pre_super"]),
            skipped=rate(tr["skipped"], tr["ended"]),
            all_lose=rate(tr["all_lose"], tr["played"]),
            caller_made=rate(tr["made"], tr["played"]),
            games_done=n_fin,
            overflow=rate(tr["overflow"], tr["overflow"] | tr["finished"]),
            rounds_per_game=(tr["game_rounds"] * tr["finished"]).sum() / jnp.maximum(n_fin, 1),
            explained_var=1.0 - var(target - tr["value"]) / var(target),
            # how well the critic predicts the final position in a game's last rollout; 0.52 = no idea
            final_mse=((tr["value"] - final) ** 2 * has_final).sum() / jnp.maximum(has_final.sum(), 1),
        )
        data = dict(x=tr["x"], hidden=tr["hidden"], action=a, logp=tr["logp"], value=tr["value"], target=target,
                    valid=valid, aux_target=aux_target, aux_ok=aux_ok & tr["train"],
                    standings=tr["standings"], standings_target=final_all, standings_ok=tr["ended"] & has_final)  # fmt: skip
        flat = jax.tree.map(lambda v: v.reshape(batch * steps, *v.shape[2:]), data)
        adv = adv.reshape(batch * steps)
        w = flat["valid"]
        n = jnp.maximum(w.sum(), 1)
        mean = (adv * w).sum() / n
        flat["adv"] = (adv - mean) / (jnp.sqrt((((adv - mean) ** 2) * w).sum() / n) + 1e-6)
        return g, flat, stats

    return jax.jit(rollout)


def make_update(tx_policy, tx_critic, clip: float, aux_weight: float, tx_standings=None):
    def policy_loss(net, mb, ent):
        x = mb["x"].astype(jnp.float32)
        logits, aux = apply_policy(net, x)
        legal = legal_in(x)
        logp_all = jax.nn.log_softmax(jnp.where(legal, logits, -1e9))
        logp = jnp.take_along_axis(logp_all, mb["action"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - mb["logp"])
        w = mb["valid"].astype(jnp.float32)
        n = jnp.maximum(w.sum(), 1.0)
        pg = (w * -jnp.minimum(ratio * mb["adv"], jnp.clip(ratio, 1 - clip, 1 + clip) * mb["adv"])).sum() / n
        entropy = (w * -(jnp.exp(logp_all) * jnp.where(legal, logp_all, 0.0)).sum(1)).sum() / n
        aw = mb["aux_ok"].astype(jnp.float32)
        aux_loss = (aw * (aux - mb["aux_target"]) ** 2).sum() / jnp.maximum(aw.sum(), 1.0)
        kl = (w * (mb["logp"] - logp)).sum() / n
        return pg - ent * entropy + aux_weight * aux_loss, dict(pg=pg, entropy=entropy, kl=kl, aux_loss=aux_loss)

    def critic_loss(net, mb):
        value = apply_critic(net, mb["x"].astype(jnp.float32), mb["hidden"])
        w = mb["valid"].astype(jnp.float32)
        return (w * (value - mb["target"]) ** 2).sum() / jnp.maximum(w.sum(), 1.0)

    def standings_loss(net, mb):
        w = mb["standings_ok"].astype(jnp.float32)[:, None]
        error = (apply_standings(net, mb["standings"]) - mb["standings_target"]) ** 2
        return (w * error).sum() / jnp.maximum(4.0 * w.sum(), 1.0)

    @jax.jit
    def step(nets, opts, mb, ent, gate=1.0):
        """``gate`` 0 holds the policy's body and move head still (the warm-up after --init)."""
        (_, info), g_p = jax.value_and_grad(policy_loss, has_aux=True)(nets["policy"], mb, ent)
        v_loss, g_c = jax.value_and_grad(critic_loss)(nets["critic"], mb)
        up_p, o_p = tx_policy.update(g_p, opts[0], nets["policy"])
        up_p = {**up_p, "body": jax.tree.map(lambda u: u * gate, up_p["body"]), "policy": jax.tree.map(lambda u: u * gate, up_p["policy"])}
        up_c, o_c = tx_critic.update(g_c, opts[1], nets["critic"])
        out = {"policy": optax.apply_updates(nets["policy"], up_p), "critic": optax.apply_updates(nets["critic"], up_c)}
        out["standings"], s_loss, o_s = nets["standings"], 0.0, opts[2]
        if tx_standings is not None:
            s_loss, g_s = jax.value_and_grad(standings_loss)(nets["standings"], mb)
            up_s, o_s = tx_standings.update(g_s, opts[2], nets["standings"])
            out["standings"] = optax.apply_updates(nets["standings"], up_s)
        return out, (o_p, o_c, o_s), info | {"value_loss": v_loss, "standings_loss": s_loss}

    return step


def make_eval(batch: int, steps: int, benchmark=None, rounds=0):
    """Greedy ``policy`` in two opposite seats vs ``opp`` in games of the kind ``rounds`` names:
    (King rate, mean reward, games)."""
    base = jnp.arange(batch) * POOL_DEALS
    kind = game_kinds(batch, (rounds,))

    @jax.jit
    def run(policy, opp, key):
        k_pool, k_steps = jax.random.split(key)
        pool = G.deal_pool(k_pool, batch * POOL_DEALS)
        g = jax.vmap(G.new_game)(pool[base], kind.open_rounds)
        learner = jnp.array([True, False, True, False])[None, :] ^ (jnp.arange(batch) % 2 == 0)[:, None]

        def one(carry, k):
            g, used, firsts, total, games = carry
            action, *_ = choose(policy, opp, g, learner, k, kind, greedy=True, benchmark=benchmark)
            g, used, _, _, after, over = _advance(g, used, base, pool, action, kind)
            done = over & ~after.overflow
            top = after.scores == after.scores.max(1, keepdims=True)
            share = (top * learner).sum(1) / top.sum(1)
            mean_r = (position_rewards(after.scores) * learner).sum(1) / 2.0
            return (g, used, firsts + (share * done).sum(), total + (mean_r * done).sum(), games + done.sum()), None

        init = (g, jnp.zeros(batch, jnp.int32), 0.0, 0.0, 0)
        (_, _, firsts, total, games), _ = jax.lax.scan(one, init, jax.random.split(k_steps, steps))
        n = jnp.maximum(games, 1)
        return firsts / n / 2.0, total / n, games

    return run


def load_policy(path: Path, like):
    """A saved policy's body and move head in ``like``'s shape; the auxiliary head stays as it is.
    A policy saved before the view told the rounds left gets that input with zero weights, so
    it plays exactly as it did."""
    data = np.load(path)
    body = [{"w": jnp.asarray(data[f"w{k}"]), "b": jnp.asarray(data[f"b{k}"])} for k in range(len(like["body"]))]
    if body[0]["w"].shape[0] == E.SIZE:
        body[0]["w"] = jnp.concatenate([body[0]["w"], jnp.zeros((VIEW - E.SIZE, body[0]["w"].shape[1]))])
    if [layer["w"].shape for layer in body] != [layer["w"].shape for layer in like["body"]]:
        raise ValueError(f"{path} does not fit --hidden and --depth")
    return {**like, "body": body, "policy": {"w": jnp.asarray(data["pw"]), "b": jnp.asarray(data["pb"])}}


def save_npz(policy, path: Path, meta: dict, standings=None) -> None:
    arrays = {}
    if standings is not None:
        for k, layer in enumerate([*standings["body"], standings["value"]]):
            arrays[f"sw{k}"], arrays[f"sb{k}"] = np.asarray(layer["w"]), np.asarray(layer["b"])
    for k, layer in enumerate(policy["body"]):
        arrays[f"w{k}"], arrays[f"b{k}"] = np.asarray(layer["w"]), np.asarray(layer["b"])
    arrays["pw"], arrays["pb"] = np.asarray(policy["policy"]["w"]), np.asarray(policy["policy"]["b"])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part.npz")  # readers never see a half-written net
    np.savez_compressed(partial, **{k: v.astype(np.float32) for k, v in arrays.items()}, meta=json.dumps(meta))
    partial.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m jaxenv.selfplay", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--batch", type=int, default=4096, help="games at once")
    p.add_argument("--steps", type=int, default=128, help="decisions per game per iteration")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--objective", choices=("weighted", "king", "delta"), default="weighted", help="what the reward pays")
    p.add_argument("--lam", type=float, default=0.95, help="TD(lambda) per own decision")
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--entropy-end", type=float, default=0.01)
    p.add_argument("--aux", type=float, default=0.25, help="weight of the round-score head")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=32768)
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--pool", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path)
    p.add_argument("--rounds", nargs="+", default=["whole"],
                   help="the kind of game: rounds to stop after (the ladder plays 5), whole, or mini; several: a mix")
    p.add_argument("--shares", type=float, nargs="*", default=[], help="each kind's share of the games (default: equal)")
    p.add_argument("--init", type=Path, help="a trained policy (net.npz) to start from")
    p.add_argument("--warmup", type=int, default=0, help="iterations that leave the starting policy's moves alone")
    p.add_argument("--rivals", type=Path, nargs="*", default=[], help="copies of other bots (train.rival) to play against")
    p.add_argument("--rival-share", type=float, default=0.3, help="share of iterations played against a rival copy")
    p.add_argument("--benchmark", type=Path, help="RL v2 weights: train against it and report the score vs it")
    p.add_argument("--benchmark-share", type=float, default=0.5, help="share of iterations played against it")
    a = p.parse_args(argv)
    a.out = a.out.expanduser()
    a.out.mkdir(parents=True, exist_ok=True)
    print("device:", jax.devices()[0], "| view:", VIEW, "| critic adds:", HIDDEN, flush=True)
    key = jax.random.key(a.seed)
    key, k_net = jax.random.split(key)
    nets = init_nets(k_net, a.hidden, a.depth)
    if a.init:
        nets["policy"] = load_policy(a.init.expanduser(), nets["policy"])
    uniform = jax.tree.map(jnp.zeros_like, nets["policy"])  # all-zero logits: uniform over legal
    tx_policy = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    tx_standings = optax.adam(1e-3)
    opts = (tx_policy.init(nets["policy"]), tx_critic.init(nets["critic"]), tx_standings.init(nets["standings"]))
    first_policy = jax.device_get(nets["policy"])
    rivals = [jax.device_get(load_policy(path.expanduser(), nets["policy"])) for path in a.rivals]
    pool = [first_policy]
    kind = game_kinds(a.batch, a.rounds, a.shares)
    rollout = make_rollout(a.batch, a.steps, a.lam, a.objective, rounds=kind)
    update = make_update(tx_policy, tx_critic, a.clip, a.aux, tx_standings)
    evaluate = make_eval(512, 4000, rounds=a.rounds[0])  # the first kind is the one the log is read by
    evaluate_other = {n: make_eval(512, 4000, rounds=n) for n in a.rounds[1:]}
    benchmark = V.load(a.benchmark.expanduser()) if a.benchmark else None
    rollout_benchmark = make_rollout(a.batch, a.steps, a.lam, a.objective, benchmark, kind) if benchmark else None
    evaluate_benchmark = make_eval(512, 4000, benchmark, a.rounds[0]) if benchmark else None
    key, k = jax.random.split(key)
    hands = G.deal_pool(k, a.batch)
    g = jax.vmap(G.new_game)(hands, kind.open_rounds)
    if any(kind_of(name) != kind_of("whole") for name in a.rounds):
        # Short games all started together would all end together, and whole iterations would
        # see no result. So the first games start at a random open round and end at different times.
        first = jax.random.randint(jax.random.fold_in(k, 1), (a.batch,), 1, jnp.minimum(game_length(kind), kind.open_rounds) + 1)
        g = g._replace(round_no=first, round=jax.vmap(R.init_round)(first, hands))
    writer = None
    if a.logdir:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(str(a.logdir.expanduser() / f"bitter-{a.out.name}"))
    log = (a.out / "log.jsonl").open("a")
    print(f"policy {sum(x.size for x in jax.tree.leaves(nets['policy'])):,} parameters, "
          f"{a.batch * a.steps:,} decisions per iteration", flush=True)  # fmt: skip
    t0, decisions = time.perf_counter(), 0
    for it in range(1, a.iterations + 1):
        key, k_opp, k_roll, k_perm, k_eval = jax.random.split(key, 5)
        opp = pool[int(jax.random.randint(k_opp, (), 0, len(pool)))]
        if rivals and float(jax.random.uniform(jax.random.fold_in(k_opp, 2))) < a.rival_share:
            opp = rivals[int(jax.random.randint(jax.random.fold_in(k_opp, 3), (), 0, len(rivals)))]
        versus = benchmark and float(jax.random.uniform(jax.random.fold_in(k_opp, 1))) < a.benchmark_share
        g, flat, stats = (rollout_benchmark if versus else rollout)(nets, opp, g, k_roll)
        n = a.batch * a.steps
        frac = (it - 1) / max(a.iterations - 1, 1)
        ent = a.entropy + (a.entropy_end - a.entropy) * frac
        for epoch in range(a.epochs):
            order = jnp.argsort(jax.random.uniform(jax.random.fold_in(k_perm, epoch), (n,)))
            for s in range(0, n - a.minibatch + 1, a.minibatch):
                mb = jax.tree.map(lambda x: x[order[s : s + a.minibatch]], flat)
                nets, opts, info = update(nets, opts, mb, ent, 0.0 if it <= a.warmup else 1.0)
        decisions += n
        if it % a.snapshot_every == 0:
            pool = (pool + [jax.device_get(nets["policy"])])[-a.pool :]
        row = {"iteration": it, "valid": float(flat["valid"].mean()), **{k: float(v) for k, v in info.items()},
               **{k: float(v) for k, v in stats.items()}, "decisions": decisions,
               "per_s": decisions / (time.perf_counter() - t0)}  # fmt: skip
        if it % a.eval_every == 0 or it == 1:
            king_r, pos_r, games = evaluate(nets["policy"], uniform, k_eval)
            king_s, pos_s, _ = evaluate(nets["policy"], first_policy, jax.random.fold_in(k_eval, 1))
            row |= {"king_vs_random": float(king_r), "pos_vs_random": float(pos_r), "king_vs_start": float(king_s),
                    "pos_vs_start": float(pos_s), "eval_games": int(games)}
            for number, (n, evaluate_n) in enumerate(evaluate_other.items()):  # the same comparison in the other kinds
                king_n, pos_n, _ = evaluate_n(nets["policy"], first_policy, jax.random.fold_in(k_eval, 100 + number))
                row |= {f"king_vs_start_r{n}": float(king_n), f"pos_vs_start_r{n}": float(pos_n)}
            for number, rival in enumerate(rivals):
                _, pos_v, _ = evaluate(nets["policy"], rival, jax.random.fold_in(k_eval, 10 + number))
                row[f"pos_vs_rival_{number}"] = float(pos_v)
            if benchmark:
                king_b, pos_b, _ = evaluate_benchmark(nets["policy"], uniform, jax.random.fold_in(k_eval, 2))
                row |= {"king_vs_benchmark": float(king_b), "pos_vs_benchmark": float(pos_b)}
                print(f"iter {it:>5}: King/position vs RL v2 {row['king_vs_benchmark']:5.1%} {row['pos_vs_benchmark']:+.2f}", flush=True)
            print(f"iter {it:>5}: King/position vs random {row['king_vs_random']:5.1%} {row['pos_vs_random']:+.2f} "
                  f"vs start {row['king_vs_start']:5.1%} {row['pos_vs_start']:+.2f} ({row['eval_games']} games) | "
                  f"entropy {row['entropy']:.2f} final_mse {row['final_mse']:.3f} ev {row['explained_var']:+.2f} kl {row['kl']:.4f} | "
                  f"dash {row['dash']:.2f} bid {row['bid']:.2f} with {row['with_']:.2f} pass {row['pass_']:.2f} super {row['super_']:.2f} | "
                  f"skip {row['skipped']:.2f} alllose {row['all_lose']:.2f} made {row['caller_made']:.2f} "
                  f"rounds/game {row['rounds_per_game']:.1f} ovf {row['overflow']:.2f} | {row['per_s']:,.0f}/s"
                  + "".join(f" | {n if kind_of(n)[0] == 0 else f'{n}-round'} games vs start {row[f'king_vs_start_r{n}']:5.1%} {row[f'pos_vs_start_r{n}']:+.2f}" for n in evaluate_other), flush=True)
            save_npz(nets["policy"], a.out / "net.npz", {"iteration": it, "args": {k: str(v) for k, v in vars(a).items()}}, nets["standings"])
        if it % 200 == 0:
            save_npz(nets["policy"], a.out / f"net-{it:05d}.npz", {"iteration": it}, nets["standings"])
        log.write(json.dumps(row) + "\n")
        log.flush()
        if writer:
            for k, v in row.items():
                writer.add_scalar(k, v, it)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
