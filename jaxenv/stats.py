"""Game statistics under uniform random play, from the engine (CPU) and from jaxenv (GPU).

    .venv-jax/bin/python -m jaxenv.stats --cpu-games 6000 --gpu-games 200000 --out stats.json

The move-by-move check (jaxenv.check) proves the two play identically on the same deals. This
compares them as distributions over independent deals: per-round rule frequencies, game
length and scores, each a mean with a 95% interval, plus throughput. If the rules agree,
every difference should sit inside the noise.
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

ROUND_STATS = ("skipped", "four_dash", "dash_calls", "super_call", "trump_changed", "all_lose",
               "doubled", "with_players", "caller_won", "risks")  # fmt: skip
GAME_STATS = ("rounds", "decisions", "fixed_replays", "top_score", "spread")


def _cpu_worker(args):
    seed, n = args
    import estimation.bots  # noqa: F401
    from estimation.engine import Bola

    rng = random.Random(seed)
    rounds = {k: [] for k in ROUND_STATS}
    games = {k: [] for k in GAME_STATS}
    t = time.perf_counter()
    games["dropped"] = []
    for _ in range(n):
        b = Bola(rng.randrange(2**31))
        decisions = 0
        supers = []
        while not b.is_over:
            rnd, ended = b.round, len(b.history)
            b.step(rng.choice(b.legal_actions()))
            decisions += 1
            if len(b.history) != ended:
                supers.append(rnd._super)
        running, out_of_range = [0, 0, 0, 0], False  # the same test as Game.overflow
        for r in b.history:
            running = [x + y for x, y in zip(running, r.scores)]
            out_of_range |= r.next_carry > 1 << 20 or max(map(abs, running)) > 1 << 30
        games["dropped"].append(out_of_range)
        if out_of_range:
            continue  # the GPU flags and drops these games, so drop the same population here
        rounds["super_call"] += supers
        for r in b.history:
            rounds["skipped"].append(r.skipped)
            rounds["four_dash"].append(sum(s.is_dash for s in r.seats) == 4)
            rounds["dash_calls"].append(sum(s.is_dash for s in r.seats))
            rounds["trump_changed"].append(r.trump_changed)
            rounds["all_lose"].append(not r.skipped and r.all_lose)
            rounds["doubled"].append(not r.skipped and r.calls >= 3)
            rounds["with_players"].append(0 if r.skipped else r.calls - 1)
            rounds["caller_won"].append(not r.skipped and r.seats[r.caller].won)
            rounds["risks"].append(0 if r.skipped else max(s.risks for s in r.seats))
        games["rounds"].append(len(b.history))
        games["decisions"].append(decisions)
        games["fixed_replays"].append(b.replays)
        games["top_score"].append(max(b.scores))
        games["spread"].append(max(b.scores) - min(b.scores))
    return rounds, games, decisions_total(games), time.perf_counter() - t


def decisions_total(games) -> int:
    return int(sum(games["decisions"]))


def cpu_stats(n_games: int, workers: int, seed: int):
    chunks = [(seed + k, n_games // workers) for k in range(workers)]
    t = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_cpu_worker, chunks)
    wall = time.perf_counter() - t
    rounds = {k: sum((p[0][k] for p in parts), []) for k in ROUND_STATS}
    games = {k: sum((p[1][k] for p in parts), []) for k in (*GAME_STATS, "dropped")}
    core_rate = sum(p[2] for p in parts) / sum(p[3] for p in parts)
    return rounds, games, {"wall_s": wall, "decisions_per_s": sum(p[2] for p in parts) / wall,
                           "per_core": core_rate, "workers": workers}  # fmt: skip


def gpu_stats(n_games: int, batch: int, seed: int):
    import jax
    import jax.numpy as jnp

    from . import game as G
    from . import rules as R

    POOL = 40  # deals per game slot per call: a game has at most 13 + 5 + 20 rounds
    steps = 4000  # every game ends within this; asserted below, so no bias against long games

    @jax.jit
    def play(key):
        k_pool, k_act = jax.random.split(key)
        pool = G.deal_pool(k_pool, batch * POOL)
        base = jnp.arange(batch) * POOL
        g = jax.vmap(G.new_game)(pool[base])
        zero = jnp.zeros(batch, jnp.int32)
        acc = {k: zero for k in ("rounds", "skipped", "four_dash", "dash_calls", "super_call", "trump_changed",
                                 "all_lose", "doubled", "with_players", "caller_won", "risks", "decisions",
                                 "dash_calls_sq", "with_players_sq", "risks_sq")}  # fmt: skip

        def one(carry, k):
            g, used, acc = carry
            mask = jax.vmap(G.game_legal)(g)
            action = jax.random.categorical(k, jnp.where(mask, 0.0, -jnp.inf))
            live = ~g.over
            r = jax.vmap(R.step)(g.round, action)
            ended = (r.phase == R.DONE) & live
            won_c = jnp.take_along_axis(r.est == r.won, jnp.maximum(r.caller, 0)[:, None], 1)[:, 0]
            played = ~r.skipped
            n_with = (r.with_w >= 0).sum(1)
            all_lose = played & ~(r.est == r.won).any(1)
            risks = jnp.where(played & (r.last >= 0), jnp.abs(r.est.sum(1) - 13) // 2, 0)
            add = {"rounds": ended, "skipped": ended & r.skipped, "four_dash": ended & (r.dash.sum(1) == 4),
                   "dash_calls": jnp.where(ended, r.dash.sum(1), 0), "super_call": ended & r.super_call,
                   "trump_changed": ended & r.trump_changed, "all_lose": ended & all_lose,
                   "doubled": ended & played & (n_with >= 2), "with_players": jnp.where(ended & played, n_with, 0),
                   "caller_won": ended & played & won_c, "risks": jnp.where(ended, risks, 0), "decisions": live}  # fmt: skip
            for c in ("dash_calls", "with_players", "risks"):
                add[c + "_sq"] = add[c] * add[c]
            acc = {k: acc[k] + add[k].astype(jnp.int32) for k in acc}
            before = g.round_no
            g = jax.vmap(G.game_step)(g, action, pool[base + used % POOL])
            used = used + (g.round_no != before)
            return (g, used, acc), None

        (g, _, acc), _ = jax.lax.scan(one, (g, zero, acc), jax.random.split(k_act, steps))
        return g, acc

    sums = {k: [] for k in ("rounds",) + ROUND_STATS + ("decisions", "dash_calls_sq", "with_players_sq", "risks_sq")}
    games = {k: [] for k in GAME_STATS}
    games["dropped"] = []
    g, acc = play(jax.random.key(seed))  # compile
    jax.block_until_ready(acc)
    t, done, decisions = time.perf_counter(), 0, 0
    it = 1
    while done < n_games:
        g, acc = play(jax.random.key(seed + it))
        it += 1
        acc = {k: np.asarray(v) for k, v in acc.items()}
        over, overflow = np.asarray(g.over), np.asarray(g.overflow)
        assert over.all(), "a game outlasted the step budget"
        keep = ~overflow
        games["dropped"] += overflow.tolist()
        decisions += int(acc["decisions"].sum())
        for k in sums:
            sums[k].append(acc[k][keep])
        scores = np.asarray(g.scores)[keep]
        games["rounds"] += acc["rounds"][keep].tolist()
        games["decisions"] += acc["decisions"][keep].tolist()
        games["fixed_replays"] += np.asarray(g.replays)[keep].tolist()
        games["top_score"] += scores.max(1).tolist()
        games["spread"] += (scores.max(1) - scores.min(1)).tolist()
        done += int(keep.sum())
    wall = time.perf_counter() - t
    tot = {k: np.concatenate(v).sum() for k, v in sums.items()}
    return tot, games, {"wall_s": wall, "decisions_per_s": decisions / wall, "batch": batch}


def mean_ci(x):
    x = np.asarray(x, float)
    m, sd = x.mean(), x.std(ddof=1)
    h = 1.96 * sd / math.sqrt(len(x))
    return {"mean": m, "low": m - h, "high": m + h, "n": len(x)}


def rate_ci(k, n):
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return {"mean": p, "low": p - h, "high": p + h, "n": int(n)}


LABELS = {
    "skipped": "rounds skipped (nobody calls)", "four_dash": "four Dash Calls",
    "dash_calls": "Dash Calls per round", "super_call": "rounds with a super call",
    "trump_changed": "fixed trump changed (replayed)", "all_lose": "everyone lost the round",
    "doubled": "doubled by 3+ calls", "with_players": "With players per round",
    "caller_won": "Caller made the call", "risks": "Risks per round",
    "rounds": "rounds per game", "decisions": "decisions per game", "fixed_replays": "fixed-trump replays per game",
    "top_score": "winner's score", "spread": "score spread, top minus bottom",
}  # fmt: skip
COUNTS = ("dash_calls", "with_players", "risks")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m jaxenv.stats", description=__doc__.split("\n")[0])
    p.add_argument("--cpu-games", type=int, default=6000)
    p.add_argument("--gpu-games", type=int, default=200_000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="jaxenv-stats.json")
    a = p.parse_args(argv)
    c_rounds, c_games, c_speed = cpu_stats(a.cpu_games, a.workers, a.seed)
    print(f"cpu: {len(c_games['rounds'])} games in {c_speed['wall_s']:.0f} s", flush=True)
    g_tot, g_games, g_speed = gpu_stats(a.gpu_games, a.batch, a.seed + 1000)
    print(f"gpu: {len(g_games['rounds'])} games in {g_speed['wall_s']:.0f} s", flush=True)
    n_c, n_g = len(c_rounds["skipped"]), int(g_tot["rounds"])
    rows = []
    for k in ROUND_STATS:
        if k in COUNTS:
            cpu = mean_ci(c_rounds[k])
            m = g_tot[k] / n_g
            h = 1.96 * math.sqrt(max(g_tot[k + "_sq"] / n_g - m * m, 0.0) / n_g)
            gpu = {"mean": m, "low": m - h, "high": m + h, "n": n_g}
        else:
            cpu, gpu = rate_ci(sum(c_rounds[k]), n_c), rate_ci(g_tot[k], n_g)
        rows.append({"stat": k, "label": LABELS[k], "kind": "count" if k in COUNTS else "rate", "cpu": cpu, "gpu": gpu})
    for k in GAME_STATS:
        rows.append({"stat": k, "label": LABELS[k], "kind": "game", "cpu": mean_ci(c_games[k]), "gpu": mean_ci(g_games[k])})
    rows.append({"stat": "dropped", "label": "games past int32 range (flagged)", "kind": "game-rate",
                 "cpu": rate_ci(sum(c_games["dropped"]), len(c_games["dropped"])),
                 "gpu": rate_ci(sum(g_games["dropped"]), len(g_games["dropped"]))})
    for r in rows:
        c, g = r["cpu"], r["gpu"]
        se = math.hypot((c["high"] - c["low"]) / 3.92, (g["high"] - g["low"]) / 3.92)
        r["z"] = 0.0 if se == 0 else (g["mean"] - c["mean"]) / se
        print(f"  {r['label']:<34} cpu {c['mean']:>10.4f}  gpu {g['mean']:>10.4f}  z {r['z']:+.2f}")
    out = {"rows": rows, "cpu_speed": c_speed, "gpu_speed": g_speed,
           "cpu_games": len(c_games["rounds"]), "gpu_games": len(g_games["rounds"]),
           "cpu_rounds": n_c, "gpu_rounds": n_g,
           "hist": {"cpu_top": np.asarray(c_games["top_score"]).tolist(), "gpu_top": np.asarray(g_games["top_score"])[:200_000].tolist(),
                    "cpu_rounds": c_games["rounds"], "gpu_rounds": g_games["rounds"][:200_000]}}  # fmt: skip
    print("speed: cpu", f"{c_speed['decisions_per_s']:,.0f}/s on {a.workers} cores;", "gpu", f"{g_speed['decisions_per_s']:,.0f}/s")
    with open(a.out, "w") as f:
        json.dump(out, f, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
