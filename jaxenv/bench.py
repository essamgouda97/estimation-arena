"""Throughput of the JAX rules: random legal play in a batch of games, games restarting.

    .venv-jax/bin/python -m jaxenv.bench --batch 1024 4096 16384

Reports decisions (actions applied) per second, compiled with one ``lax.scan`` per call, so
nothing returns to Python between steps. Compare: the Python engine plays about
``--engine-seconds`` worth of random games on one CPU core.
"""

import argparse
import random
import time

import jax
import jax.numpy as jnp

from . import game as G


POOL = 8  # deals per game per call: a round is ~65 decisions, so 200 steps need ~4


def make_run(steps: int):
    def run(g, key):
        batch = g.scores.shape[0]
        k_pool, k_act = jax.random.split(key)
        pool = G.deal_pool(k_pool, batch * POOL)  # one shuffle per call, not per step
        base = jnp.arange(batch) * POOL

        def one(carry, k):
            g, used = carry
            mask = jax.vmap(G.game_legal)(g)
            action = jax.random.categorical(k, jnp.where(mask, 0.0, -jnp.inf))
            hands = pool[base + used % POOL]
            before = g.round_no
            g = jax.vmap(G.game_step)(g, action, hands)
            used = used + (g.round_no != before)
            fresh = jax.vmap(G.new_game)(pool[base + (used + 1) % POOL])
            g = jax.tree.map(lambda a, b: jnp.where(g.over.reshape(-1, *[1] * (a.ndim - 1)), a, b), fresh, g)
            return (g, used + g.over), mask.any(1).sum()

        (g, _), live = jax.lax.scan(one, (g, jnp.zeros(batch, jnp.int32)), jax.random.split(k_act, steps))
        return g, live.sum()

    return jax.jit(run)


def engine_rate(seconds: float) -> float:
    import estimation.bots  # noqa: F401
    from estimation.engine import Bola

    rng = random.Random(0)
    steps, t = 0, time.perf_counter()
    while time.perf_counter() - t < seconds:
        b = Bola(rng.randrange(2**31))
        while not b.is_over:
            b.step(rng.choice(b.legal_actions()))
            steps += 1
    return steps / (time.perf_counter() - t)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m jaxenv.bench", description=__doc__.split("\n")[0])
    p.add_argument("--batch", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--engine-seconds", type=float, default=10.0)
    a = p.parse_args(argv)
    print("device:", jax.devices()[0])
    base = engine_rate(a.engine_seconds)
    print(f"engine (Python, 1 core): {base:,.0f} decisions/s")
    for batch in a.batch:
        run = make_run(a.steps)
        g = jax.vmap(G.new_game)(G.deal_pool(jax.random.key(0), batch))
        g, n = run(g, jax.random.key(1))  # compile
        n.block_until_ready()
        t = time.perf_counter()
        g, n = run(g, jax.random.key(2))
        n = int(n.block_until_ready())
        dt = time.perf_counter() - t
        print(f"JAX batch {batch:>6}: {n / dt:>12,.0f} decisions/s  ({n / dt / base:,.0f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
