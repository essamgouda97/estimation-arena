"""Our own guess net: who holds each unseen card, learned from Self-taught's games alone.

    .venv-jax/bin/python -m train.guess_net train --out DIR --iterations 300      # JAX environment
    python -m train.guess_net positions --out DIR --games 200                     # engine environment
    python -m train.guess_net score --out DIR                                     # engine environment

train plays Self-taught's net (bots/bitter-search/net.npz) against itself on the GPU and fits a
separate MLP on every decision as it comes: input the acting seat's raw view (jaxenv.encode),
target the true holder, among the three other seats, of each card the seat cannot see. Nothing
is written but the net. Each batch is scored before it is trained on, so the logged loss is
always on positions the net has not seen.

positions plays engine games and keeps card decisions: our raw view, the engine's RL features
(what Yosif's GuessNet reads), the true holders, hand sizes and voids. score compares, on those
positions, our net, Yosif's GuessNet and a baseline that knows only hand sizes and voids.

Seats are counted from the acting seat, the next one first, as HybridBot.sample_deals counts
them; card c is bit c, as everywhere else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEATS = 3
CARDS = 52
DEFAULT_NET = Path(__file__).resolve().parents[1] / "bots/bitter-search/net.npz"


# ---- the net, shared by training (JAX) and play (numpy) ----

def forward_np(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    """Logits ``(batch, 52, 3)``."""
    h = x.astype(np.float32, copy=False)
    for w, b in layers[:-1]:
        h = np.maximum(h @ w + b, 0.0)
    w, b = layers[-1]
    return (h @ w + b).reshape(-1, CARDS, SEATS)


def load_np(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    data = np.load(path)
    layers, k = [], 0
    while f"w{k}" in data:
        layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
        k += 1
    return layers


def chances_np(layers, x: np.ndarray) -> np.ndarray:
    logits = forward_np(layers, x)
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(-1, keepdims=True)


def labels_of(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``hidden`` ``(batch, 156)``: the three other hands, next seat first. Returns the holder of
    each card ``(batch, 52)`` and which cards are unseen ``(batch, 52)``."""
    three = hidden.reshape(-1, SEATS, CARDS)
    return three.argmax(1), three.any(1)


def cross_entropy_np(chances: np.ndarray, holder: np.ndarray, unseen: np.ndarray) -> float:
    """Mean over unseen cards of -log(chance given to the true holder)."""
    p = np.take_along_axis(chances, holder[..., None], -1)[..., 0]
    return float(-(np.log(np.clip(p, 1e-9, 1.0)) * unseen).sum() / unseen.sum())


def size_void_baseline(sizes: np.ndarray, voids: np.ndarray, unseen: np.ndarray) -> np.ndarray:
    """Each unseen card's chance per seat in proportion to that seat's hand size, zero for a seat
    known to be void in the card's suit. ``sizes`` ``(batch, 3)``, ``voids`` ``(batch, 3)`` 4-bit."""
    suit = np.arange(CARDS) // 13
    void = (voids[:, None, :] >> suit[None, :, None]) & 1
    w = np.where(void == 1, 0.0, sizes[:, None, :].astype(np.float64))
    w = np.where(w.sum(-1, keepdims=True) > 0, w, 1.0)
    return w / w.sum(-1, keepdims=True)


# ---- train (JAX) ----

def train(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import optax

    from jaxenv import game as G
    from jaxenv.encode import SIZE
    from jaxenv.selfplay import init_nets, make_rollout

    args.out.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(args.seed)
    key, k_init, k_g, k_c = jax.random.split(key, 4)
    data = np.load(args.net)  # Self-taught's net, as jaxenv.selfplay.save_npz wrote it
    body, k = [], 0
    while f"w{k}" in data:
        body.append({"w": jnp.asarray(data[f"w{k}"]), "b": jnp.asarray(data[f"b{k}"])})
        k += 1
    policy = {"body": body, "policy": {"w": jnp.asarray(data["pw"]), "b": jnp.asarray(data["pb"])},
              "aux": {"w": jnp.zeros((body[-1]["w"].shape[1], 1)), "b": jnp.zeros(1)}}  # fmt: skip
    nets = {"policy": policy, "critic": init_nets(k_c, 512, 3)["critic"]}  # the critic is unused here

    sizes = [SIZE] + [args.hidden] * args.depth + [CARDS * SEATS]
    keys = jax.random.split(k_init, len(sizes) - 1)
    params = [{"w": jax.nn.initializers.he_normal()(k, (a, b)), "b": jnp.zeros(b)}
              for k, a, b in zip(keys, sizes[:-1], sizes[1:])]  # fmt: skip

    def logits_of(params, x):
        h = x
        for layer in params[:-1]:
            h = jax.nn.relu(h @ layer["w"] + layer["b"])
        return (h @ params[-1]["w"] + params[-1]["b"]).reshape(-1, CARDS, SEATS)

    def loss_of(params, x, hidden):
        three = hidden.reshape(-1, SEATS, CARDS)
        holder = three.argmax(1)
        unseen = three.any(1)
        logp = jax.nn.log_softmax(logits_of(params, x), -1)
        nll = -jnp.take_along_axis(logp, holder[..., None], -1)[..., 0]
        return (nll * unseen).sum() / jnp.maximum(unseen.sum(), 1)

    steps_total = args.iterations * (args.batch * args.steps // args.minibatch)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(args.lr, steps_total), weight_decay=1e-4))
    opt = tx.init(params)

    @jax.jit
    def step(params, opt, x, hidden):
        loss, grads = jax.value_and_grad(loss_of)(params, x, hidden)
        updates, opt = tx.update(grads, opt, params)
        return optax.apply_updates(params, updates), opt, loss

    score = jax.jit(loss_of)
    rollout = make_rollout(args.batch, args.steps, 0.9)
    g = jax.vmap(G.new_game)(G.deal_pool(k_g, args.batch))
    log = (args.out / "log.jsonl").open("a")
    n = args.batch * args.steps
    for it in range(1, args.iterations + 1):
        key, k_roll, k_perm = jax.random.split(key, 3)
        g, flat, _ = rollout(nets, policy, g, k_roll)
        x = flat["x"].astype(jnp.float32)
        hidden = flat["hidden"]
        fresh = float(np.mean([float(score(params, x[s : s + 8192], hidden[s : s + 8192])) for s in range(0, n, 8192)]))
        order = jax.random.permutation(k_perm, n)
        total = 0.0
        for s in range(0, n - args.minibatch + 1, args.minibatch):
            rows = order[s : s + args.minibatch]
            params, opt, loss = step(params, opt, x[rows], hidden[rows])
            total += float(loss)
        row = {"iteration": it, "fresh_loss": fresh, "train_loss": total / (n // args.minibatch)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
        if it % 20 == 0 or it == args.iterations:
            save(params, args.out / "guess.npz", {"iteration": it, "fresh_loss": fresh})


def save(params, path: Path, meta: dict) -> None:
    arrays = {}
    for k, layer in enumerate(params):
        arrays[f"w{k}"] = np.asarray(layer["w"], np.float32)
        arrays[f"b{k}"] = np.asarray(layer["b"], np.float32)
    temporary = path.with_name(path.name + ".part.npz")
    np.savez(temporary, meta=json.dumps(meta), **arrays)
    temporary.replace(path)


# ---- positions and score (engine) ----

def positions_game(game: int, spec: str) -> dict[str, np.ndarray]:
    from estimation.bots.knowledge import read_knowledge
    from estimation.engine import Bola, Phase
    from estimation.engine.config import DEFAULT_CONFIG
    from estimation.rl.encoder import encode as rl_encode

    from arena.bots import maker
    from jaxenv.encode_np import encode

    seed = 50_000_000 + game  # practice seeds of their own
    players = [maker(spec, seed + 1 + s, DEFAULT_CONFIG) for s in range(4)]
    bola = Bola(seed, DEFAULT_CONFIG)
    rows: dict[str, list] = {k: [] for k in ("x", "rl", "holder", "unseen", "sizes", "voids")}
    while not bola.is_over:
        seat = bola.current_player
        obs, legal = bola.observation(seat), bola.legal_actions()
        if obs.phase is Phase.PLAY and len(legal) > 1:
            know = read_knowledge(obs)
            others = [(seat + k) % 4 for k in (1, 2, 3)]
            dealt = bola.round.hands  # the engine's hands as dealt (played cards stay in), never shown to a bot
            held = [dealt[o] & know.unseen for o in others]  # what each still holds
            assert held[0] | held[1] | held[2] == know.unseen, "the unseen cards are the others' remaining hands"
            holder = np.zeros(CARDS, np.int64)
            unseen = np.zeros(CARDS, bool)
            for i in range(SEATS):
                for c in range(CARDS):
                    if held[i] >> c & 1:
                        holder[c] = i
                        unseen[c] = True
            rows["x"].append(encode(obs, legal).astype(np.float16))
            rows["rl"].append(rl_encode(obs, DEFAULT_CONFIG).astype(np.float32))
            rows["holder"].append(holder)
            rows["unseen"].append(unseen)
            rows["sizes"].append([know.sizes[o] for o in others])
            rows["voids"].append([know.voids[o] for o in others])
        bola.step(players[seat].act(obs, legal))
    return {k: np.asarray(v) for k, v in rows.items()}


def positions(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor

    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(positions_game, range(args.games), [args.play] * args.games))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["game"] = np.concatenate([np.full(len(p["holder"]), g) for g, p in enumerate(parts)])
    np.savez_compressed(args.out / "positions.npz", **data)
    print(f"{len(data['holder']):,} card decisions from {args.games} games", flush=True)


def score(args: argparse.Namespace) -> None:
    from estimation.bots.rl import policy_at

    d = np.load(args.out / "positions.npz")
    holder, unseen = d["holder"], d["unseen"]
    ours = chances_np(load_np(args.out / "guess.npz"), d["x"].astype(np.float32))
    yosif = np.concatenate([policy_at(str(args.yosif)).card_guesses(d["rl"][s : s + 2048]) for s in range(0, len(holder), 2048)])
    base = size_void_baseline(d["sizes"], d["voids"], unseen)
    out = {
        "positions": int(len(holder)),
        "ours": cross_entropy_np(ours, holder, unseen),
        "yosif": cross_entropy_np(yosif, holder, unseen),
        "baseline": cross_entropy_np(base, holder, unseen),
    }
    out["ours_over_yosif"] = out["ours"] / out["yosif"]
    out["pass"] = bool(out["ours_over_yosif"] <= 1.10 and out["ours"] < out["baseline"])
    (args.out / "score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train")
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--net", type=Path, default=DEFAULT_NET)
    tr.add_argument("--iterations", type=int, default=300)
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--steps", type=int, default=256)
    tr.add_argument("--minibatch", type=int, default=4096)
    tr.add_argument("--hidden", type=int, default=512)
    tr.add_argument("--depth", type=int, default=3)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=7)
    po = commands.add_parser("positions")
    po.add_argument("--out", type=Path, required=True)
    po.add_argument("--games", type=int, default=200)
    po.add_argument("--workers", type=int, default=5)
    po.add_argument("--play", default="sub:bitter-net")
    sc = commands.add_parser("score")
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--yosif", type=Path, default=Path(__file__).resolve().parents[1] / "runs/expert/round-1/hybrid.pt")
    args = parser.parse_args(argv)
    {"train": train, "positions": positions, "score": score}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
