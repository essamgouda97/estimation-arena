"""Film a training run: at each saved model version, record what it thinks and how it plays.

    JAX_PLATFORMS=cpu .venv-jax/bin/python -m jaxenv.lab_capture ~/workspace/estimation-data/bitter/run2

For every new ``net.npz`` (at least ``--every`` iterations apart) it writes ``lab/snap-NNNNN.json``:
- probes: the same fixed positions asked of every version, so its opinion can be compared directly;
- game: one Bola, this version in seats 0 and 2 against HeuristicBot, on fixed deals, with the
  version's probabilities for each of its decisions. It is replayed in the JAX rules, and
  ``verified`` says whether both engines agreed at every step;
- quick: final positions over a few games against HeuristicBot (few games, so noisy).
"""

import argparse
import json
import time
from pathlib import Path
from zipfile import BadZipFile

import jax
import jax.numpy as jnp
import numpy as np

import estimation.bots  # noqa: F401
from estimation.bots import HeuristicBot
from estimation.engine import Bola, Phase, Round, cards_of
from estimation.engine.cards import deal

from arena.positions import position_shares

from . import rules as R
from .encode_np import encode
from .table_labels import label

PROBE_SEED, GAME_SEED = 4242, 7777
QUICK_SEEDS = (101, 102, 103)


class Net:
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.layers, k = [], 0
        while f"w{k}" in data:
            self.layers.append((data[f"w{k}"], data[f"b{k}"]))
            k += 1
        self.head = (data["pw"], data["pb"])
        self.iteration = int(json.loads(str(data["meta"])).get("iteration", 0))

    def probs(self, obs, legal) -> dict[int, float]:
        x = encode(obs, legal)
        for w, b in self.layers:
            x = np.maximum(x @ w + b, 0.0)
        z = (x @ self.head[0] + self.head[1])[legal]
        z = np.exp(z - z.max())
        return dict(zip(legal, (z / z.sum()).tolist()))


def probes(net: Net) -> dict:
    """The fixed positions: the opening bid after four No Dash, and the lead after an auction."""
    hands = deal(PROBE_SEED, 1)
    r = Round.new(1, hands)
    dash = net.probs(r.observation(0), r.legal_actions())
    for _ in range(4):
        r.step(R.NO_DASH_CALL)
    opening = net.probs(r.observation(0), r.legal_actions())
    lead = Round.new(1, hands)
    for a in (0, 0, 0, 0, R.FIRST_BID + 2 * 5 + 3, R.PASS, R.PASS, R.PASS, R.PASS):  # seat 0: 6 spades
        lead.step(a)
    while lead.phase is Phase.ESTIMATION:  # the others estimate 1 (or 2 where 1 is barred)
        legal = lead.legal_actions()
        lead.step(R.ESTIMATE_BASE + 1 if R.ESTIMATE_BASE + 1 in legal else legal[0])
    card = net.probs(lead.observation(lead.current_player), lead.legal_actions())
    return {
        "hand": cards_of(hands[0]),
        "dash_call": dash[R.DASH_CALL],
        "opening": {str(a): round(p, 4) for a, p in opening.items()},
        "lead": {str(a - R.PLAY_BASE): round(p, 4) for a, p in card.items()},
    }


def bits(hands) -> np.ndarray:
    return np.array([[(h >> c) & 1 for c in range(52)] for h in hands], dtype=bool)


_step = jax.jit(R.step)
_legal = jax.jit(R.legal_mask)
_init = jax.jit(R.init_round)


def play(net: Net | None, seed: int, net_seats=(0, 2), record: bool = True) -> dict:
    """One Bola against HeuristicBot; with ``record``, every event, checked in the JAX rules."""
    heuristic = HeuristicBot()
    bola = Bola(seed)
    rounds, verified = [], True
    while not bola.is_over:
        rnd = bola.round
        ev = []
        s = _init(rnd.round_no, bits(rnd.hands), -1 if rnd.fixed_trump is None else rnd.fixed_trump, rnd.carry, rnd.scores) if record else None
        info = {"no": rnd.round_no, "fixed": rnd.fixed_trump, "carry": rnd.carry, "scores": list(rnd.scores),
                "hands": [cards_of(h) for h in rnd.hands], "first": rnd.first}  # fmt: skip
        while bola.round is rnd and not bola.is_over and rnd.phase is not Phase.DONE:
            seat, obs, legal = bola.current_player, bola.observation(bola.current_player), bola.legal_actions()
            if record and sorted(np.flatnonzero(np.asarray(_legal(s))).tolist()) != legal:
                verified = False
            p = None
            if seat in net_seats:
                options = net.probs(obs, legal)
                action = max(legal, key=options.__getitem__)
                if record and len(legal) > 1:
                    top = sorted(options.items(), key=lambda kv: -kv[1])[:8]
                    p = [[a, round(x, 3)] for a, x in top]
            else:
                action = heuristic.act(obs, legal)
            won_before = obs.tricks_won
            bola.step(action)
            if record:
                s = _step(s, action)
                after = rnd.observation(seat)
                e = {"s": seat, "a": action, "l": label(action)}
                if p:
                    e["p"] = p
                if int(rnd.phase) != int(obs.phase):
                    e["ph"] = int(rnd.phase)
                if after.tricks_won != won_before:
                    e["w"] = next(k for k in range(4) if after.tricks_won[k] != won_before[k])
                if after.estimates != obs.estimates:
                    e["est"] = list(after.estimates)
                if after.trump != obs.trump:
                    e["tr"] = after.trump
                if after.caller != obs.caller:
                    e["ca"] = after.caller
                ev.append(e)
        result = rnd.result()
        if record:
            if (list(np.asarray(s.result).tolist()), int(s.next_carry)) != (list(result.scores), result.next_carry):
                verified = False
            info |= {"ev": ev, "result": list(result.scores), "skipped": result.skipped, "tricks": [x.tricks for x in result.seats]}
            rounds.append(info)
    shares = [position_shares(bola.scores, k) for k in net_seats]
    out = {"scores": list(bola.scores), "shares": [sum(x) / len(shares) for x in zip(*shares)], "rounds_played": len(bola.history)}
    if record:
        out |= {"rounds": rounds, "verified": verified, "net_seats": list(net_seats), "seed": seed}
    return out


def capture(run: Path, every: int, source: Path | None = None) -> int:
    lab = run / "lab"
    lab.mkdir(exist_ok=True)
    index = lab / "index.jsonl"
    seen = {json.loads(l)["iteration"] for l in index.read_text().splitlines()} if index.is_file() else set()
    source = source or run / "net.npz"
    if not source.is_file():
        return -1
    net = Net(source)
    if net.iteration in seen or (seen and net.iteration - max(seen) < every):
        return -1
    t = time.perf_counter()
    game = play(net, GAME_SEED)
    quick = [play(net, seed, seats, record=False) for seed in QUICK_SEEDS for seats in ((0, 2), (1, 3))]
    shares = [sum(q["shares"][p] for q in quick) / len(quick) for p in range(4)]
    snap = {"iteration": net.iteration, "probes": probes(net), "game": game, "quick": shares}
    path = lab / f"snap-{net.iteration:05d}.json"
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(snap, separators=(",", ":")))
    tmp.replace(path)
    row = {"iteration": net.iteration, "quick": shares, "verified": game["verified"], "probes": snap["probes"],
           "scores": game["scores"], "rounds": game["rounds_played"]}  # fmt: skip
    with index.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"iter {net.iteration}: verified {game['verified']}, King {shares[0]:.0%} Kooz {shares[3]:.0%}, {time.perf_counter() - t:.0f} s", flush=True)
    return net.iteration


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m jaxenv.lab_capture", description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--backfill", action="store_true", help="film the saved net-*.npz once, then exit")
    a = p.parse_args(argv)
    run = a.run.expanduser()
    if a.backfill:
        for path in sorted(run.glob("net-*.npz")) + [run / "net.npz"]:
            capture(run, a.every, path)
        return 0
    while True:
        try:
            capture(run, a.every)
        except (OSError, ValueError, EOFError, KeyError, BadZipFile) as error:  # net.npz mid-write: try again
            print("retry:", error, flush=True)
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
