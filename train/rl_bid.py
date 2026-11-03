"""Rung 3: learn to bid with reinforcement learning (PPO), card play held fixed.

    .venv-gpu/bin/python -m train.rl_bid --out bots/rl-bid/bid.npz --logdir ~/workspace/estimation-data/tb

Runs in the GPU environment with the engine installed (train/README.md). One process plays
thousands of rounds in lockstep: at every step it gathers each round's pending decision and
answers them in two batched forward passes on the GPU, one for card play, one for bidding.
That is option 1 of docs/training-design.md.

- Episode: one round (open or fixed-trump), a fresh deal, carry 1.
- Card play, every seat: the imitation net from bots/net-pimc (greedy), so cards are fast and
  identical for everyone and the only thing that differs is bidding.
- Bidding (Dash Call, auction, estimate, super call): the learner in some seats, HeuristicBot
  in the others; half the rounds are pure self-play.
- Reward per learner decision: its seat's round score minus the mean of the other three, / 50.
- Warm start: a few epochs cloning HeuristicBot's bids, so PPO starts from a sensible policy.
- Evaluation: the same deals twice with the roles swapped (learner in seats 0 and 2, then 1
  and 3, against HeuristicBot bidding): the learner's mean edge in points per round.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import estimation.bots  # noqa: F401  (import order)
from estimation.bots import HeuristicBot
from estimation.engine import Phase, Round
from estimation.engine.cards import deal
from estimation.engine.round import FIXED_ORDER, OPEN_ROUNDS
from estimation.rl.encoder import NUM_FEATURES, encode

BID_ACTIONS = 68  # 0-1 dash, 2-51 bids, 52 with, 53 pass, 54-67 estimates
PLAY_BASE = 68
ROOT = Path(__file__).resolve().parents[1]


class MLP(nn.Module):
    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class BidNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.body = MLP([NUM_FEATURES, hidden, hidden])
        self.policy = nn.Linear(hidden, BID_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.body(x))
        return self.policy(h), self.value(h)[:, 0]


def load_card_net(path: Path, device: torch.device) -> MLP:
    d = np.load(path)
    ws = []
    k = 0
    while f"w{k}" in d:
        ws.append((d[f"w{k}"].astype(np.float32), d[f"b{k}"].astype(np.float32)))
        k += 1
    net = MLP([ws[0][0].shape[0]] + [w.shape[1] for w, _ in ws]).to(device)
    with torch.no_grad():
        for layer, (w, b) in zip(net.layers, ws):
            layer.weight.copy_(torch.as_tensor(w.T))
            layer.bias.copy_(torch.as_tensor(b))
    return net.eval()


def new_round(rng: random.Random) -> Round:
    round_no = rng.randint(1, OPEN_ROUNDS + len(FIXED_ORDER))
    fixed = FIXED_ORDER[round_no - OPEN_ROUNDS - 1] if round_no > OPEN_ROUNDS else None
    return Round.new(round_no, deal(rng.randrange(2**31), round_no), fixed)


@torch.no_grad()
def play(
    rounds: list[Round],
    learner_seats: list[tuple[int, ...]],
    bidnet: BidNet,
    cardnet: MLP,
    device: torch.device,
    greedy: bool,
) -> tuple[dict[str, list], np.ndarray]:
    """Play every round to the end in lockstep. Returns the learner's decisions and the
    (rounds, 4) relative rewards."""
    heuristic = HeuristicBot()
    rec: dict[str, list] = {"x": [], "mask": [], "action": [], "logp": [], "value": [], "round": [], "seat": []}
    active = list(range(len(rounds)))
    while active:
        cards, bids, actions = [], [], {}
        for i in active:
            r = rounds[i]
            seat = r.current_player
            obs = r.observation(seat)
            if r.phase is Phase.PLAY:
                legal = r.legal_actions()
                if len(legal) == 1:
                    actions[i] = legal[0]
                else:
                    cards.append((i, encode(obs), r.legal_mask()[PLAY_BASE:]))
            elif seat in learner_seats[i]:
                bids.append((i, seat, encode(obs), r.legal_mask()[:BID_ACTIONS]))
            else:
                actions[i] = heuristic.act(obs, r.legal_actions())
        if cards:
            x = torch.as_tensor(np.stack([c[1] for c in cards]), device=device)
            m = torch.as_tensor(np.stack([c[2] for c in cards]), device=device)
            pick = cardnet(x).masked_fill(~m, -1e9).argmax(1).cpu().numpy()
            for (i, _, _), a in zip(cards, pick):
                actions[i] = PLAY_BASE + int(a)
        if bids:
            xs = np.stack([b[2] for b in bids])
            ms = np.stack([b[3] for b in bids])
            logits, value = bidnet(torch.as_tensor(xs, device=device))
            logits = logits.masked_fill(~torch.as_tensor(ms, device=device), -1e9)
            if greedy:
                a = logits.argmax(1)
            else:
                a = torch.distributions.Categorical(logits=logits).sample()
            logp = F.log_softmax(logits, 1).gather(1, a[:, None])[:, 0]
            for k, (i, seat, x, m) in enumerate(bids):
                actions[i] = int(a[k])
                rec["x"].append(x)
                rec["mask"].append(m)
                rec["action"].append(int(a[k]))
                rec["logp"].append(float(logp[k]))
                rec["value"].append(float(value[k]))
                rec["round"].append(i)
                rec["seat"].append(seat)
        still = []
        for i in active:
            rounds[i].step(actions[i])
            if rounds[i].phase is not Phase.DONE:
                still.append(i)
        active = still
    rewards = np.zeros((len(rounds), 4), np.float32)
    for i, r in enumerate(rounds):
        s = np.array(r.result().scores, np.float32)
        rewards[i] = (s - (s.sum() - s) / 3) / 50.0
    return rec, np.clip(rewards, -5, 5)


def evaluate(bidnet: BidNet, cardnet: MLP, device: torch.device, n: int, seed: int) -> float:
    """Learner bidding vs HeuristicBot bidding on the same deals, roles swapped: points/round."""
    edge = []
    for learners in ((0, 2), (1, 3)):
        rng = random.Random(seed)
        rounds = [new_round(rng) for _ in range(n)]
        _, rew = play(rounds, [learners] * n, bidnet, cardnet, device, greedy=True)
        others = tuple(s for s in range(4) if s not in learners)
        edge.append(rew[:, list(learners)].mean(1) - rew[:, list(others)].mean(1))
    return float(np.concatenate(edge).mean() * 50.0)


def warm_start(bidnet: BidNet, cardnet: MLP, device: torch.device, rounds: int, epochs: int, seed: int) -> None:
    """Clone HeuristicBot's bidding: play rounds with HeuristicBot bidding in every seat (the
    learner records but HeuristicBot decides), then fit the policy head by cross-entropy."""
    rng = random.Random(seed)
    heuristic = HeuristicBot()
    xs, ms, ys = [], [], []
    for _ in range(rounds):
        r = new_round(rng)
        while r.phase is not Phase.DONE:
            seat = r.current_player
            obs = r.observation(seat)
            legal = r.legal_actions()
            a = heuristic.act(obs, legal)
            if r.phase is not Phase.PLAY and len(legal) > 1:
                xs.append(encode(obs))
                ms.append(r.legal_mask()[:BID_ACTIONS])
                ys.append(a)
            r.step(a)
    x = torch.as_tensor(np.stack(xs), device=device)
    m = torch.as_tensor(np.stack(ms), device=device)
    y = torch.as_tensor(ys, device=device)
    opt = torch.optim.Adam(bidnet.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(y), device=device)
        for k in range(0, len(y), 2048):
            idx = order[k : k + 2048]
            logits, _ = bidnet(x[idx])
            loss = F.cross_entropy(logits.masked_fill(~m[idx], -1e9), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits, _ = bidnet(x)
        agree = (logits.masked_fill(~m, -1e9).argmax(1) == y).float().mean().item()
    print(f"warm start: {len(y)} Heuristic bidding decisions, agreement {agree:.1%}", flush=True)


def save(bidnet: BidNet, path: Path, meta: dict) -> None:
    arrays = {}
    layers = list(bidnet.body.layers) + [bidnet.policy]
    for k, layer in enumerate(layers):
        arrays[f"w{k}"] = layer.weight.detach().cpu().numpy().T.astype(np.float16)
        arrays[f"b{k}"] = layer.bias.detach().cpu().numpy().astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, meta=json.dumps(meta))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.rl_bid", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cardnet", type=Path, default=ROOT / "bots" / "net-pimc" / "net.npz")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--rounds", type=int, default=1024, help="rounds per iteration")
    p.add_argument("--epochs", type=int, default=4, help="PPO epochs per iteration")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--selfplay", type=float, default=0.5, help="share of rounds with the learner in all seats")
    p.add_argument("--warm-rounds", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-rounds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cardnet = load_card_net(a.cardnet, device)
    bidnet = BidNet(a.hidden).to(device)
    writer = None
    if a.logdir:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(a.logdir.expanduser() / f"rl-bid-{a.out.stem}"))
    if a.warm_rounds:
        warm_start(bidnet, cardnet, device, a.warm_rounds, 8, a.seed)
    best = evaluate(bidnet, cardnet, device, a.eval_rounds, 999)
    print(f"iteration 0: edge vs Heuristic bidding {best:+.2f} points/round", flush=True)
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "history": [{"iteration": 0, "edge": best}]}
    save(bidnet, a.out, meta)
    if writer:
        writer.add_scalar("eval/edge_points_per_round", best, 0)
    opt = torch.optim.Adam(bidnet.parameters(), lr=a.lr)
    rng = random.Random(a.seed + 1)
    start = time.perf_counter()
    for it in range(1, a.iterations + 1):
        rounds = [new_round(rng) for _ in range(a.rounds)]
        seats = []
        for _ in rounds:
            if rng.random() < a.selfplay:
                seats.append((0, 1, 2, 3))
            else:
                seats.append(tuple(sorted(rng.sample(range(4), 2))))
        rec, rew = play(rounds, seats, bidnet, cardnet, device, greedy=False)
        x = torch.as_tensor(np.stack(rec["x"]), device=device)
        m = torch.as_tensor(np.stack(rec["mask"]), device=device)
        act = torch.as_tensor(rec["action"], device=device)
        old_logp = torch.as_tensor(rec["logp"], device=device)
        ret = torch.as_tensor(rew[rec["round"], rec["seat"]], device=device)
        adv = ret - torch.as_tensor(rec["value"], device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        n = len(act)
        for _ in range(a.epochs):
            order = torch.randperm(n, device=device)
            for k in range(0, n, 4096):
                idx = order[k : k + 4096]
                logits, value = bidnet(x[idx])
                logits = logits.masked_fill(~m[idx], -1e9)
                logp_all = F.log_softmax(logits, 1)
                logp = logp_all.gather(1, act[idx][:, None])[:, 0]
                ratio = torch.exp(logp - old_logp[idx])
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * adv[idx]).mean()
                probs = logp_all.exp()
                entropy = -(probs * logp_all.masked_fill(~m[idx], 0)).sum(1).mean()
                vloss = F.mse_loss(value, ret[idx])
                loss = pg + 0.5 * vloss - a.entropy * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(bidnet.parameters(), 1.0)
                opt.step()
        elapsed = time.perf_counter() - start
        if writer:
            writer.add_scalar("train/mean_reward", float(ret.mean()), it)
            writer.add_scalar("train/entropy", entropy.item(), it)
            writer.add_scalar("train/value_loss", vloss.item(), it)
            writer.add_scalar("train/decisions", n, it)
            writer.add_scalar("speed/rounds_per_s", a.rounds * it / elapsed, it)
        if it % a.eval_every == 0 or it == a.iterations:
            edge = evaluate(bidnet, cardnet, device, a.eval_rounds, 999)
            meta["history"].append({"iteration": it, "edge": edge})
            if writer:
                writer.add_scalar("eval/edge_points_per_round", edge, it)
            mark = ""
            if edge > best:
                best = edge
                save(bidnet, a.out, meta)
                mark = "  saved"
            print(f"iteration {it}: edge {edge:+.2f} points/round, entropy {entropy.item():.2f}, "
                  f"{a.rounds * it / elapsed:.0f} rounds/s{mark}", flush=True)
        if writer:
            writer.flush()
    print(f"best edge {best:+.2f} points/round, wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
