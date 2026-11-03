"""Rung 3: improve card play with reinforcement learning (PPO), starting from imitation.

    .venv-gpu/bin/python -m train.rl_cards --out bots/rl-cards/net.npz --logdir ~/workspace/estimation-data/tb

The diagnosis (experiments/rung3.md): giving our bot RL v2's bidding barely closed the gap to
RL v2, so the gap is card play. This trainer fine-tunes the imitation card net by self-play.

- Episode: one round, fresh deal, carry 1. Bidding: HeuristicBot in every seat.
- Card play: the learner (sampled from its policy) in some seats; in the others either the
  learner too (self-play) or the frozen imitation net it started from.
- Reward per learner card decision: its seat's round score minus the mean of the other three,
  / 50, the same for every card of the round; a value head is the baseline.
- Start: the imitation net's weights (bots/net-pimc/net.npz) plus a fresh value head. A KL
  penalty to the starting policy keeps early updates from undoing what imitation learned.
- Evaluation: greedy learner in seats 0 and 2 vs the frozen imitation net in 1 and 3, then
  swapped, same deals: the learner's edge in points per round.
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

import estimation.bots  # noqa: F401
from estimation.bots import HeuristicBot
from estimation.engine import Phase, Round
from estimation.rl.encoder import encode

from train.rl_bid import MLP, ROOT, load_card_net, new_round

PLAY_BASE = 68


class CardPolicy(nn.Module):
    """The imitation MLP (body + 52-card head) with a value head on its last hidden layer."""

    def __init__(self, init: MLP) -> None:
        super().__init__()
        self.body = nn.ModuleList(init.layers[:-1])
        self.policy = init.layers[-1]
        self.value = nn.Linear(self.policy.in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.body:
            x = F.relu(layer(x))
        return self.policy(x), self.value(x)[:, 0]


@torch.no_grad()
def play(rounds: list[Round], learner_seats, policy: CardPolicy, frozen: MLP, device, greedy: bool):
    heuristic = HeuristicBot()
    rec = {"x": [], "mask": [], "action": [], "logp": [], "value": [], "round": [], "seat": []}
    active = list(range(len(rounds)))
    while active:
        mine, theirs, actions = [], [], {}
        for i in active:
            r = rounds[i]
            seat = r.current_player
            obs = r.observation(seat)
            legal = r.legal_actions()
            if r.phase is not Phase.PLAY:
                actions[i] = heuristic.act(obs, legal)
            elif len(legal) == 1:
                actions[i] = legal[0]
            elif seat in learner_seats[i]:
                mine.append((i, seat, encode(obs), r.legal_mask()[PLAY_BASE:]))
            else:
                theirs.append((i, encode(obs), r.legal_mask()[PLAY_BASE:]))
        if theirs:
            x = torch.as_tensor(np.stack([t[1] for t in theirs]), device=device)
            m = torch.as_tensor(np.stack([t[2] for t in theirs]), device=device)
            for (i, _, _), a in zip(theirs, frozen(x).masked_fill(~m, -1e9).argmax(1).tolist()):
                actions[i] = PLAY_BASE + a
        if mine:
            xs = np.stack([t[2] for t in mine])
            ms = np.stack([t[3] for t in mine])
            logits, value = policy(torch.as_tensor(xs, device=device))
            logits = logits.masked_fill(~torch.as_tensor(ms, device=device), -1e9)
            a = logits.argmax(1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            logp = F.log_softmax(logits, 1).gather(1, a[:, None])[:, 0]
            for k, (i, seat, x, m) in enumerate(mine):
                actions[i] = PLAY_BASE + int(a[k])
                for key, val in (("x", x), ("mask", m), ("action", int(a[k])), ("logp", float(logp[k])),
                                 ("value", float(value[k])), ("round", i), ("seat", seat)):
                    rec[key].append(val)
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


def evaluate(policy, frozen, device, n: int, seed: int) -> float:
    edge = []
    for learners in ((0, 2), (1, 3)):
        rng = random.Random(seed)
        rounds = [new_round(rng) for _ in range(n)]
        _, rew = play(rounds, [learners] * n, policy, frozen, device, greedy=True)
        others = [s for s in range(4) if s not in learners]
        edge.append(rew[:, list(learners)].mean(1) - rew[:, others].mean(1))
    return float(np.concatenate(edge).mean() * 50.0)


def save(policy: CardPolicy, path: Path, meta: dict) -> None:
    arrays = {}
    for k, layer in enumerate(list(policy.body) + [policy.policy]):
        arrays[f"w{k}"] = layer.weight.detach().cpu().numpy().T.astype(np.float16)
        arrays[f"b{k}"] = layer.bias.detach().cpu().numpy().astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, meta=json.dumps(meta))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.rl_cards", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, default=ROOT / "bots" / "net-pimc" / "net.npz")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--rounds", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.003)
    p.add_argument("--kl", type=float, default=0.05, help="weight of KL to the imitation policy")
    p.add_argument("--selfplay", type=float, default=0.5)
    p.add_argument("--value-warmup", type=int, default=5, help="iterations training only the value head")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-rounds", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = load_card_net(a.init, device)
    policy = CardPolicy(load_card_net(a.init, device)).to(device).train()
    writer = None
    if a.logdir:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(a.logdir.expanduser() / f"rl-cards-{a.out.stem}"))
    best = evaluate(policy, frozen, device, a.eval_rounds, 999)
    print(f"iteration 0: edge vs imitation net {best:+.2f} points/round (should be ~0)", flush=True)
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "history": [{"iteration": 0, "edge": best}]}
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr)
    rng = random.Random(a.seed + 1)
    start = time.perf_counter()
    best = -1e9
    for it in range(1, a.iterations + 1):
        rounds = [new_round(rng) for _ in range(a.rounds)]
        seats = [(0, 1, 2, 3) if rng.random() < a.selfplay else tuple(sorted(rng.sample(range(4), 2))) for _ in rounds]
        rec, rew = play(rounds, seats, policy, frozen, device, greedy=False)
        x = torch.as_tensor(np.stack(rec["x"]), device=device)
        m = torch.as_tensor(np.stack(rec["mask"]), device=device)
        act = torch.as_tensor(rec["action"], device=device)
        old_logp = torch.as_tensor(rec["logp"], device=device)
        ret = torch.as_tensor(rew[rec["round"], rec["seat"]], device=device)
        adv = ret - torch.as_tensor(rec["value"], device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        with torch.no_grad():
            ref_logp_all = F.log_softmax(frozen(x).masked_fill(~m, -1e9), 1)
        n = len(act)
        for _ in range(a.epochs):
            order = torch.randperm(n, device=device)
            for k in range(0, n, 8192):
                idx = order[k : k + 8192]
                logits, value = policy(x[idx])
                logp_all = F.log_softmax(logits.masked_fill(~m[idx], -1e9), 1)
                vloss = F.mse_loss(value, ret[idx])
                if it <= a.value_warmup:
                    loss = vloss
                else:
                    logp = logp_all.gather(1, act[idx][:, None])[:, 0]
                    ratio = torch.exp(logp - old_logp[idx])
                    pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * adv[idx]).mean()
                    probs = logp_all.exp()
                    entropy = -(probs * logp_all.masked_fill(~m[idx], 0)).sum(1).mean()
                    kl = (probs * (logp_all - ref_logp_all[idx]).masked_fill(~m[idx], 0)).sum(1).mean()
                    loss = pg + 0.5 * vloss - a.entropy * entropy + a.kl * kl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
        elapsed = time.perf_counter() - start
        if writer:
            writer.add_scalar("train/value_loss", vloss.item(), it)
            writer.add_scalar("train/decisions", n, it)
            writer.add_scalar("speed/rounds_per_s", a.rounds * it / elapsed, it)
            if it > a.value_warmup:
                writer.add_scalar("train/kl_to_imitation", kl.item(), it)
                writer.add_scalar("train/entropy", entropy.item(), it)
        if it % a.eval_every == 0 or it == a.iterations:
            edge = evaluate(policy, frozen, device, a.eval_rounds, 999)
            meta["history"].append({"iteration": it, "edge": edge})
            if writer:
                writer.add_scalar("eval/edge_vs_imitation", edge, it)
            mark = ""
            if edge > best and it > a.value_warmup:
                best = edge
                save(policy, a.out, meta)
                mark = "  saved"
            print(f"iteration {it}: edge {edge:+.2f} points/round, {a.rounds * it / elapsed:.0f} rounds/s{mark}", flush=True)
        if writer:
            writer.flush()
    print(f"best edge {best:+.2f}, wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
