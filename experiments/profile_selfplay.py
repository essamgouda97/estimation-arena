"""Where does self-play time go? Plays full games with four copies of one bot on one core and
splits the time into rules engine, observation encoder, neural net and everything else.

    uv run python experiments/profile_selfplay.py --bot rl:checkpoint=runs/m4-rules-v2/checkpoints/final.pt --games 20

This is the measurement behind docs/training-design.md: it decides whether batching the net
(option 1) or moving the rules onto the GPU (option 2) is the bigger win.
"""

import argparse
import cProfile
import pstats
import time

import torch

from estimation.bots import make_bot
from estimation.engine import Bola

GROUPS = (
    ("neural net (torch)", ("torch/", "torch\\")),
    ("observation encoder", ("estimation/rl/encoder", "estimation/rl/standings")),
    ("rules engine", ("estimation/engine/",)),
    ("bot logic", ("estimation/bots/", "estimation/rl/")),
)


def play(spec: str, games: int, seed: int) -> tuple[int, int]:
    decisions = 0
    for g in range(games):
        bots = [make_bot(spec, seed + 4 * g + s) for s in range(4)]
        bola = Bola(seed + g)
        while not bola.is_over:
            seat = bola.current_player
            bola.step(bots[seat].act(bola.observation(seat), bola.legal_actions()))
            decisions += 1
    return games, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)  # one self-play worker = one core
    play(args.bot, 1, 10_000)  # warm up imports and caches
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    games, decisions = play(args.bot, args.games, args.seed)
    profiler.disable()
    wall = time.perf_counter() - start
    stats = pstats.Stats(profiler).stats  # type: ignore[attr-defined]
    totals = dict.fromkeys([g for g, _ in GROUPS] + ["other python"], 0.0)
    others: dict[str, float] = {}
    for (file, _line, name), (_cc, _nc, tottime, _ct, _callers) in stats.items():
        if file == "~":  # C functions: file them by the module named in the function
            file = "torch/" if "torch" in name else ("numpy/" if "numpy" in name else file)
        for group, needles in GROUPS:
            if any(n in file for n in needles):
                totals[group] += tottime
                break
        else:
            totals["other python"] += tottime
            key = f"{file.rsplit('/', 2)[-2:]} {name}" if file != "~" else name
            others[key] = others.get(key, 0.0) + tottime
    profiled = sum(totals.values())
    print(f"{args.bot}: {games} games, {decisions} decisions, {wall:.1f} s on one core")
    print(f"  {games / wall:.2f} games/s, {decisions / wall:.0f} decisions/s, {1e6 * wall / decisions:.0f} us/decision")
    for group, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {group:<22} {100 * seconds / profiled:5.1f}%")
    top = sorted(others.items(), key=lambda kv: -kv[1])[:6]
    print("  top of 'other python':", "; ".join(f"{k} {100 * v / profiled:.1f}%" for k, v in top))


if __name__ == "__main__":
    main()
