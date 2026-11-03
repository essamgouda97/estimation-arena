# AGENTS.md

Instructions for coding agents working in this repo. Humans: see README.md.

## What this is

A competition harness for bots that play Estimation. The rules engine is
[estimation-engine](https://github.com/YosifElessawi/estimation-engine), a **private** repo
installed by uv from git. This repo holds only the harness, submissions, results and the
leaderboard page. Never copy engine code or the engine's `runs/` data into this repo.

## Setup

```bash
uv sync                    # needs read access to the engine repo (git over HTTPS or SSH)
uv sync --group rl         # optional: PettingZoo env for training (arena/env.py)
uv run pytest -q           # must pass before any commit
```

No access to the engine repo? Stop and ask the owner; do not vendor or reimplement the engine.
Optional `runs/` folder (RL v2 and Hybrid anchors): unzip the owner's Drive archive at the repo
root so `runs/m4-rules-v2/checkpoints/final.pt` exists. It is git-ignored; keep it that way.

## Map

| Path | Role |
|---|---|
| `arena/protocol.py` | Wire protocol: one JSON line in, one action id out, one process per game |
| `arena/bots.py` | Loads `bots/<name>/manifest.json`; `sub:<name>:key=value` passes options |
| `arena/run.py` | Rates one bot vs every anchor, writes `results/<name>.json` |
| `arena/sweep.py` | Tunes a python bot's options vs one opponent |
| `arena/env.py` | PettingZoo AEC environment for RL training |
| `arena/leaderboard.py` | Elo fit, renders `site/` |
| `anchors.json` | Frozen engine bots everyone is rated against |
| `docs/training-design.md` | How self-play is made GPU-bound |
| `train/` | Imitation (`record`, `imitate`) and self-play RL (`rl_cards`, `rl_bid`); see its README |
| `experiments/` | Write-ups per rung, the TensorBoard board, diagnostic bots that are never rated |

## Adding or changing a bot

1. `bots/<name>/manifest.json` plus code. Python tier: `make(seed, rules, **options)`.
   Wire tier: any language, see `bots/lowcard-wire`.
2. Tune with `arena.sweep` on a **tuning seed** (default 1000). Confirm the winner on a fresh
   seed with at least 1,000 sets before believing it. Trust a difference only when the 95%
   intervals do not overlap.
3. Rate with `arena.run`. Never tune against the leaderboard seed (`LADDER_SEED` secret).
4. Open a pull request. The `smoke` workflow plays it and comments.

## Rules of the game you must not break

- Default `RulesConfig` only. A rules change is a new season, decided by the owner.
- A bot sees only its own `Observation`. No reading other hands, no seeds, no shared state
  between games. Wire bots get a fresh process per game.
- Card decisions should stay under 0.25 s; the hard limit is 5 s.
- Raw scores can explode: an all-lose round doubles the next and doubles stack. Compare bots by
  set wins, first place and Elo, not raw score.

## Compute

- Heavy runs (sweeps over 100 sets, ratings vs RL v2 or Hybrid, any training) go on the owner's
  home server (`ssh home-server`, checkout `~/workspace/estimation-arena`), niced. The Mac stays
  a light checkout.
- GPU is a GTX 1070 (Pascal). PyTorch 2.9+ cannot use it: use PyTorch 2.8.0 with CUDA 12.6
  wheels or `jax[cuda12]`, in a separate environment from the engine's.

## Hygiene

- Never print or commit tokens, `.env` files or secrets. CI secrets: `ENGINE_TOKEN`, `LADDER_SEED`.
- Commit `results/*.json` and `site/` only from real runs; never hand-edit them.
- Keep experiments reproducible: record the command, seed and sets next to the result.
