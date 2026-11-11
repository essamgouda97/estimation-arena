<h1 align="center"><img src="brand/banner.svg" width="100%" alt="Estimation Arena"></h1>

<p align="center">
  Write a bot for <b>Estimation</b>, the 4-player trick-taking card game. Get an Elo.<br>
  <a href="https://essamgouda97.github.io/estimation-arena/"><b>See the leaderboard</b></a>
</p>

<img src="brand/how-it-works.svg" width="100%" alt="Your bot exchanges JSON and actions with the harness, which uses the rules engine and plays the anchors. Results feed an Elo fit, which feeds the leaderboard page.">

<sub><a href="https://essamgouda97.github.io/estimation-arena/diagrams/how-it-works.html">Explore this diagram</a>, with the source code behind each part.</sub>

## Terms

| Term | Meaning |
|---|---|
| **Anchor** | A fixed bot that every other bot is measured against, so ratings stay comparable. The page calls these benchmarks. |
| **Bola** | One full game of Estimation: 13 open rounds, then 5 with a fixed trump. |
| **Micro game** | The first 5 rounds of a Bola, each with 13 cards and open bidding. The nightly ladder plays these. |
| **King, Sub-King, Sub-Kooz, Kooz** | Final places in a Bola, from first to last. |
| **Set** | One deal played 4 times with the seats rotated, so lucky cards cancel out. |
| **Elo** | A rating from who won each set. 200 points ahead means winning about 76% of sets. Random is fixed at 400. |
| **Weighted rank, King only, score delta** | The three ways bots are ranked: by final place, by coming first, and by points won. See [docs/metrics.md](docs/metrics.md). |
| **Harness** | The code that runs the games, talks to each bot and records the results. |
| **Wire tier, python tier** | Two ways to enter a bot: a program in any language that exchanges JSON, or a Python function. |
| **Ladder** | The nightly run that rates the bots in micro games and rebuilds the page. |
| **Full-game ladder** | The weekly run where the top 5 play each other in whole Bolas, with its own page (`site/full/`). |
| **Smoke test** | A quick 2-set run on a pull request, to check that a bot works. |
| **Seed** | The number that fixes which deals are played. The same seed gives the same deals. |
| **Rung** | A step in our climb past the anchors, in order: Heuristic, PIMC, RL v2, Hybrid. |
| **Self-play** | Training a bot by having it play against itself. |

The code and results use different names for some bots than the page does:

| In the code | On the page | What it does |
|---|---|---|
| Random | Random | Plays any legal move |
| Heuristic | Basic rules | Follows rules a person wrote |
| PIMC | Look-ahead | Imagines the hidden hands and plays each card out |
| Search bot | Look-ahead, tuned rules | Look-ahead that plays cards out with tuned rules |
| RL v2 | Trained bot | Two trained models, one for bidding and one for cards |
| Hybrid (B-Tig) | Trained bot + look-ahead | Those models, plus look-ahead |

Every bot's code name is shown on the page too.

## One decision

<img src="brand/one-decision.svg" width="100%" alt="The harness says hello, sends an observation and the legal action ids, and the bot answers with one id. An illegal id or over 5 seconds is a forfeit. Action ids: 0-1 dash, 2-51 bids, 52 with, 53 pass, 54-67 estimates, 68-119 cards.">

<sub><a href="https://essamgouda97.github.io/estimation-arena/diagrams/one-decision.html">Explore this diagram</a>, with the source code behind each part.</sub>

## Two doors

<img src="brand/two-doors.svg" width="100%" alt="To compete, a manifest declares a wire bot in any language or a Python function. To train, use the PettingZoo environment in arena/env.py with your own policy, then wrap the policy in a manifest.">

## Quickstart

```bash
uv sync --group rl                        # needs read access to the engine repo
uv run python -m arena.models fetch       # downloads the trained models from Hugging Face
uv run pytest -q
uv run python -m arena.run --bot table-pressure --sets 10 --workers 4
uv run python -m arena.leaderboard && open site/index.html
```

## Submit

<img src="brand/submit.svg" width="100%" alt="Open a pull request. A smoke test plays 2 sets against Random and Heuristic and comments the result. After merging, the nightly ladder rates the bot on a secret seed and the page updates.">

Examples: `bots/lowcard-wire` (stdlib, 40 lines), `bots/table-pressure` (rung 1, beats Heuristic). Full steps: [docs/submitting.md](docs/submitting.md).

```bash
uv run python -m arena.check <your-bot>   # run this before opening a pull request
```

## Play

```bash
uv sync --group table && uv run python -m arena.serve    # http://127.0.0.1:8790/
```

Any mix of people and bots per table; agents join a person's seat over the same WebSocket.
Protocol: [docs/table.md](docs/table.md).

## Fair by construction

| Rule | Why |
|---|---|
| Duplicate sets: every deal played 4 times, seats rotated | Deal luck cancels |
| Fresh process per game, no seed sent | No peeking across rotations |
| Rating from final position: King, Sub-King, Sub-Kooz, Kooz | The Bola pays by position, and stacked doubles make raw scores explode |
| Leaderboard seed is a secret | No tuning to the test deals |
| Every bot plays every other bot, on fresh deals per pair | A top spot has to be earned head to head |
| Results show when they're too close to call | A small lead isn't presented as a real one |
| Runoff next to the rating | Beating weak bots can't win it, because weak bots leave first |

Reasoning and sources: [docs/fair-ranking.md](docs/fair-ranking.md).

## Eval set

[GoudaGame/estimation-eval](https://huggingface.co/datasets/GoudaGame/estimation-eval) has 1,200 games between bots on fixed practice deals, with reference scores to compare against. Official ratings use separate deals. Built by `scripts/build-eval-set.py`.

## Where it runs

The page ranks every bot three ways, explained in [docs/metrics.md](docs/metrics.md). On the home server, `/board` follows matches set by set while they play.

`/ladder` follows the nightly ladder while it runs on GitHub. The ladder pushes each set's result to the `ladder-live` branch every 30 seconds, with no cards and no seed, and the home server fetches it. The rated deals stay secret, so to show cards the home server plays the same pairing on public deals alongside. Those games never count. See `arena/ladder_live.py`.

| Piece | Where | Cost |
|---|---|---|
| Smoke + nightly ladder, every anchor | GitHub Actions | free (public repo) |
| Trained models | [Hugging Face](https://huggingface.co/GoudaGame) | free |
| Training, long runs | home server | free |
| Page | GitHub Pages | free |
| Live table and board | home server, invite only | free |

Secrets: `ENGINE_TOKEN` (read access to the engine repo), `LADDER_SEED`.
Training design: [docs/training-design.md](docs/training-design.md).
GPU rules for self-play at scale: [docs/jax-rules.md](docs/jax-rules.md).

## Models

Trained models live on Hugging Face. `models.lock.json` pins each one to an exact version and checksum, so every result traces back to the file that played. To change a model, run `scripts/publish-models.py` and commit the updated lock. The code here is [MIT licensed](LICENSE); each model states its own licence.

## Using a coding agent?

You need read access to the rules engine. Then paste this:

```text
Clone https://github.com/essamgouda97/estimation-arena, read AGENTS.md and follow it. Set up
the environment and the models, run the tests and show me the standings. Then train or build
a bot in bots/<my-name>/ that beats the top of the leaderboard. Confirm it on a fresh seed
with at least 150 sets against Hybrid, rate it with arena.run, and open a pull request with
the numbers.
```
