"""Play one submission against the anchors in duplicate sets and write ``results/<name>.json``.

    uv run python -m arena.run --bot lowcard-wire --sets 10 --seed 0 --workers 4

Each anchor gets its own run with the lineup (submission, anchor, submission, anchor): every
deal is played four times with the seats rotated, so deal luck cancels and the pairwise set
results feed the leaderboard's Elo fit. The result file is small (no moves) and goes in git.

The ladder plays micro games (``GAME_ROUNDS``): the first 5 rounds of a game, each with 13 cards
and open bidding. ``--rounds 0`` plays full games, as the weekly full-game ladder does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from estimation.arena.duplicate import ROTATIONS, ArenaRun, SetResult, bot_seed, deal_seed, slot_at
from estimation.arena.games import changed_rules
from estimation.arena.match import BotError
from estimation.arena.report import build_report
from estimation.arena.timing import DEFAULT_BUDGET, Timing
from estimation.bots import Bot
from estimation.engine import Bola, IllegalActionError, Phase, RulesConfig
from estimation.engine.config import DEFAULT_CONFIG

from .bots import ROOT, SUB_PREFIX, Manifest, load_manifest, maker
from .live import Match
from .positions import encode_scores, set_outcomes, summarize

ANCHORS_FILE = ROOT / "anchors.json"
RESULTS_DIR = ROOT / "results"
SCHEMA = 1
GAME_ROUNDS = 5  # micro games: a game's first 5 rounds, 13 cards each with open bidding
FULL_GAME = 0  # all 18 rounds: 13 open rounds, then 5 with a fixed trump


def load_anchors(only: Sequence[str] | None = None) -> list[dict[str, str]]:
    """The anchors of ``anchors.json`` whose files are present, or the ones named in ``only``
    (by name or by raw engine spec)."""
    anchors: list[dict[str, str]] = json.loads(ANCHORS_FILE.read_text())
    if only is None:
        return [a for a in anchors if not a.get("needs") or (ROOT / a["needs"]).exists()]
    chosen = []
    for want in only:
        match = next((a for a in anchors if want in (a["name"], a["spec"])), None)
        chosen.append(match or {"name": want, "spec": want})
    return chosen


def _one_thread() -> None:
    """Worker initializer: one thread per process. Left alone, every worker's PyTorch starts a
    thread per core, so 5 workers on 6 cores run 30 threads and the RL anchors crawl."""
    import torch

    torch.set_num_threads(1)


def play_game(
    seed: int, bots: Sequence[Bot], config: RulesConfig, timers: Sequence[Timing], rounds: int = GAME_ROUNDS
) -> Bola:
    """Play one game dealt from ``seed`` and stop after ``rounds`` rounds (0: the whole game).

    Each bot sees only its own seat. Card decisions are timed into ``timers[seat]``. The game's
    ``scores`` after the last round played are its result."""
    bola = Bola(seed, config)
    while not bola.is_over and not (rounds and len(bola.history) >= rounds):
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a game that is not over always has a player to act")
        legal, obs = bola.legal_actions(), bola.observation(seat)
        start = time.perf_counter()
        action = bots[seat].act(obs, legal)
        if obs.phase is Phase.PLAY:
            timers[seat].add(time.perf_counter() - start)
        try:
            bola.step(action)
        except IllegalActionError as error:
            raise BotError(f"seat {seat} ({bots[seat]!r}) chose an illegal action: {error}") from error
    return bola


def play_set(
    lineup: Sequence[str], seed: int, index: int, config: RulesConfig = DEFAULT_CONFIG,
    budget: float = DEFAULT_BUDGET, rounds: int = GAME_ROUNDS,
) -> tuple[SetResult, dict[str, Timing]]:  # fmt: skip
    """Duplicate set ``index``: one deal played once per rotation of the lineup, so every bot
    plays every hand. The same seeds as the engine's own duplicate runner."""
    if len(lineup) != 4:
        raise ValueError("a lineup has exactly 4 bots")
    deals = deal_seed(seed, index)
    timings = {name: Timing(budget) for name in dict.fromkeys(lineup)}
    scores: list[tuple[int, int, int, int]] = []
    played: list[int] = []
    for rotation in range(ROTATIONS):
        names = [lineup[slot_at(seat, rotation)] for seat in range(4)]
        bots = [maker(names[seat], bot_seed(seed, index, rotation, seat), config) for seat in range(4)]
        game = play_game(deals, bots, config, [timings[name] for name in names], rounds)
        scores.append(game.scores)
        played.append(len(game.history))
    return SetResult(index, deals, tuple(scores), tuple(played)), timings


def _task(
    lineup: tuple[str, ...], seed: int, config: RulesConfig, budget: float, rounds: int, index: int
) -> tuple[SetResult, dict[str, Timing]]:
    return play_set(lineup, seed, index, config, budget, rounds)


def run_sets(
    lineup: Sequence[str],
    sets: int,
    seed: int,
    workers: int = 1,
    config: RulesConfig = DEFAULT_CONFIG,
    budget: float = DEFAULT_BUDGET,
    progress: bool = False,
    kind: str = "rating",
    rounds: int = GAME_ROUNDS,
    stop_when_settled: bool = False,
) -> ArenaRun:
    """``estimation.arena.run_arena`` with submissions allowed in the lineup, playing ``rounds``
    rounds a game (0: whole games). ``stop_when_settled`` ends a two-bot match as soon as its result
    is settled (``arena.direct.is_settled``), so a lopsided match doesn't play all ``sets``."""
    from .direct import is_settled

    names = tuple(lineup)
    task = partial(_task, names, seed, config, budget, rounds)
    timings = {name: Timing(budget) for name in dict.fromkeys(names)}
    results: list[SetResult] = []
    start = time.perf_counter()
    match = Match(names, sets, seed, kind)

    def collect(outcome: tuple[SetResult, dict[str, Timing]]) -> None:
        result, set_timings = outcome
        results.append(result)
        for name, timing in set_timings.items():
            timings[name].merge(timing)
        match.update(results)
        if progress:
            elapsed = time.perf_counter() - start
            print(f"  set {len(results)}/{sets}, {elapsed:.0f} s", file=sys.stderr, flush=True)

    def settled() -> bool:
        two = list(dict.fromkeys(names))
        if not stop_when_settled or len(two) != 2 or not is_settled(set_outcomes(results, names, two[0])):
            return False
        if progress:
            print(f"  settled after {len(results)} sets", file=sys.stderr, flush=True)
        return True

    if workers == 1:
        for index in range(sets):
            collect(task(index))
            if settled():
                break
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_one_thread) as pool:
            # Sets come back in order, so a match always stops at the same set for the same seed.
            for outcome in pool.map(task, range(sets)):
                collect(outcome)
                if settled():
                    pool.shutdown(wait=True, cancel_futures=True)
                    break
    match.update(results, finished=True)
    return ArenaRun(
        lineup=names,
        seed=seed,
        config=config,
        results=tuple(results),
        timings=timings,
        wall_seconds=time.perf_counter() - start,
        workers=workers,
    )


def _interval(value: Any) -> dict[str, float | None]:
    finite = lambda x: x if x == x and abs(x) != float("inf") else None  # noqa: E731
    return {"mean": value.mean, "low": finite(value.low), "high": finite(value.high), "sets": value.n}


def _timing(timing: Timing) -> dict[str, float | int]:
    return {
        "count": timing.count,
        "mean_s": timing.mean,
        "p95_s": timing.quantile(0.95),
        "max_s": timing.max,
        "budget_s": timing.budget,
        "over_budget": timing.over,
    }


def engine_version(rounds: int = GAME_ROUNDS) -> str:
    """The engine's version and commit, and how many rounds a game had (``/r5`` for micro games,
    nothing for whole games). A result played on another commit or another game length must be
    replayed, and every check that compares this string does that."""
    try:
        engine = distribution("estimation-engine")
    except PackageNotFoundError:
        version = "unknown"
    else:
        source = json.loads(engine.read_text("direct_url.json") or "{}")
        commit = source.get("vcs_info", {}).get("commit_id", "")
        version = f"{engine.version}+{commit[:8]}" if commit else engine.version
    return f"{version}/r{rounds}" if rounds else version


def base_engine(engine: str) -> str:
    """The engine part of ``engine_version``, without the game length."""
    return engine.split("/")[0]


def play_submission(
    manifest: Manifest,
    anchors: Sequence[dict[str, str]],
    sets: int,
    seed: int,
    workers: int,
    progress: bool,
    rounds: int = GAME_ROUNDS,
    stop_when_settled: bool = False,
) -> dict[str, Any]:
    me = SUB_PREFIX + manifest.name
    versus: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    position_pairs: list[dict[str, Any]] = []
    king_pairs: list[dict[str, Any]] = []
    for anchor in anchors:
        lineup = (me, anchor["spec"], me, anchor["spec"])
        if progress:
            print(f"[{manifest.name}] vs {anchor['name']} ({anchor['spec']})", file=sys.stderr)
        run = run_sets(lineup, sets, seed, workers, progress=progress, rounds=rounds, stop_when_settled=stop_when_settled)
        report = build_report(run)
        mine = next(b for b in report.bots if b.name == me)
        places, by_position = summarize(run.results, lineup)
        position_pairs += by_position["weighted"]
        king_pairs += by_position["king"]
        versus.append(
            {
                "anchor": anchor["name"],
                "spec": anchor["spec"],
                "sets": report.sets,
                "games": mine.bolas,
                "mean_score": mine.mean_score,
                "rsa": _interval(mine.rsa),
                "first_place": _interval(mine.first),
                "positions": places[me]["shares"],
                "position_reward": places[me]["reward"],
                "king": places[me]["king"],
                "delta": places[me]["delta"],
                "scores": encode_scores(run.results),
                "timing": _timing(run.timings[me]),
                "wall_s": run.wall_seconds,
            }
        )
        for pair in report.pairs:
            pairs.append(
                {
                    "a": _public(pair.a),
                    "b": _public(pair.b),
                    "a_wins": pair.a_wins,
                    "draws": pair.draws,
                    "b_wins": pair.b_wins,
                }
            )
    return {
        "schema": SCHEMA,
        "bot": manifest.name,
        "manifest": manifest.to_dict(),
        "engine": engine_version(rounds),
        "rounds": rounds,
        "rules": changed_rules(DEFAULT_CONFIG),
        "seed": seed,
        "sets": sets,
        "played_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versus": versus,
        "pairs": pairs,
        "position_pairs": position_pairs,
        "king_pairs": king_pairs,
    }


def bot_hash(manifest: Manifest) -> str:
    """A fingerprint of what decides the bot's play: every file in its folder except
    manifest.json, plus the manifest fields that change play. Editing a description or the
    method diagram keeps the hash, so earlier results against RL v2 and Hybrid survive."""
    digest = hashlib.sha256()
    play = {k: v for k, v in manifest.to_dict().items() if k in ("kind", "command", "entry", "timeout_s")}
    digest.update(json.dumps(play, sort_keys=True).encode())
    for path in sorted(manifest.folder.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "manifest.json":
            digest.update(path.relative_to(manifest.folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def keep_unplayed(result: dict[str, Any], old: dict[str, Any] | None) -> list[str]:
    """Carry over the old result's games against anchors this run could not play (the RL and
    Hybrid anchors exist only where runs/ is unzipped), if the bot and engine are unchanged.
    Returns the anchors carried over."""
    if not old or old.get("bot_hash") != result["bot_hash"] or old.get("engine") != result["engine"]:
        return []
    played = {v["spec"] for v in result["versus"]}
    kept = [v for v in old.get("versus", []) if v["spec"] not in played]
    for v in kept:
        v["carried_from"] = old.get("played_at")
        result["versus"].append(v)
    me = SUB_PREFIX + result["bot"]
    for key in ("pairs", "position_pairs", "king_pairs"):
        for pair in old.get(key, []):
            other = pair["b"] if pair["a"] == me else pair["a"]
            if other in {v["spec"] for v in kept}:
                result[key].append(pair)
    return [v["anchor"] for v in kept]


def anchors_to_play(
    anchors: list[dict[str, str]], old: dict[str, Any] | None, bot_hash: str, engine: str
) -> list[dict[str, str]]:
    """The anchors an unchanged bot has not met yet; every anchor if the bot or engine changed."""
    if not old or old.get("bot_hash") != bot_hash or old.get("engine") != engine:
        return anchors
    played = {v["spec"] for v in old.get("versus", [])}
    return [a for a in anchors if a["spec"] not in played]


def strongest_first(anchors: list[dict[str, str]]) -> list[dict[str, str]]:
    """The anchors by their current rating, best first, so a new bot meets its toughest benchmark
    first. An anchor without a rating yet goes last."""
    from .leaderboard import build

    rating = {row["key"]: row["elo"] for row in build(ROOT)["rows"] if row["anchor"] and row["elo"] is not None}
    return sorted(anchors, key=lambda a: -rating.get(a["spec"], float("-inf")))


def _public(name: str) -> str:
    """Pair records name submissions ``sub:<name>`` and anchors by spec; keep both as is."""
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.run", description=__doc__.split("\n")[0])
    parser.add_argument("--bot", required=True, help="submission folder name under bots/")
    parser.add_argument("--sets", type=int, default=10, help="duplicate sets per anchor (10)")
    parser.add_argument("--seed", type=int, default=0, help="run seed (0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (1)")
    parser.add_argument(
        "--anchors",
        help="comma-separated anchor names or specs (default: every anchor whose files exist)",
    )
    parser.add_argument("--out", type=Path, help="result file (default results/<bot>.json)")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--only-new", action="store_true", help="skip games already played by this exact bot")
    parser.add_argument("--hide-seed", action="store_true", help="leave the seed out of the result and the output")
    parser.add_argument("--stop-when-settled", action="store_true", help="end a match once more sets could not change its result")
    parser.add_argument(
        "--due",
        action="store_true",
        help="print the specs of the anchors still to play, one per line, and play nothing",
    )
    args = parser.parse_args(argv)
    if args.sets < 1 or args.workers < 1:
        parser.error("--sets and --workers must be at least 1")
    manifest = load_manifest(args.bot)
    if manifest.needs and not (ROOT / manifest.needs).exists():
        if not args.due:
            print(f"{manifest.name}: skipped, needs {manifest.needs} (its last result is kept)")
        return 0
    anchors = load_anchors(args.anchors.split(",") if args.anchors else None)
    if not anchors:
        parser.error("no anchors available: check anchors.json")
    out = args.out or RESULTS_DIR / f"{manifest.name}.json"
    old = json.loads(out.read_text()) if out.is_file() else None
    if args.only_new:
        anchors = anchors_to_play(anchors, old, bot_hash(manifest), engine_version())
    if args.due:
        # The ladder plays one anchor at a time and saves after each, so a slow bot that runs out
        # of time keeps the anchors it finished and carries on from there the next night.
        for anchor in strongest_first(anchors):
            print(anchor["spec"])
        return 0
    if args.only_new:
        if not anchors:
            print(f"{manifest.name}: unchanged, nothing new to play")
            return 0
    result = play_submission(manifest, anchors, args.sets, args.seed, args.workers, args.progress, stop_when_settled=args.stop_when_settled)
    result["bot_hash"] = bot_hash(manifest)
    if args.hide_seed:
        result["seed"] = None
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = keep_unplayed(result, old)
    if kept:
        print(f"kept earlier games vs {', '.join(kept)} (bot and engine unchanged)")
    out.write_text(json.dumps(result, indent=1) + "\n")
    seed_note = "" if args.hide_seed else f", seed {args.seed}"
    print(f"{manifest.name}: {args.sets} sets vs {len(anchors)} anchors{seed_note}")
    for v in result["versus"]:
        rsa = v["rsa"]
        print(
            f"  vs {v['anchor']:<16} RSA {rsa['mean']:+7.1f}  first place {v['first_place']['mean']:.0%}"
            f"  p95 {v['timing']['p95_s'] * 1000:.0f} ms"
        )
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
