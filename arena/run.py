"""Play one submission against the anchors in duplicate sets and write ``results/<name>.json``.

    uv run python -m arena.run --bot lowcard-wire --sets 10 --seed 0 --workers 4

Each anchor gets its own run with the lineup (submission, anchor, submission, anchor): every
deal is played four times with the seats rotated, so deal luck cancels and the pairwise set
results feed the leaderboard's Elo fit. The result file is small (no moves) and goes in git.
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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from estimation.arena.duplicate import ArenaRun, SetResult, play_set
from estimation.arena.games import changed_rules
from estimation.arena.report import build_report
from estimation.arena.timing import DEFAULT_BUDGET, Timing
from estimation.engine import RulesConfig
from estimation.engine.config import DEFAULT_CONFIG

from .bots import ROOT, SUB_PREFIX, Manifest, load_manifest, maker

ANCHORS_FILE = ROOT / "anchors.json"
RESULTS_DIR = ROOT / "results"
SCHEMA = 1


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


def _task(
    lineup: tuple[str, ...], seed: int, config: RulesConfig, budget: float, index: int
) -> tuple[SetResult, dict[str, Timing]]:
    return play_set(lineup, seed, index, config, make=maker, budget=budget)


def run_sets(
    lineup: Sequence[str],
    sets: int,
    seed: int,
    workers: int = 1,
    config: RulesConfig = DEFAULT_CONFIG,
    budget: float = DEFAULT_BUDGET,
    progress: bool = False,
) -> ArenaRun:
    """``estimation.arena.run_arena`` with submissions allowed in the lineup."""
    names = tuple(lineup)
    task = partial(_task, names, seed, config, budget)
    timings = {name: Timing(budget) for name in dict.fromkeys(names)}
    results: list[SetResult] = []
    start = time.perf_counter()

    def collect(outcome: tuple[SetResult, dict[str, Timing]]) -> None:
        result, set_timings = outcome
        results.append(result)
        for name, timing in set_timings.items():
            timings[name].merge(timing)
        if progress:
            elapsed = time.perf_counter() - start
            print(f"  set {len(results)}/{sets}, {elapsed:.0f} s", file=sys.stderr, flush=True)

    if workers == 1:
        for index in range(sets):
            collect(task(index))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_one_thread) as pool:
            for outcome in pool.map(task, range(sets)):
                collect(outcome)
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


def engine_version() -> str:
    try:
        return version("estimation-engine")
    except PackageNotFoundError:
        return "unknown"


def play_submission(
    manifest: Manifest,
    anchors: Sequence[dict[str, str]],
    sets: int,
    seed: int,
    workers: int,
    progress: bool,
) -> dict[str, Any]:
    me = SUB_PREFIX + manifest.name
    versus: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for anchor in anchors:
        lineup = (me, anchor["spec"], me, anchor["spec"])
        if progress:
            print(f"[{manifest.name}] vs {anchor['name']} ({anchor['spec']})", file=sys.stderr)
        run = run_sets(lineup, sets, seed, workers, progress=progress)
        report = build_report(run)
        mine = next(b for b in report.bots if b.name == me)
        versus.append(
            {
                "anchor": anchor["name"],
                "spec": anchor["spec"],
                "sets": report.sets,
                "games": mine.bolas,
                "mean_score": mine.mean_score,
                "rsa": _interval(mine.rsa),
                "first_place": _interval(mine.first),
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
        "engine": engine_version(),
        "rules": changed_rules(DEFAULT_CONFIG),
        "seed": seed,
        "sets": sets,
        "played_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versus": versus,
        "pairs": pairs,
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
    for pair in old.get("pairs", []):
        other = pair["b"] if pair["a"] == me else pair["a"]
        if other in {v["spec"] for v in kept}:
            result["pairs"].append(pair)
    return [v["anchor"] for v in kept]


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
    args = parser.parse_args(argv)
    if args.sets < 1 or args.workers < 1:
        parser.error("--sets and --workers must be at least 1")
    manifest = load_manifest(args.bot)
    if manifest.needs and not (ROOT / manifest.needs).exists():
        print(f"{manifest.name}: skipped, needs {manifest.needs} (its last result is kept)")
        return 0
    anchors = load_anchors(args.anchors.split(",") if args.anchors else None)
    if not anchors:
        parser.error("no anchors available: check anchors.json")
    result = play_submission(manifest, anchors, args.sets, args.seed, args.workers, args.progress)
    result["bot_hash"] = bot_hash(manifest)
    out = args.out or RESULTS_DIR / f"{manifest.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.is_file() else None
    kept = keep_unplayed(result, old)
    if kept:
        print(f"kept earlier games vs {', '.join(kept)} (bot and engine unchanged)")
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{manifest.name}: {args.sets} sets vs {len(anchors)} anchors, seed {args.seed}")
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
