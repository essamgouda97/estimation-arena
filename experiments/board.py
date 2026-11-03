"""Pictures-only training board for TensorBoard: methods, data, results and a game replay.

    .venv-gpu/bin/python experiments/board.py --data ~/workspace/estimation-data/rung2 \\
        --trace trace.json --tb ~/workspace/estimation-data/tb

Writes one TensorBoard run, ``board``, with image tabs:
methods/ (one diagram per bot), data/ (the recorded decisions), results/ (leaderboard, regret
ladder, matches vs PIMC), games/ (a replay: drag the step slider). Needs matplotlib + tensorboard.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BG, FG, MUTED, GRID = "#111418", "#e8e6e0", "#8a8780", "#2a2d33"
BLUE, ORANGE, GREEN, RED, GOLD, GRAY = "#3987e5", "#eb6834", "#1baf7a", "#e66767", "#eda100", "#5f5e5a"
SUITS, RANKS = "♣♦♥♠", "23456789TJQKA"
SUIT_COLOR = {0: FG, 1: RED, 2: RED, 3: FG}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
})


def card_label(c: int) -> str:
    return RANKS[c % 13] + SUITS[c // 13]


# ---------- methods ----------

def flow(boxes: list[tuple[str, str]], title: str) -> plt.Figure:
    """A left-to-right chain of boxes; each is (label, color)."""
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    n = len(boxes)
    w, gap = (11.4 - 0.5 * (n - 1)) / n, 0.5
    for i, (label, color) in enumerate(boxes):
        x = 0.3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 1.3, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color + "33", ec=color, lw=1.5))
        ax.text(x + w / 2, 1.15, label, ha="center", va="center", fontsize=11.5, color=FG)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.05, 1.15), (x + w + gap - 0.05, 1.15),
                                         arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.5))
    ax.text(0.3, 2.25, title, fontsize=15, color=FG, weight="bold")
    return fig


def methods() -> dict[str, plt.Figure]:
    return {
        "methods/1_heuristic": flow([("hand value\nper trump", GRAY), ("bid / estimate\nnearest value", GRAY),
                                     ("win until\nestimate", GRAY), ("then duck", GRAY)], "Heuristic"),
        "methods/2_table_pressure": flow([("Heuristic", GRAY), ("count trumps\nnobody can beat", BLUE),
                                          ("enough to finish?", BLUE), ("duck now,\nthey win later", GREEN)],
                                         "table-pressure (rung 1)"),
        "methods/3_pimc": flow([("legal cards", GRAY), ("deal hidden cards\nx30", ORANGE),
                                ("play each out,\nHeuristic rollouts", ORANGE), ("best average\nscore", GREEN)],
                               "PIMC (anchor)"),
        "methods/4_imitation": flow([("PIMC-100 plays\n800 games", ORANGE), ("602k decisions\n+ card values", GRAY),
                                     ("net 512x3\nGTX 1070", BLUE), ("net picks\nthe card", BLUE)], "pimc-clone"),
        "methods/6_rl_cards": flow([("imitation net\n(rung 2)", BLUE), ("self-play rounds\nPPO, GPU batched", "#8e7cf0"),
                                    ("+6.7 points\nper round", GREEN)], "rl-cards (rung 3)"),
        "methods/7_rl_pimc": flow([("RL card net\nkeeps its top 2", BLUE), ("deal hidden\ncards x50", ORANGE),
                                   ("play each\nof the 2 out", ORANGE), ("best average", GREEN)], "rl-pimc (rung 3)"),
        "methods/5_net_pimc": flow([("legal cards", GRAY), ("net keeps\nits top 3", BLUE),
                                    ("deal hidden\ncards x50", ORANGE), ("play each\nof the 3 out", ORANGE),
                                    ("best average", GREEN)], "net-pimc (rung 2)"),
    }


# ---------- data ----------

def data_figures(shards: list[str]) -> dict[str, plt.Figure]:
    parts = [np.load(s) for s in shards[:120]]
    legal = np.concatenate([p["legal"] for p in parts])
    values = np.concatenate([p["values"] for p in parts])
    choice = np.concatenate([p["choice"] for p in parts]).astype(int)
    figs = {}

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = legal.sum(1)
    ax.hist(counts, bins=np.arange(1.5, 14.5), color=BLUE, rwidth=0.85)
    ax.set_xlabel("legal cards per decision")
    figs["data/legal_cards"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    v = np.where(legal, values, -np.inf)
    srt = np.sort(v, 1)
    gap = srt[:, -1] - srt[:, -2]
    ax.hist(np.clip(gap, 0, 10), bins=60, color=ORANGE)
    ax.axvline(np.median(gap), color=FG, ls="--", lw=1)
    ax.set_xlabel("best card minus second best (points)")
    figs["data/near_ties"] = fig

    fig, ax = plt.subplots(figsize=(10, 3.6))
    chosen = np.bincount(choice, minlength=52)
    offered = legal.sum(0)
    rate = np.where(offered > 0, chosen / np.maximum(offered, 1), np.nan).reshape(4, 13)
    im = ax.imshow(rate, cmap="magma", vmin=0, vmax=np.nanmax(rate), aspect="auto")
    ax.set_yticks(range(4), list(SUITS))
    ax.set_xticks(range(13), list(RANKS))
    ax.set_title("how often the teacher plays a card when it is legal", color=MUTED, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.025)
    figs["data/play_rate_by_card"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    best = values[np.arange(len(choice)), choice]
    ax.hist(np.clip(best, -40, 60), bins=80, color=GREEN)
    ax.set_xlabel("teacher's value of its card (own minus others, points)")
    figs["data/value_of_choice"] = fig
    return figs


# ---------- results ----------

def diagnosis() -> plt.Figure:
    """Rung 3: swapping either half with RL v2's leaves the gap the same."""
    rows = [("Heuristic bids + net-pimc cards", -20.8, -29.5, -12.1),
            ("RL v2 bids + net-pimc cards", -18.0, -26.0, -10.0),
            ("Heuristic bids + RL v2 cards", -19.4, -30.5, -8.2)]
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, (label, m, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=RED, lw=3, solid_capstyle="round")
        ax.plot(m, i, "o", color=RED, ms=9)
    ax.axvline(0, color=FG, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("points per game vs RL v2")
    return fig


def leaderboard(path: Path) -> plt.Figure:
    rows = [r for r in json.loads(path.read_text())["rows"] if r["elo"] is not None and r["name"] != "Random"]
    rows.sort(key=lambda r: r["elo"])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1))
    colors = [GRAY if r["anchor"] else BLUE for r in rows]
    ax.barh([r["name"] for r in rows], [r["elo"] for r in rows], color=colors, height=0.6)
    for i, r in enumerate(rows):
        ax.text(r["elo"] + 20, i, f"{r['elo']:,.0f}", va="center", color=FG, fontsize=11)
    ax.set_xlim(3000, max(r["elo"] for r in rows) + 400)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=GRAY, label="anchors (Yosif)"), Patch(color=BLUE, label="ours")],
              frameon=False, labelcolor=FG, loc="lower right")
    ax.set_xlabel("Elo")
    return fig


def regret_ladder(nets: Path, gap_log: Path) -> plt.Figure:
    meta = json.loads(str(np.load(nets / "full-512x3.npz")["meta"]))
    last = meta["history"][-1]
    gap = gap_log.read_text()
    pimc30 = float(re.search(r"PIMC-30 \(anchor\) regret:\s+([\d.]+)", gap).group(1))
    heur = float(re.search(r"HeuristicBot regret:\s+([\d.]+)", gap).group(1))
    items = [("random card", meta["baselines"]["random"], GRAY), ("Heuristic", heur, GRAY),
             ("net alone", last["regret"], BLUE), ("PIMC-30 anchor", pimc30, ORANGE),
             ("best of net top 2", last["top2_regret"], GREEN), ("best of net top 3", last["top3_regret"], GREEN)]
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [i[0] for i in items][::-1]
    ax.barh(names, [i[1] for i in items][::-1], color=[i[2] for i in items][::-1], height=0.6)
    for k, (_, val, _) in enumerate(items[::-1]):
        ax.text(val + 0.04, k, f"{val:.2f}", va="center", color=FG, fontsize=11)
    ax.set_xlabel("points lost per card vs PIMC-100")
    return fig


DEFAULT_TOP = {"net-pimc": 3, "rl-pimc": 2}  # search bots run without a top= option use these


def matches(logs: list[str], opponent: str = "pimc") -> plt.Figure:
    pat = re.compile(r"^\s+(.+?)\s+sets\s+(\d+)-(\d+)-(\d+)\s+first\s+[\d.]+%\s+RSA\s+([+-][\d.]+)\s+\[([+-][\d.]+),\s*([+-][\d.]+)\]")
    head_re = re.compile(r"^(\S+) vs (\S+), \d+ sets")
    rows = []
    for log in logs:
        bot = vs = None
        for line in Path(log).read_text().splitlines():
            h = head_re.match(line)
            if h:
                bot, vs = h.group(1), h.group(2)
                continue
            m = pat.match(line)
            if not m or bot is None or not vs.startswith(opponent):
                continue
            g = m.group(1)
            net = re.search(r"net=\S*?([\w-]+)\.npz", g)
            tag = f" [{net.group(1)}]" if net else ""
            if bot == "combo":
                label = (g.replace(";checkpoint=runs/m4-rules-v2/checkpoints/final.pt", "")
                          .replace("sub;", "").replace("bid=", "bids ").replace("cards=", "+ cards "))
            elif "top=" in g:
                sm = re.search(r"samples=(\d+)", g)
                label = f"{bot}{tag}: top {re.search(r'top=(\d+)', g).group(1)}" + (f" x {sm.group(1)} deals" if sm else "")
            elif bot in DEFAULT_TOP:
                label = f"{bot}{tag}: top {DEFAULT_TOP[bot]}"
            else:
                label = f"{bot}{tag}: net alone"
            rows.append((label, int(m.group(2)), int(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))))
    fig, ax = plt.subplots(figsize=(9, 0.6 * max(1, len(rows)) + 1))
    for i, (label, w, l, rsa, lo, hi) in enumerate(rows):
        color = GREEN if lo > 0 else (RED if hi < 0 else GOLD)
        ax.plot([lo, hi], [i, i], color=color, lw=3, solid_capstyle="round")
        ax.plot(rsa, i, "o", color=color, ms=9)
        ax.text(max(hi, 0) + 1.5, i, f"{w}-{l}", va="center", color=MUTED, fontsize=11)
    ax.axvline(0, color=FG, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel(f"points per game vs {'PIMC' if opponent == 'pimc' else 'RL v2'}, 95% interval")
    return fig


# ---------- games ----------

def draw_card(ax, x: float, y: float, c: int, ec: str = GRID, lw: float = 1.0, alpha: float = 1.0) -> None:
    ax.add_patch(FancyBboxPatch((x, y), 0.62, 0.86, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#1c2026", ec=ec, lw=lw, alpha=alpha))
    ax.text(x + 0.31, y + 0.43, card_label(c), ha="center", va="center", fontsize=11,
            color=SUIT_COLOR[c // 13], alpha=alpha)


def replay_frame(step: dict, seats: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(15, 6.2))
    fig._manual = True
    table = fig.add_axes([0.01, 0.02, 0.6, 0.9])
    table.set_xlim(0, 14)
    table.set_ylim(0, 8.4)
    table.axis("off")
    trump = "Sans" if step["trump"] == 4 else SUITS[step["trump"]]
    fig.text(0.01, 0.95, f"round {step['round']}   trump {trump}", fontsize=15, color=FG, weight="bold")
    trick = {s: c for s, c in step["trick"]}
    legal, top, chosen = set(step["legal"]), set(step["top"]), step["chosen"]
    for seat in range(4):
        y = 7.2 - seat * 1.55
        color = BLUE if seats[seat] == "net-pimc" else ORANGE
        est = step["estimates"][seat]
        table.text(0, y + 0.43, f"{seat} {seats[seat]}", color=color, fontsize=12, va="center")
        table.text(0, y + 0.0, f"won {step['won'][seat]} · est {'-' if est is None else est}", color=MUTED, fontsize=11, va="center")
        hand = sorted(step["hands"][seat], key=lambda c: (c // 13, c % 13))
        for k, c in enumerate(hand):
            if seat == 0 and c == chosen:
                ec, lw = GOLD, 3
            elif seat == 0 and c in top:
                ec, lw = GREEN, 2
            elif seat == 0 and c in legal:
                ec, lw = BLUE, 1.5
            else:
                ec, lw = GRID, 1
            draw_card(table, 2.6 + k * 0.72, y, c, ec, lw, 1.0 if seat != 0 or c in legal else 0.55)
        if seat in trick:
            draw_card(table, 12.6, y, trick[seat], FG, 2)
    bars = fig.add_axes([0.66, 0.12, 0.15, 0.78])
    vals = fig.add_axes([0.84, 0.12, 0.15, 0.78], sharey=bars)
    order = sorted(range(len(step["legal"])), key=lambda i: -step["prob"][i])
    labels = [card_label(step["legal"][i]) for i in order]
    ys = np.arange(len(order))[::-1]
    pc = [GOLD if step["legal"][i] == chosen else (GREEN if step["legal"][i] in top else GRAY) for i in order]
    bars.barh(ys, [step["prob"][i] for i in order], color=pc, height=0.6)
    bars.set_yticks(ys, labels)
    bars.set_title("net", color=BLUE, fontsize=12)
    bars.set_xlim(0, 1)
    vals.barh(ys, [step["teacher"][i] for i in order], color=pc, height=0.6)
    vals.axvline(0, color=MUTED, lw=1)
    vals.set_title("PIMC-100 value", color=ORANGE, fontsize=12)
    plt.setp(vals.get_yticklabels(), visible=False)
    return fig


def score_race(trace: dict) -> plt.Figure:
    s = np.array(trace["scores"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for seat in range(4):
        color = BLUE if trace["seats"][seat] == "net-pimc" else ORANGE
        ax.plot(range(1, len(s) + 1), s[:, seat], color=color, lw=2, ls="-" if seat < 2 else "--",
                marker="o", ms=4, label=f"{seat} {trace['seats'][seat]}")
    ax.set_xlabel("round")
    ax.legend(frameon=False, labelcolor=FG)
    ax.grid(color=GRID)
    return fig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="rung2 folder: pimc100/, nets/, *.log")
    p.add_argument("--leaderboard", type=Path, default=Path("site/leaderboard.json"))
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--tb", type=Path, required=True)
    p.add_argument("--png", type=Path, help="also save every figure as PNG here")
    a = p.parse_args()
    data = a.data.expanduser()
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(a.tb.expanduser() / "board"))

    def emit(tag: str, fig: plt.Figure, step: int = 0) -> None:
        if not getattr(fig, "_manual", False):
            fig.tight_layout()
        if a.png:
            a.png.mkdir(parents=True, exist_ok=True)
            fig.savefig(a.png / f"{tag.replace('/', '_')}_{step:03d}.png", dpi=90)
        writer.add_figure(tag, fig, step)
        plt.close(fig)

    for tag, fig in methods().items():
        emit(tag, fig)
    for tag, fig in data_figures(sorted(glob.glob(str(data / "pimc100" / "*.npz")))).items():
        emit(tag, fig)
    emit("results/leaderboard", leaderboard(a.leaderboard))
    emit("results/regret_ladder", regret_ladder(data / "nets", data / "teacher-gap.log"))
    emit("results/vs_pimc", matches(sorted(glob.glob(str(data / "*.log")))))
    emit("results/rung3_diagnosis", diagnosis())
    rung3 = data.parent / "rung3"
    if rung3.is_dir():
        emit("results/vs_rl_v2", matches(sorted(glob.glob(str(rung3 / "*.log"))), opponent="rl:"))
    trace = json.loads(a.trace.read_text())
    for i, step in enumerate(trace["steps"]):
        emit("games/replay", replay_frame(step, trace["seats"]), i)
    emit("games/score_race", score_race(trace))
    writer.close()
    print(f"board: {len(trace['steps'])} replay frames")


if __name__ == "__main__":
    main()
