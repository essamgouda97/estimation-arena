"""What every number on the leaderboards means: one explanation, shown on both pages."""

_PLACES = ("King", "Sub-King", "Sub-Kooz", "Kooz")


def _scale(values: tuple[str, str, str, str]) -> str:
    cells = "".join(f'<div class="g-cell g{k}"><span>{_PLACES[k]}</span><b>{v}</b></div>' for k, v in enumerate(values))
    return f'<div class="g-scale" role="img" aria-label="{", ".join(f"{p} {v}" for p, v in zip(_PLACES, values))}">{cells}</div>'


GUIDE = f"""<section class="guide" id="guide" aria-labelledby="guide-title">
<h2 id="guide-title">What the numbers mean</h2>
<p class="g-lead">Bots are ranked in three ways, using the same games. Each way asks a different question.</p>
<div class="g-metrics">
<article><h3>Weighted rank</h3><p>How high does it usually finish? Each game pays by final place, and first and last count most. This is the default ranking.</p>{_scale(("+1", "+0.2", "−0.2", "−1"))}<p class="g-even">Evenly matched bots average 0.</p></article>
<article><h3>King only</h3><p>How often does it come first? Nothing else counts.</p>{_scale(("1", "0", "0", "0"))}<p class="g-even">Evenly matched bots come first 25% of the time.</p></article>
<article><h3>Score delta</h3><p>How many points does it win by? This is its average game score minus its opponent's. One huge game can move it.</p><div class="g-formula"><span>its score</span><b>−</b><span>opponent's score</span></div><p class="g-even">Evenly matched bots average 0.</p></article>
</div>
<h3 class="g-read">Reading the standings</h3>
<ul class="g-terms">
<li>The standings show each of the three as a rating, like chess Elo. A bot rated 200 points higher wins about 76% of sets. Random is fixed at 400, and ratings compare only within a column.</li>
<li>Ranked games are short: the first 5 rounds of a game, each with 13 cards and open bidding, so every bot can be rated every night. Once a week the top 5 also play whole games of 18 rounds against each other, on their own page.</li>
<li>A set is one deal played four times with the seats rotated, so lucky cards cancel out.</li>
<li>Benchmark bots are fixed, and every other bot is measured against them.</li>
<li>An early result comes from only a few test matches, so treat a small lead as a tie.</li>
</ul>
</section>"""

# Each page defines --guide-line and --muted; the place colours are the same on both.
STYLE = """.guide{margin-top:36px}.guide h2{margin:0 0 6px}.guide h3{margin:0 0 6px;font-size:17px}
.g-lead{margin:0 0 16px;max-width:70ch}
.g-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:24px 32px;margin-bottom:24px}
.g-metrics p{margin:0}.g-metrics article>p:first-of-type{min-height:6em}
.g-scale{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin:10px 0 8px}
.g-cell{padding:6px 4px;border-radius:3px;text-align:center;color:#1c2147;font-size:12px;line-height:1.3}.g-cell b{display:block;font-size:16px}
.g0{background:#f2c14e}.g1{background:#f8e3a8}.g2{background:#f3a6a8}.g3{background:#e5484d;color:#fff}
.g-formula{display:flex;gap:10px;align-items:center;margin:10px 0 8px;min-height:54px}.g-formula span{flex:1;padding:8px 4px;text-align:center;border:1px solid var(--guide-line);border-radius:3px;font-size:13px}
.g-even{color:var(--muted);font-size:14px}
.g-read{margin:0 0 6px}.g-terms{margin:0;padding:0 0 0 18px;max-width:75ch;color:var(--muted);font-size:14px}.g-terms li{margin:0 0 4px}
@media(max-width:680px){.g-metrics article>p:first-of-type{min-height:0}}
"""
