# How the leaderboard stays fair

What we do, and which competition or paper it comes from. Researched on 7 November 2026.

## What we took

| Practice | Where it comes from | What we do |
|---|---|---|
| Every program plays every other | The Annual Computer Poker Competition (ACPC) plays round robins, and every pair meets in its two-player events. The World Computer-Bridge Championship has used a round robin before its knockout | Every rated bot plays every other, 20 duplicate sets per pair |
| Same deals with seats swapped | ACPC duplicate matches | Already how sets work: every deal four times, seats rotated |
| Fresh deals for each extra match | ACPC 2016 reused one card sequence to separate its top agents. The organizers said this left an effective sample size of two, and they reran the finals | Each pair and batch gets its own deals, derived from the secret seed |
| Say when a result is too close | ACPC reports 95% intervals and flags placings it can't separate with confidence | A sign test marks each head-to-head as clear or too close to call |
| Extra games at the top | ACPC reran the top four, then the top two. The bridge championship sends its top four to a knockout | A close pair among the top four plays more batches, up to 100 sets |
| Stop a match once it is decided | Chess engine testing: Stockfish's Fishtest ends each test with a sequential probability ratio test instead of a fixed number of games | A match stops when its sign test falls below 1 in 1,000, checked after each set from the 10th. Between equal bots a 40-set match stops early less than 0.5% of the time. A shutout stops after 11 sets |
| Toughest benchmark first | Our own choice, so a new bot's first rating comes from its strongest benchmark | `arena.run --due` lists benchmarks by rating, best first, and the round robin starts with the best pairs |
| A ranking that ignores wins over weak bots | ACPC runs total bankroll and instant runoff side by side, and they can disagree | The rating and the runoff are both shown, and the page says when they disagree |

## Why not a rating alone

Elo can't express cycles, where A beats B, B beats C and C beats A. That's a known weakness in
multi-agent evaluation. Nothing in a card game rules cycles out, so ratings from shared opponents
can't settle who is better. Direct games feed the same rating fit, and the head-to-head table
shows any cycles instead of hiding them.

## What we looked at and left out

- **Alpha-Rank and Nash averaging:** built to handle cycles, but Nash averaging is limited to
  two-player zero-sum games, and both are hard to read at a glance. A table of results and a
  runoff are easier to check.
- **TrueSkill with matchmaking:** suits large fields where not everyone can meet. Twelve bots can
  all meet.

## Watching games without giving away the test

A bot could memorize a deal it has seen. The rated deals therefore stay secret, and the games people can watch use public deals and don't count.

## Known limits

- **Two bots at a table:** a match seats two copies of each bot. ACPC's three-player events
  played every combination of three programs. Our equivalent, four different bots at one table,
  isn't done yet.
- **Benchmark games share deals:** a bot's matches against each benchmark use the same seed. New
  direct games don't. Changing this would mean replaying every benchmark match.
- **Near-copies count separately:** research on evaluation notes that cloned agents can skew
  Elo. Several of our bots are close relatives, and nothing limits an author's entries yet.
- **A model-only change** doesn't void results by itself. See [submitting.md](submitting.md).

## How good the evidence is

The ACPC and bridge details come from organizers' reports and press coverage, from 2010 to 2017.
We didn't find ACPC's full rulebook, or any published fair-ladder standard for four-player
trick-taking bots, so this adapts the closest practice rather than following a settled standard.

## Sources

- [ACPC 2016 results amendment](https://groups.google.com/g/computer-poker/c/yOS44TbK0ao)
- [ACPC 2017 no-limit results](https://groups.google.com/g/computer-poker/c/_f-c1ZJwK5w)
- [IEEE Spectrum on the ACPC format](https://spectrum.ieee.org/a-texas-hold-em-tournament-for-ais)
- [ACPC 2010 news](https://poker.cs.ualberta.ca/news_2010.html)
- [World Computer-Bridge Championship reports](https://allevybridge.com/allevy/computerbridge/ICGAJ2012.html)
- [Multiagent Evaluation under Incomplete Information, NeurIPS 2019](https://arxiv.org/pdf/1909.09849v4)
- [Voting-as-Evaluation, including the cloned-agent point](https://export.arxiv.org/pdf/2312.03121)
