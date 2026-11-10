---
license: mit
tags:
  - card-games
  - estimation
---

# Guess model for Self-taught + guesses

For each card a player cannot see, guesses which of the other three players holds it. It reads the same view of the table as the Self-taught model and was trained only on the Self-taught model's games against itself, with no human games and nothing from other bots. The file is a NumPy `.npz` of the network's weights (three hidden layers of 512).

The Self-taught + guesses bot uses it to imagine the hidden hands it searches over.

Rated on the [Estimation Arena leaderboard](https://essamgouda97.github.io/estimation-arena/). Code and how to run it: [estimation-arena](https://github.com/essamgouda97/estimation-arena) (`train/guess_net.py`, `bots/guess-search`). Running it currently needs the rules engine, which is private.
