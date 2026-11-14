---
license: mit
tags:
  - reinforcement-learning
  - card-games
  - estimation
---

# Self-taught

Plays Estimation with one neural network for every decision, bids and cards alike. It learned from nothing by playing itself, rewarded for where it finished at the table: first whole games, then the five-round games the leaderboard plays. The file also holds a small second network that learned, from the same games, where a seat is likely to finish given the scores so far; the look-ahead bots use it to judge a round by what it does to the game. The file is a NumPy `.npz` of the network's weights. Used by the Self-taught, Self-taught + look-ahead and Self-taught + guesses bots.

Rated on the [Estimation Arena leaderboard](https://essamgouda97.github.io/estimation-arena/). Code and how to run it: [estimation-arena](https://github.com/essamgouda97/estimation-arena). Running it currently needs the rules engine, which is private.
