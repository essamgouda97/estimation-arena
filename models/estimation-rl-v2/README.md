---
license: mit
tags:
  - reinforcement-learning
  - card-games
  - estimation
---

# RL v2 (Trained bot)

By Yosif Elessawi, published with his permission under the MIT licence. Bid and card networks, copied from a look-ahead bot at the start and then trained by self-play, with 816,000 games in the final stage. No human games were used. It is the benchmark the other bots are measured against. `final.pt` is a PyTorch checkpoint holding only weights and numbers. `rlv2.npz` holds the same weights as NumPy arrays, for the GPU self-play code in `jaxenv/rlv2.py`.

Rated on the [Estimation Arena leaderboard](https://essamgouda97.github.io/estimation-arena/). Code and how to run it: [estimation-arena](https://github.com/essamgouda97/estimation-arena). Running it currently needs the rules engine, which is private.
