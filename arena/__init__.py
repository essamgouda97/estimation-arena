"""Estimation arena: duplicate matches, ratings and the leaderboard.

Games run one per worker process, so numerical libraries get one thread each: without this,
numpy's BLAS starts a thread per core in every worker and they fight (set before numpy loads).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
