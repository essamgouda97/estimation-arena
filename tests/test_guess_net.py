"""Expectation 1: the guess net's training labels line up with the engine and with the bot."""

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from jaxenv.check import bits, random_round  # noqa: E402
from jaxenv.rules import init_round  # noqa: E402
from jaxenv.selfplay import hidden_of  # noqa: E402
from train.guess_net import labels_of, size_void_baseline  # noqa: E402


def test_labels_are_the_engine_hands_next_seat_first():
    rng = random.Random(11)
    mismatches = 0
    for _ in range(200):
        r = random_round(rng)
        state = init_round(r.round_no, bits(r.hands))
        for seat in range(4):
            hidden = np.asarray(hidden_of(jax.tree.map(lambda v: v[None], state), np.array([seat])))
            holder, unseen = labels_of(hidden)
            for c in range(52):
                owners = [k for k in (1, 2, 3) if r.hands[(seat + k) % 4] >> c & 1]
                want_unseen = bool(owners)
                if unseen[0, c] != want_unseen or (want_unseen and holder[0, c] != owners[0] - 1):
                    mismatches += 1
    assert mismatches == 0


def test_views_match_the_bot_encoder():
    """The net trains on jaxenv.encode and plays on encode_np: same numbers at every step of
    whole rounds (raises on the first difference)."""
    from jaxenv.check import check_rounds

    assert check_rounds(64, 12, batch=64, views=True) > 0


def test_baseline_respects_voids_and_sizes():
    sizes = np.array([[2, 1, 1]])
    voids = np.array([[0b0001, 0, 0]])  # the next seat has no spades (suit 0)
    p = size_void_baseline(sizes, voids, np.ones((1, 52), bool))
    assert p[0, 0, 0] == 0.0 and np.isclose(p[0, 0].sum(), 1.0)
    assert np.allclose(p[0, 20], [0.5, 0.25, 0.25])
