import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from pettingzoo.test import api_test  # noqa: E402

from arena.env import EstimationEnv  # noqa: E402


def test_pettingzoo_api():
    api_test(EstimationEnv(), num_cycles=200)


def test_random_masked_game_ends_with_win_rewards_summing_to_one():
    env = EstimationEnv()
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    total = dict.fromkeys(env.possible_agents, 0.0)
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        total[agent] += reward
        if terminated or truncated:
            env.step(None)
            continue
        legal = np.flatnonzero(obs["action_mask"])
        env.step(int(rng.choice(legal)))
    assert abs(sum(total.values()) - 1.0) < 1e-9
