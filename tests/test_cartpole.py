"""CartPole-v1: next-state parity with gymnasium's CartPoleEnv, termination, truncation.

rustpool and gymnasium use different RNGs, so the test reads the env's exact f64 state through
``query_state(env_id, "state")``, puts it into gymnasium's env, steps both with the same action
and compares the next states (and the termination flag).
"""

import json

import numpy as np
import pytest

import rustpool

gym_cartpole = pytest.importorskip("gymnasium.envs.classic_control.cartpole")

NUM_ENVS, TRANSITIONS = 64, 10_000


def state_of(pool, env_id):
    s = pool.query_state(env_id, "state")
    return np.asarray(json.loads(s) if isinstance(s, str) else s, dtype=np.float64)


def test_registered_with_defaults():
    assert "CartPole-v1" in rustpool.list_envs()
    pool = rustpool.make("CartPole-v1", num_envs=2)
    try:
        assert pool.observation_spec() == {"obs": ([4], "float32")}
        assert tuple(pool.action_spec()) == (2,)
        assert pool.action_mask_width() == 0
    finally:
        pool.close()


def test_next_state_parity_with_gymnasium():
    gym_env = gym_cartpole.CartPoleEnv()
    gym_env.reset(seed=0)
    pool = rustpool.make("CartPole-v1", num_envs=NUM_ENVS, batch_size=NUM_ENVS, seed=1)
    rng = np.random.default_rng(0)
    ids = np.arange(NUM_ENVS, dtype=np.int32)
    checked = terminations = 0
    max_err = 0.0
    try:
        pool.async_reset()
        obs, _, _, _ = pool.recv()
        while checked < TRANSITIONS:
            before = np.stack([state_of(pool, i) for i in ids])
            steps_before = [int(pool.query_state(i, "steps")) for i in ids]
            np.testing.assert_array_equal(obs["obs"], before.astype(np.float32))
            actions = rng.integers(0, 2, NUM_ENVS).astype(np.int32)
            pool.send(actions, ids)
            obs, reward, done, info = pool.recv()
            assert np.all(reward == 1.0)
            after = np.stack([state_of(pool, i) for i in ids])
            for i in ids:
                gym_env.state = before[i].copy()
                gym_env.steps_beyond_terminated = None
                g_obs, g_rew, g_term, _, _ = gym_env.step(int(actions[i]))
                g_state = np.asarray(gym_env.state, dtype=np.float64)
                truncated = steps_before[i] + 1 >= 500
                assert bool(done[i]) == (g_term or truncated)
                if done[i]:
                    terminations += int(g_term)
                    continue  # rustpool auto-reset: `after` is the next episode's start
                err = float(np.max(np.abs(after[i] - g_state)))
                max_err = max(max_err, err)
                assert err <= 1e-6, (i, before[i], after[i], g_state)
                checked += 1
    finally:
        pool.close()
    assert terminations > 0
    assert max_err <= 1e-6


def test_truncates_at_500_steps():
    """A balancing controller keeps the pole up; the episode then ends by truncation at 500."""
    pool = rustpool.make("CartPole-v1", num_envs=4, batch_size=4, seed=3)
    ids = np.arange(4, dtype=np.int32)
    lengths = np.zeros(4, int)
    ended = np.zeros(4, bool)
    try:
        pool.async_reset()
        obs, _, _, _ = pool.recv()
        for _ in range(600):
            x, x_dot, theta, theta_dot = obs["obs"].T
            actions = (theta + 0.5 * theta_dot + 0.01 * x + 0.05 * x_dot > 0).astype(np.int32)
            pool.send(actions, ids)
            obs, _, done, _ = pool.recv()
            lengths += ~ended
            ended |= done
            if ended.all():
                break
    finally:
        pool.close()
    assert ended.all()
    assert np.all(lengths == 500), lengths
