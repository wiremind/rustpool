"""A seeded rollout of valid actions on every masked task: the first observation is not
terminal, every mask row keeps a valid action (auto-reset included), and episodes do end."""

import numpy as np

STEPS = 120  # past Maze's 100-step limit, so every task ends some episodes


def _recv_all(pool):
    return [pool.recv() for _ in range(pool.num_envs // pool.batch_size)]


def test_valid_action_rollout_keeps_every_env_playable(env_factory):
    env = env_factory(num_envs=8, batch_size=4, seed=123)
    rng = np.random.default_rng(0)
    env.async_reset()
    batches = _recv_all(env)
    for *_, done, info in batches:
        assert not done.any(), "an initial observation is terminal"
        assert info["action_mask"].any(axis=1).all(), "an initial state has no valid action"
    dones = 0
    for _ in range(STEPS):
        # Per env, in env-id order, so the draws do not depend on which batch an env came in.
        masks = {int(e): m for *_, info in batches
                 for e, m in zip(info["env_id"], info["action_mask"], strict=True)}
        env_ids = np.array(sorted(masks), np.int32)
        actions = np.array([rng.choice(np.flatnonzero(masks[e])) for e in env_ids], np.int32)
        env.send(actions, env_ids)
        batches = _recv_all(env)
        for *_, done, info in batches:
            assert info["action_mask"].any(axis=1).all(), "a state has no valid action"
            dones += int(done.sum())
    assert dones > 0, "no episode ended"
