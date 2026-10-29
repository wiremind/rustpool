import numpy as np


def _obs_spec_shapes(env_pool):
    spec = env_pool.observation_spec()
    return {key: tuple(shape) for key, (shape, _dtype) in spec.items()}


def test_initial_reset_and_receive(env_pool):
    """Resetting should deliver one batch per worker with correctly shaped data."""
    env_pool.async_reset()

    expected_batches = env_pool.num_envs // env_pool.batch_size
    received_ids = set()
    obs_shapes = _obs_spec_shapes(env_pool)
    action_dim = env_pool.action_spec()[0]

    for _ in range(expected_batches):
        obs, rew, done, info = env_pool.recv()

        assert rew.shape == (env_pool.batch_size,)
        assert done.shape == (env_pool.batch_size,)
        assert info["env_id"].shape == (env_pool.batch_size,)
        assert info["action_mask"].shape == (env_pool.batch_size, action_dim)

        for key, shape in obs_shapes.items():
            assert obs[key].shape == (env_pool.batch_size, *shape)

        received_ids.update(info["env_id"].tolist())

    assert received_ids == set(range(env_pool.num_envs)), "Did not receive all env_ids on reset."


def test_invalid_action_ends_only_its_own_episode(env_pool):
    """A masked-out action ends that env's episode with reward exactly 0 and auto-resets it to a
    playable state; a valid action sent alongside keeps its env running."""
    env_pool.async_reset()
    masks = {}
    for _ in range(env_pool.num_envs // env_pool.batch_size):
        _, _, _, info = env_pool.recv()
        for env_id, mask in zip(info["env_id"], info["action_mask"], strict=True):
            masks[int(env_id)] = mask

    # Chosen among all envs (not one async batch), so the choice does not depend on timing.
    invalid = next(e for e in sorted(masks) if not masks[e].all())
    valid = next(e for e in sorted(masks) if e != invalid and masks[e].any())
    actions = np.array([np.flatnonzero(~masks[invalid])[0], np.flatnonzero(masks[valid])[0]], np.int32)
    env_pool.send(actions, np.array([invalid, valid], np.int32))  # exactly batch_size in flight
    _, reward, done, info = env_pool.recv()
    row = {int(e): i for i, e in enumerate(info["env_id"])}

    assert done[row[invalid]] and reward[row[invalid]] == 0.0
    assert not done[row[valid]]
    assert info["action_mask"][row[invalid]].any(), "the auto-reset state is not playable"
