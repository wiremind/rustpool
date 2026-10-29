import numpy as np
import pytest

import rustpool


def test_mismatched_send_lengths_raises_error(env_pool):
    """send() should reject mismatched action/env_id lengths for every task."""
    env_pool.async_reset()
    # Use any valid action index to avoid unrelated failures
    action_index = np.array([0], dtype=np.int32)
    env_ids = np.arange(2, dtype=np.int32)

    with pytest.raises(ValueError, match="actions length.*must match env_ids length"):
        env_pool.send(action_index, env_ids)


def test_out_of_bounds_env_id_raises_error(env_pool):
    """send() should detect env_ids that exceed the configured pool size."""
    env_pool.async_reset()
    invalid_env_id = np.array([env_pool.num_envs], dtype=np.int32)
    action_index = np.array([0], dtype=np.int32)

    with pytest.raises(IndexError, match="env_id.*out of bounds"):
        env_pool.send(action_index, invalid_env_id)


@pytest.mark.parametrize("batch_size, expected", [(None, 4), (0, 4), (-1, 4), (5, 4), (2, 2)])
def test_make_clamps_batch_size_to_num_envs(batch_size, expected):
    """make() turns a missing, non-positive or oversized batch_size into num_envs (EnvPool's
    behaviour); the constructor itself rejects them (test_config_validation.py)."""
    kwargs = {} if batch_size is None else {"batch_size": batch_size}
    with rustpool.make("Maze-v0", num_envs=4, **kwargs) as pool:
        assert pool.batch_size == expected


def test_make_rejects_an_unknown_task_and_the_constructor_an_oversized_batch():
    with pytest.raises(ValueError, match="Unknown task_id"):
        rustpool.make("Nope-v0")
    with pytest.raises(ValueError, match="batch_size"):
        rustpool.RlPool(num_envs=2, batch_size=3, task_id="Maze-v0")


def test_binpack_with_zero_items_is_done_at_once_with_an_empty_mask():
    with rustpool.make("BinPack-v0", num_envs=1, batch_size=1, max_items=0) as pool:
        pool.async_reset()
        _, _, done, info = pool.recv()
        assert done[0]
        assert info["action_mask"].shape == (1, 0)
