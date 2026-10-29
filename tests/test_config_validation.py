"""Out-of-range pool and env options raise ValueError at construction, never a worker panic."""

import numpy as np
import pytest

import rustpool
from rustpool import _rustpool


@pytest.mark.parametrize(
    "task_id, kwargs",
    [
        ("Maze-v0", {"width": 0}),
        ("Maze-v0", {"height": -3}),
        ("Maze-v0", {"max_episode_steps": 0}),
        ("BinPack-v0", {"max_ems": 0}),
        ("BinPack-v0", {"prob_split_one_item": 2.0}),
        ("BinPack-v0", {"split_eps": -0.1}),
        ("CartPole-v1", {"max_episode_steps": 0}),
    ],
)
def test_out_of_range_env_option_raises(task_id, kwargs):
    with pytest.raises(ValueError):
        rustpool.make(task_id, num_envs=2, **kwargs)


@pytest.mark.parametrize("num_envs, batch_size", [(0, 0), (4, 0)])
def test_empty_pool_raises(num_envs, batch_size):
    with pytest.raises(ValueError):
        _rustpool.RlPool(num_envs=num_envs, batch_size=batch_size, task_id="Maze-v0")


def test_smallest_valid_maze_steps():
    with rustpool.make("Maze-v0", num_envs=2, width=1, height=1) as pool:
        pool.async_reset()
        _, _, _, info = pool.recv()
        pool.send(np.zeros(2, np.int32), info["env_id"])
        pool.recv()


@pytest.mark.parametrize("task_id", rustpool.list_envs())
def test_unknown_env_option_raises_type_error(task_id):
    with pytest.raises(TypeError, match=rf"{task_id}: unknown field `not_an_option`"):
        rustpool.make(task_id, num_envs=2, not_an_option=1)


def test_mistyped_env_option_raises_type_error():
    with pytest.raises(TypeError, match="Maze-v0"):
        rustpool.make("Maze-v0", num_envs=2, width="ten")
