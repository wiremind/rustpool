"""The gymnasium vector-env wrapper: spaces, env options, and NEXT_STEP auto-reset."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.vector import AutoresetMode  # noqa: E402

import rustpool  # noqa: E402
from rustpool.envpool_api.gymnasium import RustPoolGymnasiumWrapper  # noqa: E402


@pytest.mark.parametrize("task_id", rustpool.list_envs())
def test_spaces_contain_what_the_env_returns(task_id):
    env = RustPoolGymnasiumWrapper(task_id, num_envs=3, seed=0)
    try:
        assert isinstance(env, gym.vector.VectorEnv)
        assert env.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
        obs, info = env.reset(seed=1)
        assert env.observation_space.contains(obs)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
        assert reward.shape == terminated.shape == truncated.shape == (3,)
        assert info["action_mask"].shape[0] == 3  # width 0 for an env without a mask
    finally:
        env.close()


def test_env_options_reach_the_env():
    env = RustPoolGymnasiumWrapper("Maze-v0", num_envs=4, width=3, height=3, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert np.all(obs["agent_pos"] < 3)
    finally:
        env.close()


def test_next_step_autoreset_matches_gymnasium():
    env = RustPoolGymnasiumWrapper("CartPole-v1", num_envs=2, seed=0)
    try:
        env.reset(seed=0)
        for _ in range(1000):
            _, _, terminated, _, _ = env.step(np.zeros(2, np.int64))
            if terminated.any():
                break
        assert terminated.any()
        # The call after the terminal step resets those envs: reward 0, not done.
        _, reward, terminated_next, _, _ = env.step(np.zeros(2, np.int64))
        assert np.all(reward[terminated] == 0.0)
        assert not np.any(terminated_next[terminated])
    finally:
        env.close()


def test_rejects_async_and_same_step_options():
    with pytest.raises(ValueError):
        RustPoolGymnasiumWrapper("Maze-v0", num_envs=4, batch_size=2)
    with pytest.raises(ValueError):
        RustPoolGymnasiumWrapper("Maze-v0", num_envs=4, autoreset="same_step")
