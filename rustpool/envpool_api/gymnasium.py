"""A ``gymnasium.vector.VectorEnv`` over a rustpool pool (needs ``pip install rustpool[gymnasium]``)."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode
    from gymnasium.vector.utils import batch_space
except ImportError as err:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "rustpool.envpool_api.gymnasium needs gymnasium>=1.1: pip install 'rustpool[gymnasium]'"
    ) from err

from . import make


def _single_space(shape: list[int], dtype: str) -> gym.spaces.Space:
    shape = tuple(shape)
    if dtype == "bool":
        return gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.bool_)
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        return gym.spaces.Box(low=info.min, high=info.max, shape=shape, dtype=np_dtype)
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np_dtype)


class RustPoolGymnasiumWrapper(gym.vector.VectorEnv):
    """``num_envs`` environments of ``task_id`` as a synchronous gymnasium vector env.

    The pool is built with ``rustpool.make(task_id, num_envs=..., **kwargs)``, so every pool
    option (``seed``, ``num_threads``, ``dispatch``) and every environment option (``width``,
    ``max_items``, ...) is accepted. The pool always runs with ``batch_size == num_envs`` and
    ``autoreset="next_step"``, which is gymnasium's ``AutoresetMode.NEXT_STEP``: the step that
    ends an episode returns its terminal observation, and the next step resets that
    environment, ignores its action and returns reward 0.

    The engine reports one ``done`` flag per step, so an episode cut by its step limit comes
    back as ``terminated=True``; ``truncated`` is always ``False``. ``info`` carries the
    pool's ``env_id`` and ``action_mask`` arrays.
    """

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, task_id: str, num_envs: int = 4, **kwargs: Any):
        batch_size = kwargs.pop("batch_size", num_envs)
        if batch_size != num_envs:
            raise ValueError("the gymnasium wrapper is synchronous: batch_size must equal num_envs")
        autoreset = kwargs.pop("autoreset", "next_step")
        if autoreset != "next_step":
            raise ValueError("the gymnasium wrapper uses autoreset='next_step' (gymnasium's NEXT_STEP)")
        self._pool = make(task_id, num_envs=num_envs, batch_size=num_envs,
                          autoreset="next_step", **kwargs)
        self._env_ids = np.arange(num_envs, dtype=np.int32)

        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Dict(
            {key: _single_space(shape, dtype)
             for key, (shape, dtype) in self._pool.observation_spec().items()}
        )
        self.single_action_space = gym.spaces.Discrete(self._pool.action_spec()[0])
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        """Reset every environment. An integer seed gives env ``i`` the seed ``seed + i``; a
        sequence of ``num_envs`` seeds gives env ``i`` the seed ``seed[i]``."""
        if seed is None:
            seeds = None
        elif isinstance(seed, (int, np.integer)):
            seeds = np.arange(seed, seed + self.num_envs, dtype=np.int64)
        else:
            seeds = np.asarray(seed, dtype=np.int64)
        self._pool.async_reset(seeds)
        obs, _, _, info = self._pool.recv()
        return obs, info

    def step(self, actions):
        """Step every environment; returns ``(obs, reward, terminated, truncated, info)``."""
        self._pool.send(np.asarray(actions, dtype=np.int32), self._env_ids)
        obs, reward, done, info = self._pool.recv()
        return obs, reward, done, np.zeros_like(done), info

    def close_extras(self, **kwargs: Any) -> None:
        """Close the underlying pool (called by gymnasium's ``close()``)."""
        self._pool.close()
