"""Zero-copy PyTorch interop for ``rustpool.RlPool``.

rustpool's ``recv`` builds *fresh, writable* NumPy arrays each call, so wrapping them with
``torch.from_numpy`` is genuinely zero-copy (no extra allocation, no warning) — preferable to
DLPack here, which only adds capsule-lifetime hazards for a CPU producer.

The one sharp edge is the **int32 contract**: ``send`` / ``snapshot`` / ``simulate_batch``
take ``int32`` arrays, but torch integer tensors default to int64. ``TorchEnvPool``
casts actions to int32 at the boundary so you never hit an opaque PyO3 ``TypeError``.

Example
-------
>>> from rustpool.torch_api import TorchEnvPool
>>> pool = TorchEnvPool("Maze-v0", num_envs=8, device="cpu")
>>> obs, reward, done, info = pool.reset()
>>> obs["agent_pos"].shape, info["action_mask"].dtype
(torch.Size([8, 2]), torch.bool)
>>> obs, reward, done, info = pool.step(torch.zeros(8, dtype=torch.int64))  # cast for you

A pool built elsewhere (e.g. by an external env crate, a different Python class from
``rustpool.RlPool``) is wrapped with ``TorchEnvPool.from_pool``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import rustpool
from rustpool._pool_check import check_sync_pool

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "rustpool.torch_api requires PyTorch. Install it with: pip install 'rustpool[torch]'"
    ) from exc


def to_torch(value: Any, device: Any = None) -> Any:
    """Recursively convert NumPy arrays (or a dict of them) to torch tensors (zero-copy)."""
    if isinstance(value, dict):
        return {k: to_torch(v, device) for k, v in value.items()}
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    return tensor.to(device) if device is not None else tensor


def _actions_to_numpy(actions: Any) -> np.ndarray:
    if isinstance(actions, torch.Tensor):
        return actions.detach().to(torch.int32).cpu().numpy()
    return np.asarray(actions, dtype=np.int32)


class TorchEnvPool:
    """A synchronous (``batch_size == num_envs``) PyTorch view of a rustpool pool.

    ``reset`` / ``step`` return ``(obs, reward, done, info)`` where ``obs`` is a dict of
    tensors, ``reward``/``done`` are tensors, and ``info`` carries ``env_id`` and (if the env
    exposes one) ``action_mask`` as tensors. Everything is moved to ``device`` if given.
    """

    def __init__(self, task_id: str, num_envs: int, *, device: Any = None, seed: int = 0,
                 **kwargs: Any):
        pool = rustpool.make(
            task_id, num_envs=int(num_envs), batch_size=int(num_envs), seed=seed, **kwargs
        )
        self._init_from_pool(pool, task_id, device)

    @classmethod
    def from_pool(cls, pool: Any, *, device: Any = None,
                  task_id: str | None = None) -> TorchEnvPool:
        """Wrap an existing pool. Any object with rustpool's pool API works (duck-typed), so a
        pool an external env crate built with ``RlPool::new_with_options`` (a different
        Python class) qualifies. It must be synchronous (``batch_size == num_envs``). ``close``
        closes it."""
        obj = cls.__new__(cls)
        obj._init_from_pool(pool, task_id, device)
        return obj

    def _init_from_pool(self, pool: Any, task_id: str | None, device: Any) -> None:
        check_sync_pool(pool, "TorchEnvPool")
        self.task_id = task_id
        self.num_envs = int(pool.num_envs)
        self.device = torch.device(device) if device is not None else None
        self._pool = pool
        self._env_ids = np.arange(self.num_envs, dtype=np.int32)
        self.num_actions = int(self._pool.action_spec()[0])
        self.observation_spec = self._pool.observation_spec()

    def _wrap(self, obs, reward, done, info):
        return (
            to_torch(obs, self.device),
            to_torch(reward, self.device),
            to_torch(done, self.device),
            {k: to_torch(v, self.device) for k, v in info.items()},
        )

    def reset(self, seeds: Any = None):
        """Reset all envs (optionally with an ``int64[num_envs]`` seed array)."""
        if seeds is not None:
            seeds = (
                seeds.detach().cpu().numpy() if isinstance(seeds, torch.Tensor) else seeds
            )
            seeds = np.asarray(seeds, dtype=np.int64)
        self._pool.async_reset(seeds)
        return self._wrap(*self._pool.recv())

    def step(self, actions: Any):
        """Step all envs with ``actions`` (any int tensor/array; cast to int32 for you)."""
        self._pool.send(_actions_to_numpy(actions), self._env_ids)
        return self._wrap(*self._pool.recv())

    # --- state-cloning substrate for tree search (AlphaZero / MCTS) -------------------
    def snapshot(self, env_ids: Any = None):
        """Clone live envs into the registry; returns an ``int32`` tensor of state-id handles.

        ``env_ids=None`` snapshots all envs. The handles are the roots a search forks from
        with ``simulate_batch``; free them with ``release_batch``.
        """
        ids = self._env_ids if env_ids is None else _actions_to_numpy(env_ids)
        state_ids = np.asarray(self._pool.snapshot(ids), dtype=np.int32)
        return to_torch(state_ids, self.device)

    def simulate_batch(self, state_ids: Any, actions: Any):
        """Fork-and-step stored states without mutating them (the MCTS substrate).

        Returns ``(obs, reward, done, info)`` like ``step``, with the children's new
        handles in ``info["state_id"]`` (an ``int32`` tensor). ``state_ids``/``actions`` are
        equal-length int tensors/arrays (cast to int32 for you).
        """
        obs, reward, done, info = self._pool.simulate_batch(
            _actions_to_numpy(state_ids), _actions_to_numpy(actions)
        )
        return self._wrap(obs, reward, done, info)

    def release_batch(self, state_ids: Any) -> None:
        """Free registry entries for ``state_ids`` (``-1`` / unknown entries are ignored)."""
        self._pool.release_batch(_actions_to_numpy(state_ids))

    # --- factored / autoregressive action masks --------------------------------------
    def action_factorization(self):
        """The env's factored-action spec (dict) or ``None`` for a flat action space."""
        return self._pool.action_factorization()

    def query_mask(self, prefixes: Any, env_ids: Any = None):
        """Non-advancing per-factor conditional mask, as a ``bool`` tensor ``[B, pad_width]``.

        ``prefixes`` is ``int32[B, num_factors-1]`` (``-1`` = unset). Returns the mask over the
        next factor for each env. An env with no action factorization raises ``ValueError``.
        """
        ids = self._env_ids if env_ids is None else _actions_to_numpy(env_ids)
        return to_torch(self._pool.query_mask(_actions_to_numpy(prefixes), ids), self.device)

    def close(self) -> None:
        """Close the underlying pool (stops its workers, frees its stored states)."""
        self._pool.close()

    def __len__(self) -> int:
        return self.num_envs
