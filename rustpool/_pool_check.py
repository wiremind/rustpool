"""Duck-typed check shared by the JAX and PyTorch bridges (imports neither)."""

from __future__ import annotations

from typing import Any

_POOL_METHODS = ("async_reset", "send", "recv", "action_spec", "observation_spec", "close")


def check_sync_pool(pool: Any, who: str) -> None:
    """Duck-typed check that ``pool`` has rustpool's pool API and is synchronous."""
    missing = [m for m in _POOL_METHODS if not callable(getattr(pool, m, None))]
    if missing or not hasattr(pool, "num_envs") or not hasattr(pool, "batch_size"):
        raise TypeError(f"{who} needs a rustpool pool (num_envs, batch_size, "
                        f"{', '.join(_POOL_METHODS)}); {type(pool).__name__} lacks "
                        f"{missing or ['num_envs/batch_size']}")
    if int(pool.batch_size) != int(pool.num_envs):
        raise ValueError(f"{who} is synchronous: the pool needs batch_size == num_envs, got "
                         f"batch_size={pool.batch_size}, num_envs={pool.num_envs}")
