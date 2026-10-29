"""rustpool — a pure-Rust parallel RL environment-execution engine for Python.

rustpool runs many reinforcement-learning environments on native Rust worker threads
(true parallelism, no GIL) behind an asynchronous, EnvPool-style API, and adds two things
EnvPool does not have:

* a **state-cloning substrate** (``snapshot`` / ``simulate_batch`` / ``release_batch``)
  that lets a batched MCTS / AlphaZero search fork arbitrary states, and
* **factored / autoregressive action masks** (``query_mask`` / ``action_factorization``).

Quick start
-----------
>>> import numpy as np, rustpool
>>> rustpool.list_envs()
['BinPack-v0', 'CartPole-v1', 'Maze-v0']
>>> pool = rustpool.make("Maze-v0", num_envs=8, batch_size=8)
>>> pool.async_reset()
>>> obs, reward, done, info = pool.recv()
>>> pool.send(np.zeros(len(info["env_id"]), dtype=np.int32), info["env_id"])
>>> pool.close()

Optional integrations are imported lazily, so ``import rustpool`` stays lightweight and
does **not** import JAX or PyTorch:

* ``rustpool.jax_api`` — a jittable JAX bridge (``JaxPool``) built on ``io_callback``.
* ``rustpool.torch_api`` — zero-copy NumPy↔PyTorch helpers and a batched env adapter.
* ``rustpool.envpool_api.gymnasium`` — a gymnasium ``VectorEnv`` wrapper.
"""

import warnings
from typing import Any

from rustpool._rustpool import RlPool

from .envpool_api import list_envs, make

try:  # populated from the installed distribution metadata when available
    from importlib.metadata import version

    __version__ = version("rustpool")
except Exception:  # pragma: no cover - source checkout without dist-info
    from rustpool._rustpool import __version__  # type: ignore[no-redef]

__all__ = ["make", "list_envs", "RlPool", "__version__"]


def __getattr__(name: str) -> Any:
    # ``MyEnvPool`` is the pool's name before 0.2.0, kept as a deprecated alias.
    if name == "MyEnvPool":
        warnings.warn("rustpool.MyEnvPool is renamed rustpool.RlPool", DeprecationWarning,
                      stacklevel=2)
        return RlPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
