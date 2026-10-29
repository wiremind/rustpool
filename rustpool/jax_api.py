"""A jittable, optionally multi-device JAX bridge for ``rustpool.RlPool``.

The rustpool engine runs on native host threads, so it cannot live *inside* an XLA
computation the way a pure-JAX environment (Pgx, Brax) can. Instead this module crosses the
JAX↔host boundary with ``jax.experimental.io_callback``: every ``reset`` / ``step`` /
``snapshot`` / ``simulate_batch`` / ``release_batch`` is a host callback that drives the pool
and returns NumPy arrays, which JAX ingests as device arrays.

This is what makes the PPO and AlphaZero showcases "pure JAX that calls rustpool": the
training loop is a single ``jax.jit`` / ``jax.pmap`` + ``lax.scan`` program, and only the env
transitions hop to the host.

Single- and multi-GPU
---------------------
``JaxPool`` holds ``num_devices`` independent pools (one per device) and routes each
callback to the right pool with ``jax.lax.axis_index("device")``. The *same* object
therefore works:

* under ``jax.jit`` on one device — ``axis_index`` falls back to pool ``0``;
* under ``jax.pmap(..., axis_name="device")`` over N devices — replica *d* hits pool *d*.

So ``num_envs`` is the per-device env count and the global batch is ``num_devices * num_envs``.

Honesty note (vs EnvPool's ``env.xla()``): EnvPool exposes its step as a *pure* XLA custom
call needing no host round-trip; rustpool's ``io_callback`` synchronizes to the Python host
once per call. The trade-off buys an *external, non-differentiable* simulator (and the
``snapshot``/``simulate_batch`` search substrate) that cannot be a JAX pytree.

Example
-------
>>> import jax, rustpool.jax_api as rj
>>> pool = rj.JaxPool("Maze-v0", num_envs=8)            # single device
>>> out = jax.jit(pool.reset)(jax.random.key(0))
>>> out.obs["agent_pos"].shape, out.action_mask.shape
((8, 2), (8, 4))

A pool built elsewhere (e.g. by an external env crate through ``RlPool::new_with_options``,
which is a different Python class from ``rustpool.RlPool``) is wrapped with
``JaxPool.from_pools``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import rustpool
from rustpool._pool_check import check_sync_pool


class EnvOutput(NamedTuple):
    """One synchronous batch of environment outputs (leading axis = ``num_envs``).

    ``action_mask`` is always present and ``[num_envs, mask_width]`` with
    ``mask_width = pool.action_mask_width()``: ``num_actions`` when the env emits a flat mask;
    the factorization's ``pad_width`` when it emits the stage-0 factor mask instead
    (``BinPack-v0(factored=True)`` by default; the later stages come from ``query_mask``); ``0``
    when it emits none (``BinPack-v0(factored=False, emit_flat_mask=False)``, ``CartPole-v1``):
    a ``[num_envs, 0]`` array, so code that masks logits must treat a zero-width mask as
    "every action valid".
    """

    obs: dict[str, jax.Array]
    reward: jax.Array  # float32[num_envs]
    done: jax.Array  # bool[num_envs]
    action_mask: jax.Array  # bool[num_envs, mask_width] (num_actions, pad_width or 0)
    env_id: jax.Array  # int32[num_envs]


_NP_DTYPE = {"float32": np.float32, "f32": np.float32, "int32": np.int32, "i32": np.int32,
             "bool": np.bool_, "boolean": np.bool_}


def _bound_axes() -> dict[str, int] | None:
    """Named axes bound while tracing (``{}`` under plain ``jit``), or ``None`` when this JAX
    does not expose them (then routing falls back to ``axis_index`` alone)."""
    try:
        from jax._src import core

        return dict(core.get_axis_env().axis_sizes)
    except Exception:  # noqa: BLE001 - a private API: any failure means "unknown"
        return None


class JaxPool:
    """A synchronous (``batch_size == num_envs``), device-sharded JAX view of rustpool.

    The constructor builds one pool per device with ``rustpool.make``;
    ``from_pools`` wraps pools that already exist instead.

    Parameters
    ----------
    task_id, num_envs : the env id and the per-device number of environments.
    num_devices : how many independent host pools to create (one per JAX device). Default 1.
        Pass ``jax.local_device_count()`` to shard across all local GPUs for a ``pmap`` run.
    seed : base seed; pool *d* is offset by ``d * 10_000`` so devices do not share streams.
    **kwargs : forwarded to ``rustpool.make`` (env config, ``num_threads``, ``dispatch``,
        ``autoreset``).
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int,
        *,
        num_devices: int = 1,
        seed: int = 0,
        **kwargs: Any,
    ):
        # Synchronous regime: recv returns all envs each call, sorted by env_id.
        pools = [
            rustpool.make(
                task_id,
                num_envs=int(num_envs),
                batch_size=int(num_envs),
                seed=seed + d * 10_000,
                **kwargs,
            )
            for d in range(int(num_devices))
        ]
        self._init_from_pools(pools, task_id)

    @classmethod
    def from_pools(cls, pools: Any, task_id: str | None = None) -> JaxPool:
        """Wrap existing pools, one per device (a single pool is one device).

        Any object with rustpool's pool API works (duck-typed), so a pool an external env crate
        built with ``RlPool::new_with_options`` (a different Python class) qualifies. Every
        pool must be synchronous (``batch_size == num_envs``) with the same ``num_envs``.
        ``close`` closes them.
        """
        pools = list(pools) if isinstance(pools, (list, tuple)) else [pools]
        obj = cls.__new__(cls)
        obj._init_from_pools(pools, task_id)
        return obj

    def _init_from_pools(self, pools: list[Any], task_id: str | None) -> None:
        if not pools:
            raise ValueError("JaxPool needs at least one pool")
        for pool in pools:
            check_sync_pool(pool, "JaxPool")
        sizes = {int(pool.num_envs) for pool in pools}
        if len(sizes) != 1:
            raise ValueError(f"every pool needs the same num_envs, got {sorted(sizes)}")
        self.task_id = task_id
        self.num_envs = sizes.pop()
        self.num_devices = len(pools)
        self._pools = pools
        self._env_ids = np.arange(self.num_envs, dtype=np.int32)

        action_spec = self._pools[0].action_spec()
        self.num_actions = int(action_spec[0])
        # Real width of the emitted mask rows (flat, stage-0 pad_width, or 0).
        self.mask_width = int(self._pools[0].action_mask_width())
        self._obs_spec: dict[str, tuple[list[int], str]] = self._pools[0].observation_spec()

        self.factorization = self._pools[0].action_factorization()
        self._pad_width = int(self.factorization["pad_width"]) if self.factorization else 0

        self._out_template = self._build_template(include_state_id=False, batch=self.num_envs)

    # ---- device routing (works under jit -> 0, and pmap -> axis index) -----------------
    def _device_id(self) -> jax.Array:
        axes = _bound_axes()
        if axes is not None and len(self._pools) > 1:
            if "device" not in axes and axes:
                # pmap over another axis name: every replica would silently use pool 0.
                raise ValueError(
                    f"JaxPool holds {len(self._pools)} pools, one per device: map it with "
                    f"jax.pmap(..., axis_name='device'), not over {sorted(axes)}"
                )
            if "device" in axes and axes["device"] != len(self._pools):
                raise ValueError(
                    f"axis 'device' has size {axes['device']} but JaxPool holds "
                    f"{len(self._pools)} pools (num_devices)"
                )
        try:
            return jax.lax.axis_index("device")
        except (NameError, ValueError):
            return jnp.asarray(0, dtype=jnp.int32)

    # ---- result templates for io_callback ---------------------------------------------
    def _build_template(self, include_state_id: bool, batch: int):
        B = batch
        obs = {
            key: jax.ShapeDtypeStruct((B, *shape), jnp.dtype(_NP_DTYPE[dtype]))
            for key, (shape, dtype) in self._obs_spec.items()
        }
        out = EnvOutput(
            obs=obs,
            reward=jax.ShapeDtypeStruct((B,), jnp.float32),
            done=jax.ShapeDtypeStruct((B,), jnp.bool_),
            action_mask=jax.ShapeDtypeStruct((B, self.mask_width), jnp.bool_),
            env_id=jax.ShapeDtypeStruct((B,), jnp.int32),
        )
        if include_state_id:
            return (out, jax.ShapeDtypeStruct((B,), jnp.int32))
        return out

    # ---- host packing -----------------------------------------------------------------
    def _pack(self, obs, reward, done, info, batch: int | None = None) -> EnvOutput:
        B = self.num_envs if batch is None else batch
        mask = info.get("action_mask")
        if mask is None:
            mask = np.ones((B, self.mask_width), dtype=np.bool_)
        else:
            mask = np.asarray(mask, dtype=np.bool_)
        obs_out = {
            key: np.asarray(obs[key], dtype=_NP_DTYPE[self._obs_spec[key][1]])
            for key in self._obs_spec
        }
        return EnvOutput(
            obs=obs_out,
            reward=np.asarray(reward, dtype=np.float32),
            done=np.asarray(done, dtype=np.bool_),
            action_mask=mask,
            env_id=np.asarray(info.get("env_id", self._env_ids), dtype=np.int32),
        )

    # ---- host callbacks (device_id picks the per-device pool) --------------------------
    def _host_reset(self, device_id, key) -> EnvOutput:
        pool = self._pools[int(device_id)]
        key = np.asarray(key)
        base_seed = int(key.reshape(-1)[-1]) & 0x7FFFFFFF
        seeds = (base_seed + np.arange(self.num_envs)).astype(np.int64)
        pool.async_reset(seeds)
        return self._pack(*pool.recv())

    def _host_step(self, device_id, actions) -> EnvOutput:
        pool = self._pools[int(device_id)]
        pool.send(np.asarray(actions, dtype=np.int32), self._env_ids)
        return self._pack(*pool.recv())

    def _host_snapshot(self, device_id, env_ids) -> np.ndarray:
        pool = self._pools[int(device_id)]
        ids = pool.snapshot(np.asarray(env_ids, dtype=np.int32))
        return np.asarray(ids, dtype=np.int32)

    def _host_simulate(self, device_id, state_ids, actions):
        pool = self._pools[int(device_id)]
        obs, reward, done, info = pool.simulate_batch(
            np.asarray(state_ids, dtype=np.int32), np.asarray(actions, dtype=np.int32)
        )
        out = self._pack(obs, reward, done, info, batch=len(state_ids))
        new_ids = np.asarray(info["state_id"], dtype=np.int32)
        return out, new_ids

    def _host_release(self, device_id, state_ids) -> np.ndarray:
        pool = self._pools[int(device_id)]
        pool.release_batch(np.asarray(state_ids, dtype=np.int32))
        return np.asarray(0, dtype=np.int32)

    def _host_query_mask(self, device_id, prefixes, env_ids) -> np.ndarray:
        pool = self._pools[int(device_id)]
        m = pool.query_mask(
            np.asarray(prefixes, dtype=np.int32), np.asarray(env_ids, dtype=np.int32)
        )
        return np.asarray(m, dtype=np.bool_)

    def _host_query_state_mask(self, device_id, state_ids, prefixes) -> np.ndarray:
        pool = self._pools[int(device_id)]
        m = pool.query_state_mask(
            np.asarray(state_ids, dtype=np.int32), np.asarray(prefixes, dtype=np.int32)
        )
        return np.asarray(m, dtype=np.bool_)

    def _require_factorization(self, name: str) -> None:
        # At trace time: under jit the callback's exception would surface without its type.
        if not self._pad_width:
            raise ValueError(
                f"{name}: the pool's env has no action factorization (its conditional masks have "
                "width 0); it needs a factored action space"
            )

    # ---- jittable / pmappable JAX API -------------------------------------------------
    def reset(self, key: jax.Array) -> EnvOutput:
        """Reset every env (seeded from ``key``) and return the initial ``EnvOutput``."""
        return jax.experimental.io_callback(
            self._host_reset, self._out_template, self._device_id(), key, ordered=True
        )

    def step(self, actions: jax.Array) -> EnvOutput:
        """Step every env with ``actions`` (``int32[num_envs]``); auto-resets on terminal."""
        return jax.experimental.io_callback(
            self._host_step, self._out_template, self._device_id(), actions, ordered=True
        )

    def snapshot(self, env_ids: jax.Array) -> jax.Array:
        """Clone the given envs into the registry; returns ``int32[len(env_ids)]`` handles.

        Length-agnostic: you may snapshot any subset of the pool's envs.
        """
        struct = jax.ShapeDtypeStruct((env_ids.shape[0],), jnp.int32)
        return jax.experimental.io_callback(
            self._host_snapshot, struct, self._device_id(), env_ids, ordered=True,
        )

    def simulate_batch(
        self, state_ids: jax.Array, actions: jax.Array
    ) -> tuple[EnvOutput, jax.Array]:
        """Fork-and-step stored states without mutating them (the MCTS substrate).

        Returns ``(EnvOutput, new_state_ids)`` where ``new_state_ids`` are the children's
        handles (free them with ``release_batch``).

        Uses ``ordered=False``: a fork-and-step is a pure function of ``(state_ids, actions)``
        (the registry clones-on-read, results are deterministic), so it composes inside the
        control flow of a search library such as ``mctx`` where ordered effects are not allowed.

        Length-agnostic: ``state_ids`` and ``actions`` may be any (equal) length.
        """
        template = self._build_template(include_state_id=True, batch=state_ids.shape[0])
        return jax.experimental.io_callback(
            self._host_simulate, template, self._device_id(), state_ids, actions,
            ordered=False,
        )

    def release_batch(self, state_ids: jax.Array) -> jax.Array:
        """Free registry entries for ``state_ids`` (``-1`` / unknown entries are ignored)."""
        return jax.experimental.io_callback(
            self._host_release, jax.ShapeDtypeStruct((), jnp.int32), self._device_id(),
            state_ids, ordered=False,
        )

    def action_factorization(self):
        """The env's factored-action spec (dict) or ``None`` for a flat action space."""
        return self.factorization

    def query_mask(self, prefixes: jax.Array, env_ids: jax.Array) -> jax.Array:
        """Non-advancing per-factor conditional masks (the factored-action transport).

        ``prefixes`` is ``int32[B, num_factors-1]`` (``-1`` = unset); returns a
        ``bool[B, pad_width]`` mask over the next factor for each env. Non-advancing and pure
        in the frozen state, so ``ordered=False`` — it composes inside the rollout's autoregressive
        sampling loop (the chosen action, which feeds ``step``, depends on it, so order is kept
        by data flow).
        """
        self._require_factorization("query_mask")
        struct = jax.ShapeDtypeStruct((env_ids.shape[0], self._pad_width), jnp.bool_)
        return jax.experimental.io_callback(
            self._host_query_mask, struct, self._device_id(), prefixes, env_ids, ordered=False
        )

    def query_state_mask(self, state_ids: jax.Array, prefixes: jax.Array) -> jax.Array:
        """``query_mask`` on **stored states** (``snapshot`` / ``simulate_batch`` handles)
        instead of the live envs: ``bool[len(state_ids), pad_width]``. Lets a tree search read
        per-factor masks at any node without the flat mask. Pure, so ``ordered=False``.
        """
        self._require_factorization("query_state_mask")
        struct = jax.ShapeDtypeStruct((state_ids.shape[0], self._pad_width), jnp.bool_)
        return jax.experimental.io_callback(
            self._host_query_state_mask, struct, self._device_id(), state_ids, prefixes,
            ordered=False,
        )

    def close(self) -> None:
        """Close every per-device pool (stops their workers, frees their stored states)."""
        for pool in self._pools:
            pool.close()

    def __len__(self) -> int:
        return self.num_envs
