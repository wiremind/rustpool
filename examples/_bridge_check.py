"""Tiny self-check for rustpool.jax_api.JaxPool (used by tests/test_examples_smoke.py).

Exercises reset/step under jit and the snapshot/simulate_batch/release MCTS substrate,
asserts branching determinism, and prints ``BRIDGE_OK`` on success.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

import rustpool.jax_api as rj


def main() -> None:
    pool = rj.JaxPool("Maze-v0", num_envs=8, seed=0)

    out = jax.jit(pool.reset)(jax.random.key(0))
    assert out.obs["agent_pos"].shape == (8, 2)
    assert out.action_mask.shape == (8, 4)

    def rollout(key):
        o0 = pool.reset(key)

        def body(carry, _):
            o, k = carry
            k, sk = jax.random.split(k)
            logits = jnp.where(o.action_mask, 0.0, -1e9)
            a = jax.random.categorical(sk, logits).astype(jnp.int32)
            nxt = pool.step(a)
            return (nxt, k), nxt.reward

        (_, _), rews = jax.lax.scan(body, (o0, key), None, length=10)
        return rews

    rews = jax.jit(rollout)(jax.random.key(1))
    assert rews.shape == (10, 8)

    pool.reset(jax.random.key(2))
    sids = jax.jit(pool.snapshot)(jnp.arange(8, dtype=jnp.int32))
    o1, c1 = jax.jit(pool.simulate_batch)(sids, jnp.zeros(8, jnp.int32))
    o2, c2 = jax.jit(pool.simulate_batch)(sids, jnp.ones(8, jnp.int32))
    o1b, _ = jax.jit(pool.simulate_batch)(sids, jnp.zeros(8, jnp.int32))
    assert (np.asarray(c1) != np.asarray(sids)).all(), "children must be fresh ids"
    assert (np.asarray(o1.obs["agent_pos"]) != np.asarray(o2.obs["agent_pos"])).any()
    assert (np.asarray(o1.obs["agent_pos"]) == np.asarray(o1b.obs["agent_pos"])).all()
    pool.release_batch(sids)
    pool.release_batch(c1)
    pool.release_batch(c2)
    pool.close()
    print("BRIDGE_OK")


if __name__ == "__main__":
    main()
