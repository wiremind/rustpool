# The JAX bridge

`rustpool.jax_api.JaxPool` makes the pool callable from inside `jax.jit` and `jax.pmap`, so a
whole training loop (rollout, losses, updates) can be one compiled JAX program. The environments
still run on host threads: the bridge crosses to the host with `jax.experimental.io_callback`.
Each call runs the pool synchronously (`batch_size == num_envs`) and returns device arrays.

```python
import jax
import jax.numpy as jnp
from rustpool.jax_api import JaxPool

pool = JaxPool("Maze-v0", num_envs=8)
out = jax.jit(pool.reset)(jax.random.key(0))
print(out.obs["agent_pos"].shape, out.action_mask.shape)   # (8, 2) (8, 4)

@jax.jit
def rollout(out, key):
    def body(carry, k):
        out = carry
        logits = jnp.where(out.action_mask, 0.0, -jnp.inf)    # uniform over valid actions
        actions = jax.random.categorical(k, logits).astype(jnp.int32)
        out = pool.step(actions)
        return out, out.reward
    return jax.lax.scan(body, out, jax.random.split(key, 16))

out, rewards = rollout(out, jax.random.key(1))
print(rewards.shape)                                        # (16, 8)
```

`reset(key)` reseeds the environments from the key: env `i` gets `base + i`, where `base` is
derived from `key`, so the same key gives the same episodes. `step(actions)` takes
`int32[num_envs]`. Both return an `EnvOutput(obs, reward, done, action_mask, env_id)`.
`action_mask` is always present: `bool[num_envs, pool.mask_width]`. An environment that emits
no mask (`CartPole-v1`, or `BinPack-v0` with `factored=False, emit_flat_mask=False`) gives a
zero-width `[num_envs, 0]` array: treat that as "every action valid".

The constructor's extra arguments go to `rustpool.make`, so every pool option works here too:
`JaxPool("Maze-v0", 8, num_threads=2, dispatch="per_worker", autoreset="next_step")`.

## One host trip per call

EnvPool's `env.xla()` exposes its step as a pure XLA operation, with no host round trip.
rustpool's bridge synchronises with the Python host on every call. For a cheap environment that
is the dominant cost, and the reason to prefer EnvPool or a pure-JAX environment when one exists.
In exchange you get a simulator that is not a JAX program, and the search primitives below.

## Ordered and unordered calls

`io_callback` either keeps program order (`ordered=True`) or lets XLA schedule the call freely
(`ordered=False`, required inside control flow such as `mctx`'s search loop).

| Call | Ordered | Why |
| --- | --- | --- |
| `reset`, `step` | yes | they advance the live environments: order is the semantics |
| `snapshot` | yes | it must see the state after the steps that precede it |
| `simulate_batch` | no | a pure function of `(state_ids, actions)`: it never changes a stored state |
| `query_mask`, `query_state_mask` | no | read-only; the chosen action, which feeds `step`, depends on the mask, so data flow keeps the order |
| `release_batch` | no | so a search can free nodes inside its loop |

The catch with unordered calls: inside one jitted function, XLA may run a `release_batch` before
a `simulate_batch` or `query_state_mask` of the same ids when nothing in the data connects them,
and the read then fails with `invalid state_id`. Make the release depend on the read's output,
or release in a later call. Do not `vmap` a `JaxPool`; use one pool per device under `pmap`.

## Several devices

`JaxPool(..., num_devices=n)` holds one independent pool per device and routes each call with
`jax.lax.axis_index("device")`. The same object works under `jit` (pool 0) and under
`pmap(..., axis_name="device")` (replica `d` uses pool `d`). `num_envs` is per device. Under
`pmap` over another axis name, or over a `"device"` axis whose size is not `num_devices`, every
call raises `ValueError` when traced, instead of routing every replica to pool 0.

```python
n = jax.local_device_count()
pool_n = JaxPool("Maze-v0", num_envs=8, num_devices=n)
outs = jax.pmap(pool_n.reset, axis_name="device")(jax.random.split(jax.random.key(0), n))
print(outs.obs["agent_pos"].shape)                          # (n, 8, 2)
pool_n.close()
```

On a CPU-only machine you can rehearse this with
`XLA_FLAGS=--xla_force_host_platform_device_count=2`.

## Wrapping a pool you already have

`JaxPool.from_pools(pools)` wraps existing pools (one per device) instead of building them. It
accepts any object with the pool API, which matters for pools built by
[your own crate](../guides/own-crate.md): they are a different Python class from
`rustpool.RlPool`. Each pool must be synchronous (`batch_size == num_envs`).

```python
import rustpool

base = rustpool.make("CartPole-v1", num_envs=8, batch_size=8, seed=4, autoreset="next_step")
wrapped = JaxPool.from_pools(base)
wrapped.reset(jax.random.key(0))
out = wrapped.step(jnp.ones(8, jnp.int32))
print(out.obs["obs"].shape)                                 # (8, 4)
wrapped.close()
pool.close()
```
