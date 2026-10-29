# Quickstart

## Install

```bash
pip install rustpool                 # the engine: just numpy
pip install "rustpool[gymnasium]"    # + the gymnasium vector-env wrapper
pip install "rustpool[jax]"          # + the JaxPool bridge
pip install "rustpool[torch]"        # + the PyTorch view
pip install "rustpool[examples]"     # + flax, optax, mctx for the PPO / AlphaZero examples
```

To build from a checkout instead, see [Write an environment inside rustpool](guides/env-in-rustpool.md#build-and-test).

## Make a pool

`rustpool.make(task_id, **kwargs)` builds a pool. The pool options and the environment's own
config go in the same call:

```python
import numpy as np
import rustpool

print(rustpool.list_envs())     # ['BinPack-v0', 'CartPole-v1', 'Maze-v0']

pool = rustpool.make(
    "Maze-v0",
    num_envs=16,        # environments in the pool
    batch_size=16,      # how many ready environments recv returns
    seed=0,             # env i is seeded seed + i
    width=6, height=6,  # Maze-v0's own config
)
print(pool.num_threads, pool.dispatch, pool.autoreset)
print(pool.observation_spec(), pool.action_spec())   # {'agent_pos': ([2], 'float32')} (4,)
```

## A synchronous loop

With `batch_size == num_envs`, every `recv` returns the whole pool, sorted by `env_id`. Pair
each action with the `env_id` it is for:

```python
pool.async_reset()                              # non-blocking
obs, reward, done, info = pool.recv()           # blocks until 16 envs are ready
rng = np.random.default_rng(0)

episodes = 0
for _ in range(200):
    mask = info["action_mask"]                  # bool[16, 4]: Maze masks off-grid moves
    actions = (rng.random(mask.shape) * mask).argmax(axis=1).astype(np.int32)
    pool.send(actions, info["env_id"])          # non-blocking
    obs, reward, done, info = pool.recv()
    episodes += int(done.sum())
print("episodes finished:", episodes)
```

`obs` is a dict of NumPy arrays with a leading batch axis; `reward` is `float32[batch]`, `done`
is `bool[batch]`, and `info` holds `env_id` (and `action_mask` when the environment has one).
When an episode ends the environment resets itself: see
[Episodes and auto-reset](concepts/autoreset.md) for exactly what that call returns.

## An asynchronous loop

With `batch_size < num_envs`, `recv` returns the first `batch_size` environments to finish, and
the rest keep stepping. Send actions only for the environments you just received:

```python
pool.close()
pool = rustpool.make("Maze-v0", num_envs=16, batch_size=4, num_threads=4)
pool.async_reset()
for _ in range(200):
    obs, reward, done, info = pool.recv()       # 4 envs, whichever were ready first
    actions = rng.integers(0, 4, size=len(info["env_id"])).astype(np.int32)
    pool.send(actions, info["env_id"])          # only these 4 step again
pool.close()
```

This is how the policy's inference overlaps with simulation: while you compute actions for one
batch, the workers are stepping the others.

## JAX and PyTorch

The bridges run the pool synchronously and return device arrays or tensors:

```python
import jax
from rustpool.jax_api import JaxPool

pool = JaxPool("Maze-v0", num_envs=8)
out = jax.jit(pool.reset)(jax.random.key(0))    # EnvOutput(obs, reward, done, action_mask, env_id)
out = jax.jit(pool.step)(jax.numpy.zeros(8, jax.numpy.int32))
pool.close()
```

```python
import torch
from rustpool.torch_api import TorchEnvPool

pool = TorchEnvPool("Maze-v0", num_envs=8, device="cpu")
obs, reward, done, info = pool.reset()          # tensors; info["action_mask"] is bool
obs, reward, done, info = pool.step(torch.zeros(8, dtype=torch.int64))  # cast to int32 for you
pool.close()
```

Next: [The pool, threads and dispatch](concepts/pool.md).
