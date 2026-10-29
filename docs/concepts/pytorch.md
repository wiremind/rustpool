# PyTorch

`rustpool.torch_api.TorchEnvPool` is a synchronous (`batch_size == num_envs`) view of a pool
that returns PyTorch tensors. `recv` builds fresh, writable NumPy arrays every call, so the
conversion is `torch.from_numpy`: zero-copy, then moved to `device` if you give one.

```python
import torch
from rustpool.torch_api import TorchEnvPool

pool = TorchEnvPool("Maze-v0", num_envs=8, device="cpu", seed=0)
obs, reward, done, info = pool.reset()
print(obs["agent_pos"].dtype, info["action_mask"].dtype)    # torch.float32 torch.bool

for _ in range(10):
    logits = torch.zeros(8, 4).masked_fill(~info["action_mask"], float("-inf"))
    actions = torch.distributions.Categorical(logits=logits).sample()    # int64
    obs, reward, done, info = pool.step(actions)            # cast to int32 for you
```

The pool takes `int32` arrays; torch's integer default is `int64`. `TorchEnvPool` casts actions,
env ids and state ids at the boundary, so you never pass a dtype by hand.

## Search and factored masks

The same search primitives as the pool, returning tensors:

```python
roots = pool.snapshot()                                     # every env; an int32 tensor
obs2, reward2, done2, child = pool.simulate_batch(roots, torch.zeros(8, dtype=torch.int64))
pool.release_batch(child["state_id"])
pool.release_batch(roots)
print(pool.action_factorization())                          # None: Maze has a flat action space
pool.close()
```

`query_mask(prefixes, env_ids=None)` returns the next factor's mask as a `bool` tensor (see
[Factored action masks](factored-actions.md)).

## Wrapping a pool you already have

`TorchEnvPool.from_pool(pool, device=...)` wraps an existing synchronous pool, including one
built by [your own crate](../guides/own-crate.md):

```python
import rustpool

base = rustpool.make("CartPole-v1", num_envs=4, batch_size=4, autoreset="next_step")
tpool = TorchEnvPool.from_pool(base, device="cpu")
obs, reward, done, info = tpool.reset()
print(obs["obs"].shape)                                     # torch.Size([4, 4])
tpool.close()
```
