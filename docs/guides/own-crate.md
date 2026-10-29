# Bring your own crate

Keep your environment in **its own Rust crate** and still get the whole engine: the parallel
pool, both auto-reset conventions, state forks for search, factored masks and the JAX / PyTorch
bridges. You never edit rustpool's source. Your crate depends on rustpool's Rust crate
(`_rustpool`), implements the same `RlEnv` trait as the built-in environments, and exports
rustpool's pool class from its own Python module.

Both routes take the same pool options and seed the same way:

| | Inside rustpool | Your own crate |
| --- | --- | --- |
| Constructor | `rustpool.make(task_id, ...)` | `RlPool::new_with_options(num_envs, batch_size, factory, options)` |
| Options | `seed`, `num_threads`, `dispatch`, `autoreset` keywords | `PoolOptions { seed, num_threads, dispatch, autoreset }` |
| Env `i` is built as | `factory(seed + i)` | `factory(seed + i)` |
| Python class | `rustpool.RlPool` | the same Rust type, registered by your module |

The code on this page is `examples/rust/external_env.rs` in the repository: a complete crate in
one file, which rustpool's tests compile and import as a real external module.

## 1. The crate

```toml
# Cargo.toml
[package]
name = "external_env"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
# A local checkout while you develop, or a release tag:
_rustpool = { path = "../rustpool" }
# _rustpool = { git = "https://github.com/wiremind/rustpool", tag = "<release tag>" }
pyo3 = { version = "0.29", features = ["extension-module", "macros"] }   # the same pyo3 as rustpool
rand = "0.8"
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "external-env"
version = "0.1.0"
dependencies = ["rustpool", "numpy"]

[tool.maturin]
module-name = "external_env"
```

Use the same `pyo3` (and `numpy`) crate versions as rustpool: the pool class is a PyO3 class
compiled into your extension.

## 2. The environment

The same trait as an environment inside rustpool (see
[its contract](env-in-rustpool.md#1-implement-rlenv)). This one is a corridor with an action
mask:

```rust
--8<-- "examples/rust/external_env.rs:env"
```

## 3. The module

A `make` with the same signature as `rustpool.make`, and the pool class exported under your
module. `PoolOptions::parse` reads the option strings exactly as `rustpool.make` does, including
its `ValueError`s:

```rust
--8<-- "examples/rust/external_env.rs:module"
```

From Rust you can also fill the options directly:

```rust
--8<-- "examples/rust/external_env.rs:options"
```

## 4. Build and use it

```bash
pip install maturin
maturin develop --release        # builds and installs external_env into the active venv
```

```python
import numpy as np
import external_env
import rustpool

pool = external_env.make(num_envs=8, batch_size=8, seed=0, autoreset="next_step", length=6)
print(pool.num_threads, pool.dispatch, pool.autoreset)
print(isinstance(pool, rustpool.RlPool))     # False: your module registers its own class
print(type(pool).__module__)                   # "rustpool._rustpool" all the same (see below)
assert isinstance(pool, external_env.RlPool)

pool.async_reset()
obs, reward, done, info = pool.recv()            # obs["pos"]: int32[8, 1]; info["action_mask"]: bool[8, 2]
for _ in range(20):
    pool.send(np.ones(8, np.int32), info["env_id"])
    obs, reward, done, info = pool.recv()

roots = np.asarray(pool.snapshot(info["env_id"]), np.int32)     # Clone => search works too
pool.release_batch(roots)
pool.close()
```

The class's name and module are fixed when rustpool is compiled, so your pool's class reports
`__module__ == "rustpool._rustpool"` and `__name__ == "RlPool"` (also in its repr), although it
is your module's own class. To check for it, use `isinstance` against the class your module
exports (`external_env.RlPool` here), never a lookup by module and name.

Because the class is not `rustpool.RlPool`, the bridges accept any object with the pool API.
Wrap a synchronous pool (`batch_size == num_envs`) with `JaxPool.from_pools` or
`TorchEnvPool.from_pool`:

```python
import jax
from rustpool.jax_api import JaxPool

jpool = JaxPool.from_pools(external_env.make(num_envs=8, batch_size=8))
out = jax.jit(jpool.reset)(jax.random.key(0))
out = jax.jit(jpool.step)(jax.numpy.ones(8, jax.numpy.int32))
print(out.obs["pos"].shape, out.action_mask.shape)   # (8, 1) (8, 2)
jpool.close()
```

## What you get, and what you don't

You get everything the pool does: `send` / `recv` in both regimes, `dispatch`, `autoreset`,
`snapshot` / `simulate_batch` / `release_batch` (if your environment is `Clone`),
`query_mask` / `query_state_mask` (if it implements a factorization), `query_state`, and the
bridges through `from_pools` / `from_pool`.

What stays with `rustpool.make` only: the registry of task ids and the per-task Python default
configs. Your `make` takes whatever environment arguments you give it.
