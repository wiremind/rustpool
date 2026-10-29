# rustpool

**A pure-Rust parallel reinforcement-learning environment-execution engine for Python.**

rustpool runs many RL environments on native Rust worker threads (true parallelism, no GIL)
behind an asynchronous, [EnvPool](https://github.com/sail-sg/envpool)-style `send`/`recv` API,
and adds the primitives EnvPool does not have:

- 🌳 a **state-cloning substrate** — `snapshot` / `simulate_batch` / `release_batch` — that lets
  a batched **MCTS / AlphaZero** search fork arbitrary states of an *external, non-differentiable*
  simulator (the thing on-device JAX search like `mctx`/Pgx cannot do unless your env is a pytree);
- 🎛️ **factored / autoregressive action masks** (`query_mask` / `action_factorization`) so a policy
  can sample a large flat action one component at a time under conditional ∃-witness masks;
- ⚡ a **first-class JAX bridge** (`rustpool.jax_api.JaxPool`) and **zero-copy PyTorch interop**
  (`rustpool.torch_api.TorchEnvPool`), plus a gymnasium wrapper.

It is built with [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs); the heavy lifting
(threading, GIL management, NumPy batching, Rayon-parallel simulation) is all in safe Rust.

```python
import numpy as np, rustpool

print(rustpool.list_envs())                 # ['BinPack-v0', 'CartPole-v1', 'Maze-v0']
pool = rustpool.make("Maze-v0", num_envs=64, batch_size=64)
pool.async_reset()
obs, reward, done, info = pool.recv()        # obs: dict of NumPy arrays
actions = np.zeros(len(info["env_id"]), dtype=np.int32)
pool.send(actions, info["env_id"])           # non-blocking
obs, reward, done, info = pool.recv()
pool.close()
```

## Install

```bash
pip install rustpool                 # core: just numpy
pip install "rustpool[gymnasium]"    # + the gymnasium vector-env wrapper
pip install "rustpool[jax]"          # + the JaxPool bridge
pip install "rustpool[examples]"     # + flax/optax/mctx to run the PPO/AlphaZero examples
pip install "rustpool[torch]"        # + zero-copy PyTorch interop
```

Prebuilt manylinux2014 wheels are published for CPython 3.11 / 3.12 / 3.13 / 3.14 (Linux x86-64).
Build from source needs a Rust toolchain (pinned in `rust-toolchain.toml`).

## Why rustpool? (vs EnvPool)

|  | EnvPool | rustpool |
| --- | --- | --- |
| Core | C++ (3 buffer-queue components) | Rust threads + `crossbeam` channels |
| Python binding | pybind11 | PyO3 + maturin |
| Async batched `recv` (act on the first `batch_size` ready) | ✅ | ✅ |
| In-worker auto-reset | ✅ | ✅ |
| **State clone / fork for search** | ❌ | ✅ `snapshot`/`simulate_batch`/`release_batch` |
| **Factored / autoregressive action masks** | ❌ | ✅ `query_mask`/`action_factorization` |
| JAX integration | pure `env.xla()` (no host hop) | `io_callback` bridge (one host hop/step) |
| PyTorch / gymnasium adapters | ✅ | ✅ |
| Bundled env suite (Atari, MuJoCo, …) | ✅ large | ❌ write envs in Rust (`RlEnv` trait) |

rustpool is positioned as a **capability superset for external simulators**, not as a faster
EnvPool. EnvPool's pure-XLA `env.xla()` step needs no host round-trip; rustpool's `io_callback`
synchronizes to the host each step — the trade-off that buys you a non-pytree simulator and the
search substrate. See the [benchmarking methodology](https://github.com/wiremind/rustpool/blob/main/benchmarks/README.md) for how the
two are compared fairly.

## Capabilities

### Asynchronous, agent-driven batching

`recv()` returns as soon as `batch_size` environments are ready (not all `num_envs`), so with
`batch_size < num_envs` the agent acts on whichever envs finish first while slow envs keep
stepping — overlapping policy inference with simulation. The blocking receive releases the GIL.

### Worker threads (several pools per process)

Each pool spawns its own Rust worker threads, one shard of envs per thread. By default a pool
uses every CPU available to the process (`std::thread::available_parallelism()`, which honours
the cgroup CPU quota and affinity mask), capped at `num_envs`. That is right for one pool, but a
process holding several pools (e.g. one per GPU) then runs `n_pools x n_cpus` workers and
oversubscribes the machine. Set the count per pool:

- `rustpool.make(..., num_threads=N)` / `RlPool(..., num_threads=N)` (keyword-only), or
- the `RUSTPOOL_NUM_THREADS=N` environment variable, read when each pool is **constructed** (so
  it can differ between pools built one after another; external env crates honour it too
  unless they pass `PoolOptions { num_threads: Some(n), .. }` to `RlPool::new_with_options`).

The explicit keyword wins over the variable, and the count is always capped at `num_envs` (a
thread past that would own no env). A `RUSTPOOL_NUM_THREADS` that is not a positive integer, or
`num_threads=0`, raises `ValueError`; an empty variable counts as unset. The keyword must be a
non-negative `int` (Python's own conversion raises `OverflowError` for a negative one and
`TypeError` for a float or string). `pool.num_threads` reports the value used.

`simulate_batch` (tree search) and the observation stacking in `recv` run on Rayon's **global**
pool instead, which is one per process (shared by all pools) and sized by the standard
`RAYON_NUM_THREADS` variable (default: all available CPUs). It must be set before the first
Rayon call in the process.

### Stateless simulation for MCTS / AlphaZero

```python
import numpy as np, rustpool
pool = rustpool.make("Maze-v0", num_envs=8, batch_size=8); pool.async_reset()
_, _, _, info = pool.recv()
roots = pool.snapshot(info["env_id"])                                   # freeze live states
obs, rew, done, child = pool.simulate_batch(                            # fork-and-step (no mutation)
    np.array(roots, np.int32), np.zeros(len(roots), np.int32))
pool.release_batch(np.array(child["state_id"], np.int32))              # free tree nodes
pool.close()
```

`simulate_batch` clones-on-read so a parent state can be expanded any number of times
(persistent / immutable-node semantics), stepping the clones in parallel with Rayon. This is the
substrate the AlphaZero example drives through `mctx`.

### JAX (single- and multi-GPU)

```python
import jax, jax.numpy as jnp, rustpool.jax_api as rj
pool = rj.JaxPool("Maze-v0", num_envs=256)          # num_devices=n: one pool per device for pmap
out  = jax.jit(pool.reset)(jax.random.key(0))
out  = jax.jit(pool.step)(jnp.zeros(256, jnp.int32))
pool.close()
```

### PyTorch (zero-copy)

```python
from rustpool.torch_api import TorchEnvPool
pool = TorchEnvPool("Maze-v0", num_envs=64, device="cpu")   # or "cuda"
obs, reward, done, info = pool.reset()              # tensors, masks included
pool.close()
```

## Examples

Runnable PPO and AlphaZero scripts, in JAX and in PyTorch, live in
[`examples/`](https://github.com/wiremind/rustpool/tree/main/examples) — see [`examples/README.md`](https://github.com/wiremind/rustpool/blob/main/examples/README.md). They run on a single
GPU or shard across all local GPUs (`--num-devices 0`, Anakin-style pmap):

```bash
pip install "rustpool[examples]"
python examples/ppo_jax.py --task Maze-v0          # PPO, all local GPUs
python examples/alphazero_jax.py --num-simulations 32   # true-simulator MCTS via mctx
python examples/ppo_torch.py                       # PPO in torch; needs rustpool[torch]
python examples/alphazero_torch.py                 # AlphaZero in torch (batched MCTS)
```

## Built-in environments

| Task id | Description |
| --- | --- |
| `Maze-v0` | Grid navigation to a corner goal; sparse +1 reward; 4 masked actions. A learning/sanity task. |
| `BinPack-v0` | 3D container loading (EMS-based); dense (volume) or sparse reward; large factored item×EMS action space with a validity mask. |
| `CartPole-v1` | The classic cart-pole task, a port of gymnasium's (same dynamics, 500-step limit); 2 actions, no mask. |

Construct with `rustpool.make(task_id, num_envs=..., batch_size=..., **kwargs)`; pass
`task_id`/env kwargs (validated in Rust: an unknown or mistyped key raises `TypeError`). See
`rustpool.list_envs()`.

## Writing your own environment

rustpool is a generic execution engine: you can keep your environment in **its own crate** and let
rustpool provide the threading, GIL management, NumPy batching, and the MCTS / factored-action
substrate. Two ways to add an env:

- **In-tree** — implement the `RlEnv` trait under `src/envs/` and register it (see
  [`CONTRIBUTING.md`](https://github.com/wiremind/rustpool/blob/main/CONTRIBUTING.md)); it then works through `rustpool.make("YourEnv-v0", ...)`.
- **External crate** — depend on the `_rustpool` Rust crate and inject a factory; you never touch
  rustpool's source. `RlPool::new_with_options` takes the same options as the Python
  constructor (`seed`, `num_threads`, `dispatch`, `autoreset`), and in both routes env `i` is
  built as `factory(seed + i)`. This is how a production env crate consumes rustpool:

```toml
# your_env/Cargo.toml
[lib]
crate-type = ["cdylib", "rlib"]
[dependencies]
_rustpool = { git = "https://github.com/wiremind/rustpool.git", tag = "<release tag>" }
pyo3 = { version = "0.29", features = ["extension-module", "macros"] }
```

```rust
// your_env/src/lib.rs
use _rustpool::{ArrayData, FactorizationSpec, GenericObs, RlPool, PoolOptions, RlEnv};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone)]               // Clone => snapshot / simulate_batch (MCTS) for free
struct MyEnv { /* ... */ }

impl RlEnv for MyEnv {
    fn reset(&mut self, seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>) { /* ... */ }
    fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>) { /* ... */ }
    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { /* {key:(shape,dtype)} */ }
    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> { /* (num_actions,) */ }
    fn obs_keys(&self) -> Vec<String> { /* ... */ }
    // Optional: opt into the factored action space (see docs/concepts/factored-actions.md)
    fn action_factorization(&self) -> Option<FactorizationSpec> { None }
    fn conditional_mask(&self, _prefix: &[i32]) -> Vec<bool> { Vec::new() }
}

#[pyfunction]
#[pyo3(signature = (num_envs=4, batch_size=4, seed=0, *, num_threads=None, dispatch="auto",
                    autoreset="same_step", **kwargs))]
fn make(
    num_envs: usize, batch_size: usize, seed: u64, num_threads: Option<usize>, dispatch: &str,
    autoreset: &str, kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<RlPool> {
    // Same options and the same parsing (ValueError on a bad string) as rustpool.make.
    let options = PoolOptions::parse(seed, num_threads, dispatch, autoreset)?;
    // Env i is factory(seed + i), as in rustpool.make.
    let factory = Box::new(move |env_seed| Box::new(MyEnv::new(env_seed)) as Box<dyn RlEnv>);
    RlPool::new_with_options(num_envs, batch_size, factory, options)
}

#[pymodule]
fn _your_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RlPool>()?;   // re-export the pool so Python gets the full send/recv/MCTS API
    m.add_function(wrap_pyfunction!(make, m)?)?;
    Ok(())
}
```

Your env then gets the entire engine for free: async `send`/`recv`, in-worker auto-reset (either
convention), the `snapshot`/`simulate_batch`/`release_batch` MCTS substrate (if `Clone`), the
JAX/PyTorch bridges (`JaxPool.from_pools(pool)`, `TorchEnvPool.from_pool(pool)`: your pool is a
different Python class from `rustpool.RlPool`, and the bridges accept it), and — if you implement `action_factorization` + `conditional_mask` — factored autoregressive masks
via `query_mask`. See [`CONTRIBUTING.md`](https://github.com/wiremind/rustpool/blob/main/CONTRIBUTING.md), [`src/core/rl_env.rs`](https://github.com/wiremind/rustpool/blob/main/src/core/rl_env.rs) (the trait), and
[`docs/concepts/factored-actions.md`](https://github.com/wiremind/rustpool/blob/main/docs/concepts/factored-actions.md).

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[examples,dev]"   # maturin, pytest, ruff, the examples' jax/flax/torch
maturin develop                       # rebuild the Rust extension in place after Rust edits
pytest                                # Python tests
cargo test                            # Rust unit tests
```

## Releasing (maintainers)

See [Releasing in CONTRIBUTING.md](https://github.com/wiremind/rustpool/blob/main/CONTRIBUTING.md#releasing).

## License

Dual-licensed under either of [MIT](https://github.com/wiremind/rustpool/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/wiremind/rustpool/blob/main/LICENSE-APACHE) at your option.

## Citing

If you use rustpool in academic work, see [`CITATION.cff`](https://github.com/wiremind/rustpool/blob/main/CITATION.cff).
