# Write an environment inside rustpool

This guide adds an environment to rustpool itself, so it becomes a task of `rustpool.make`.
The worked example is `CartPole-v1` (`src/envs/cartpole/mod.rs`), the smallest environment in the
tree. To keep your environment in its own crate instead, see
[Bring your own crate](own-crate.md): it uses the same trait and gets the same pool options.

## Build and test

```bash
git clone https://github.com/wiremind/rustpool && cd rustpool
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"       # maturin, pytest, ruff
maturin develop --release        # rebuild the extension: rerun after every Rust edit
cargo test --all                 # Rust unit tests
pytest -q                        # Python tests (they exercise the compiled extension)
```

The Python tests import the compiled module, so a stale build silently tests old Rust code: run
`maturin develop` after each change.

## 1. Implement `RlEnv`

Every environment implements one trait (`src/core/rl_env.rs`):

```rust
--8<-- "src/core/rl_env.rs:trait"
```

The contract, method by method:

| Method | Must return |
| --- | --- |
| `reset(seed)` | `(obs, 0.0, false, mask)`. Reseed from `seed` when it is `Some`, else continue the env's own random stream. |
| `step(action)` | `(obs, reward, done, mask)` for one transition. Do **not** reset on `done`: the pool does it (see [auto-reset](../concepts/autoreset.md)). |
| `observation_spec(py)` | `{key: (shape, dtype)}`, dtype one of `"float32"`, `"int32"`, `"bool"`. |
| `action_spec(py)` | `(num_actions,)`. |
| `obs_keys()` | The observation keys, in the order `obs` lists them. |

`obs` is a `Vec<ArrayData>` with one entry per key in `obs_keys()` order, each flattened (row
major) to the size its spec declares: `ArrayData::Float32`, `Int32` or `Bool`. The pool checks
every observation against the spec. `mask` is a `Vec<bool>` of length `num_actions` (`true` =
valid) or empty if the environment has no mask; it becomes `info["action_mask"]`.

CartPole's implementation:

```rust
--8<-- "src/envs/cartpole/mod.rs:impl"
```

Derive `Clone` on your environment (CartPole does): any `Clone` environment gets `snapshot` /
`simulate_batch` / `release_batch` for free, through the blanket `RlEnvClone` impl. Everything
an environment owns must be `Send + Sync`, since it lives on a worker thread.

## 2. A config and a registration

`rustpool.make(task_id, **kwargs)` hands the environment's kwargs to a parser registered under
`task_id`. The parser deserialises them with `registry::parse_env_kwargs` (serde, through
`pythonize`; with `#[serde(deny_unknown_fields)]` on the struct, an unknown key or a mistyped
value raises `TypeError("<task_id>: ...")` in Python), validates them (raise `ValueError` for an
out-of-range value), and returns a factory: a closure that builds one environment
from a seed. The pool calls it once per environment, as `factory(seed + i)`.

```rust
--8<-- "src/envs/cartpole/mod.rs:config"
```

Then wire the module in:

1. `src/envs/mod.rs`: `pub mod your_env;`
2. `src/registry.rs`, in `build_registry`: `crate::envs::your_env::register_env(&mut m);`
3. Optionally, Python-side default kwargs in `_DEFAULT_CONFIGS`
   (`rustpool/envpool_api/__init__.py`); serde's `#[serde(default)]` values apply anyway.

After `maturin develop`, the task is available:

```python
import rustpool

assert "CartPole-v1" in rustpool.list_envs()
pool = rustpool.make("CartPole-v1", num_envs=4, batch_size=4, max_episode_steps=50)
print(pool.observation_spec(), pool.action_spec())   # {'obs': ([4], 'float32')} (2,)
pool.close()
```

## 3. Optional capabilities

All defaulted on the trait, so an environment opts in only to what it needs:

- **`query_state(key) -> Option<String>`**: a JSON escape hatch, read from Python with
  `pool.query_state(env_id, key)`. CartPole answers `"state"` with its exact `f64` state, which
  its tests use to check the dynamics against gymnasium.
- **`action_factorization()` + `conditional_mask(prefix)`**: a factored action space, read with
  `query_mask` and `query_state_mask`. See [Factored action masks](../concepts/factored-actions.md);
  `BinPack-v0` (`src/envs/binpack/`) is the worked example.

## 4. Test it

Add Rust unit tests in the module (`#[cfg(test)]`, as CartPole does) and Python tests under
`tests/`. The shared fixtures in `tests/conftest.py` run every registered environment that has an
action mask through generic checks (spec shapes, seeding, invalid actions); an environment
without a mask is listed in `UNMASKED_TASKS` there and tested on its own, like
`tests/test_cartpole.py`.
