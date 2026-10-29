# Changelog

All notable changes to rustpool are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project aims to follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may carry breaking changes).

## [Unreleased] — publication-readiness pass

### Changed
- **BREAKING: `close()` frees the stored search states**, and `simulate_batch` and
  `query_state_mask` then raise `RuntimeError` like every other call (they kept working on a
  closed pool, holding the states until garbage collection). `release_batch` and a second
  `close()` stay no-ops.
- **BREAKING: `query_mask` and `query_state_mask` raise `ValueError` on an env with no action
  factorization** (they returned `None`, or `JaxPool` / `TorchEnvPool` an all-false
  `[n, pad_width]` mask). `JaxPool` raises when the function is traced. On a factored env, a
  call with no rows returns `bool[0, pad_width]`.
- **BREAKING: an unknown environment option raises `TypeError`** (`TypeError("Maze-v0: unknown
  field \`widht\`, expected one of ...")`); it used to be ignored silently. A mistyped value
  also raises `TypeError` (it raised a bare `Exception`). External crates can use the same
  `registry::parse_env_kwargs` helper. `make()` no longer passes BinPack a `max_episode_steps`
  that BinPack does not have.
- **BREAKING: the pool class is `RlPool`.** Its runtime name (repr, errors, `type(pool)`) was
  `MyEnvPool`, with `RlPool` only a Python alias. `MyEnvPool` stays as a deprecated alias:
  `rustpool.MyEnvPool` warns (`DeprecationWarning`), and the Rust type alias is
  `#[deprecated]`.
- **BREAKING for external crates: PyO3 0.29** (from 0.22), with numpy and pythonize 0.29.
  This adds **Python 3.14** support (wheels and CI), and clears two RustSec advisories against
  PyO3 0.22. `RlEnv::observation_spec` / `action_spec` now return `PyResult<Py<PyAny>>`
  (`PyObject` no longer exists), and a crate implementing `RlEnv` must depend on `pyo3 = "0.29"`.
  Most other changes are renames: `PyDict::new_bound` becomes `PyDict::new`, `to_object(py)`
  becomes `into_pyobject(py)?.into_any().unbind()`, and `with_gil` / `allow_threads` become
  `attach` / `detach`.
- **BREAKING: gymnasium is an optional extra** (`pip install "rustpool[gymnasium]"`, which needs
  gymnasium 1.1 or later); `pip install rustpool` installs only NumPy.
- **BREAKING: `RustPoolGymnasiumWrapper` is a `gymnasium.vector.VectorEnv`** built with
  `rustpool.make(task_id, num_envs=..., **kwargs)`, with `task_id` as its first argument. It
  forwards every environment option (it used to drop them, so `width=3` still built a 10x10
  maze), declares observation bounds that contain what the env returns (unbounded floats, where
  it declared `[0, 1]`), and auto-resets on the next step through the engine
  (`AutoresetMode.NEXT_STEP`) instead of a Python-side step counter that flagged truncation
  without resetting the env. It is synchronous (`batch_size == num_envs`), reports
  `truncated=False`, and has a docs page.
- **BREAKING: `BinPack-v0(factored=True)` without the flat mask now emits the stage-0 (item)
  mask** in `info["action_mask"]` / `JaxPool`'s `EnvOutput.action_mask`, padded to the
  factorization's `pad_width` (it was `[B, 0]`); `action_mask_width()` returns `pad_width`. It
  equals `query_mask` at the empty prefix, so a factored sampler needs `K-1` `query_mask`
  round-trips per step instead of `K`. `emit_flat_mask=False` on a flat env still emits `[B, 0]`.
- **Per-worker dispatch, chosen automatically** (`RlPool(..., dispatch="auto" | "per_env" |
  "per_worker")`, keyword-only, default `"auto"`). `per_env` is the previous protocol: one
  channel message (and worker wake-up) per env each way. `per_worker` sends one message per
  worker thread per `send` / `query_mask` call and returns that worker's results together, so
  channel operations per call drop from O(rows) to O(threads). `auto` picks `per_worker` when a
  call has more rows than there are worker threads, else `per_env`. Results are identical in
  every mode (trajectories, async bookkeeping and `query_mask` rows; `tests/test_dispatch.py`);
  only the transport changes. External env crates (`new_with_factory*`) get `auto` too.
- **`recv` stacks small batches on the calling thread**: observation columns go to Rayon only
  above 2^18 stacked elements; below that, handing the copy to the pool cost more than the copy.

### Documentation
- **The documentation site is rewritten for the current engine** (MkDocs + Material, built
  locally with `mkdocs build --strict`; not published). Concepts: the pool and dispatch,
  auto-reset conventions, factored masks, state forks and the registry's lifetime, the JAX
  bridge (which calls are ordered), PyTorch. Guides: an environment inside rustpool (CartPole),
  an environment in your own crate, training with the examples. References: the Python API
  (now the pool's full surface, read from the compiled module), the Rust API, this changelog.
  `tests/test_docs_snippets.py` runs every Python snippet; the guides include their Rust code
  from compiled files, and the hand-written Rust signatures are checked against `src/`.
- **`examples/rust/external_env.rs`**: a complete external env crate in one file (built as a
  `cdylib` Cargo example); `tests/test_external_crate.py` imports it as its own module and drives
  it through the pool options, the search calls and both bridges.
- CI gains a `docs` job that only builds the site.

### Internal (unstable, outside semver)
- **BinPack-v0 bench queries** `query_state(env_id, "bench:flat" | "bench:chain" |
  "bench:place"[":reps"])`, `reps` in 1..=1000: on a clone of the frozen state, the time and
  `can_fit` tests of the flat mask, of the factored chain for one committed action, and of placing
  it (EMS update and re-sort). Calls are counted by counting twins of the mask functions that run
  only in these queries (the step path has no counter). A benchmarking hook: its keys and reply
  may change in any release.

### Removed
- **BREAKING: the Stoa wrapper (`rustpool.envpool_api.StoaRustpoolWrapper`) and the `stoa`
  extra are removed.** Code built on Stoa keeps its own copy of the wrapper; rustpool's JAX
  entry point is `rustpool.jax_api.JaxPool`.
- `rerun_plotters/` (an unreferenced, untested BinPack visualisation script from before the
  environments moved out) and the `viz` extra that only it used.
- **Research experiment harnesses moved out of the library**: the engine comparison against
  EnvPool and gymnasium, the thread-scaling, mask-cost, PPO, search and AlphaZero benchmarks
  and their tests. `benchmarks/bench_engine.py`,
  `bench_factored.py`, `README.md` and `reference.json` are back to their v0.1.8 versions, and
  the `bench` extra (EnvPool) is gone.

### Fixed
- **`simulate_batch([], [])` returns empty arrays** (it raised "No observations to stack"),
  like the other calls on zero rows; and the compiled module's docstring is its own again (it
  had been attached to the internal test hook).
- **BinPack-v0 no longer kills a worker when used before its first reset.** A step, a
  snapshot fork or a `bench:*` query on a never-reset env indexed empty per-container state,
  and the caller then panicked too. The env now starts as a valid instance with no items: a step
  ends the episode (auto-reset then starts a real one) and the bench query answers `None`.
- **The gymnasium wrapper accepts a NumPy integer seed** (`reset(seed=np.int64(5))` raised
  `TypeError`).
- **A multi-device `JaxPool` mapped over the wrong axis raises instead of misrouting.** Under
  `pmap` with an axis name other than `"device"`, every replica silently used pool 0; it now
  raises `ValueError` at trace time, as does a `"device"` axis whose size is not `num_devices`.
  Plain `jit` still uses pool 0. A child-process test runs `JaxPool` under `pmap` on two forced
  host devices.
- **Every env-id error is `IndexError("env_id -1 is out of bounds for num_envs 4")`.**
  `snapshot` raised `ValueError` and reported `-1` as `18446744073709551615`; `query_state`
  raised `OverflowError` for a negative id. A bad state id stays `ValueError`, now always
  worded `invalid state_id N (unknown or released)`.
- **`snapshot` accepts a repeated `env_id`**: each occurrence gets its own handle. It used to
  panic (`PanicException`, which `except Exception` does not catch).
- **Out-of-range options raise `ValueError` when the pool is built** instead of killing a worker
  on the first step: `Maze-v0` `width` / `height` / `max_episode_steps` below 1, `BinPack-v0`
  `max_ems` below 1, `prob_split_one_item` outside `[0, 1]` and `split_eps` outside `[0, 0.5]`,
  and `num_envs` or `batch_size` of 0.
- **`with rustpool.make(...) as pool:` works**: the pool gains `__enter__` / `__exit__` (closing
  the pool on exit), which `close()`'s docstring already promised.
- **`MyEnvPool.__module__` is `rustpool._rustpool`** (it was `builtins`), so tools that resolve
  the class by name, such as the docs generator, find it.
- **A rejected `send` dispatches nothing.** `send` checked each `env_id` while sending, so an
  invalid id (out of range, or negative) in the middle of a batch raised only after the actions
  before it had reached their workers: those envs stepped once more than the caller knew. Every
  id is now validated first; `query_mask` also rejects negative ids up front.

### Added
- **`autoreset="same_step" | "next_step"`** (`RlPool(...)` / `rustpool.make`, keyword-only,
  default `"same_step"`, getter `pool.autoreset`). `same_step` is the existing behaviour: the
  terminal step returns the next episode's first observation with `done=True`. `next_step` is
  EnvPool's and gymnasium's convention: the terminal step returns the terminal observation, and
  the env's next `send` is the reset (its action ignored; reward 0, `done=False`).
- **External env crates get every pool option**: `MyEnvPool::new_with_options(num_envs,
  batch_size, factory, PoolOptions { seed, num_threads, dispatch, autoreset })`, with
  `PoolOptions::parse` for the Python constructor's string forms. Env `i` is `factory(seed + i)`
  in both routes. `new_with_factory` / `new_with_factory_and_threads` are kept as wrappers with
  the default options (seed 0, so their `factory(i)` is unchanged).
- **`JaxPool.from_pools(pools)` and `TorchEnvPool.from_pool(pool)`** wrap pools built elsewhere,
  duck-typed, so a pool from an external env crate (a different Python class from
  `rustpool.MyEnvPool`) gets the JAX and PyTorch bridges. The pool must be synchronous
  (`batch_size == num_envs`).
- **BinPack attention networks** (`examples/networks.py`): `BinPackAttention`, a one-layer,
  single-head version of Jumanji's BinPack actor-critic (item and EMS tokens, self- and
  cross-attention), and `BinPackLatent`, the same with a trunk linear in the token count (16
  learned latent tokens). One module serves both head types: flat scores every (item, EMS) pair
  (`I*C*E` logits); factored scores item, then container | item, then EMS | item, container
  (`I + C + E`), never materialising the action grid.
- **`CartPole-v1`**: a Rust port of gymnasium's `CartPoleEnv` (f64 Euler dynamics, 500-step
  limit, reward 1 per step, no action mask) for engine-overhead benchmarks; next states equal
  gymnasium's over 10k random transitions (`query_state(env_id, "state")` exposes the f64 state).
- **BinPack-v0 `shuffle_items`** (default `False`, so the defaults are unchanged): shuffles the
  generated items, and their ground-truth locations with them, with the env RNG at every reset.
  Without it the `C` instances of a multi-container env are concatenated in order, so an item's
  slot tells which container's perfect packing it came from.
- **BinPack-v0: several containers and item orientations**, in place (same env name, new kwargs
  only): `num_containers=C` (default 1; `max_items`/`max_ems` become per container, one instance
  generated per container, a separate EMS list and sorted-EMS cache per container) and
  `orientations=O` in {1, 2, 6} (default 1; `can_fit` tests the oriented dims). Factor chain
  `item -> orientation -> container -> ems` with single-option stages dropped; flat action
  `((item * O + o) * C + c) * max_ems + ems`, `|A| = 1250 * O * C^2` at the default sizes; dense
  reward normalised by the total capacity. New obs keys `items_container` / `items_orientation`
  only when the feature is on. At `C = O = 1` the env is bit-identical to before (anchor test:
  hashed seeded trajectories, 64 envs x 300 steps, flat / sparse / factored). The JAX AlphaZero
  sampler and `networks.factored_flat_logits` handle any number of factors; `alphazero_jax.py`
  gained `--num-containers` / `--orientations`.
- **Factored AlphaZero without the full mask**: `alphazero_jax.py --factored` now runs BinPack
  with `factored=True` and no flat mask; each tree node's children are `--num-candidates` full
  actions sampled without replacement from the factored policy (stochastic beam search), with
  legality read per factor on stored states via `JaxPool.query_state_mask`. The previous mode
  (full mask, every action a child, factored heads composed into the full prior) is now
  `--task BinPack-v0` in both `alphazero_jax.py` and `alphazero_torch.py`; **BREAKING** for the
  examples' CLI: `alphazero_torch.py --factored` is gone.
- **Conditional masks of stored states**: `MyEnvPool.query_state_mask(state_ids, prefixes,
  width=None)` returns the non-advancing factored masks of registry handles (`snapshot` /
  `simulate_batch`), `[n, W]` in input order (duplicate ids allowed), so a batched tree search can
  read per-factor masks at any node. `state_id < 0` answers an all-false row without a lookup;
  unknown/released ids and rows whose width is not `W` (`width`, else the spec's `pad_width`)
  raise `ValueError`; `None` for an env without conditional masks. Reads the registry without
  cloning (states are now held as `Arc`s, so the lock is released before the parallel mask work)
  and with the GIL released. Also on
  `JaxPool.query_state_mask` and `StoaRustpoolWrapper.query_state_mask(state_ids, prefixes, width)`.
- **Configurable worker-thread count per pool**: keyword-only `num_threads=` on
  `MyEnvPool`/`make`, or the `RUSTPOOL_NUM_THREADS` env var read at pool construction (also
  honoured by `new_with_factory`; `new_with_factory_and_threads` takes it explicitly). Unset keeps
  the old `available_parallelism()` default; a non-positive / non-integer `RUSTPOOL_NUM_THREADS`
  or `num_threads=0` raises `ValueError` (a negative or non-int keyword fails Python's int
  conversion: `OverflowError` / `TypeError`); always capped at
  `num_envs`. New read-only `pool.num_threads`. Lets a process with one pool per device split its
  CPUs instead of spawning `n_pools x n_cpus` workers. Rayon's global pool (used by
  `simulate_batch` / obs stacking) stays sized by `RAYON_NUM_THREADS`.
- **First-class JAX bridge** `rustpool.jax_api.JaxPool`: jittable, pmappable `reset`/`step`/
  `snapshot`/`simulate_batch`/`release_batch` via `io_callback`; one independent pool per device
  routed by `jax.lax.axis_index` for single- and multi-GPU (Anakin-style) use.
- **Zero-copy PyTorch interop** `rustpool.torch_api.TorchEnvPool` (+ `to_torch`), with an
  explicit int32-action contract at the boundary.
- **Examples** (`examples/`): single-file PPO (`ppo_jax.py`) and AlphaZero (`alphazero_jax.py`)
  in pure JAX driving rustpool — AlphaZero uses `mctx` with the real simulator as the
  `recurrent_fn` — plus single-file PyTorch PPO (`ppo_torch.py`, CleanRL-style with
  `CategoricalMasked`) and AlphaZero (`alphazero_torch.py`, a hand-written array-batched MCTS
  over `snapshot`/`simulate_batch`), with shared helper modules `networks.py` (JAX) and
  `torch_common.py` (PyTorch).
- **PyTorch state-cloning substrate**: `TorchEnvPool` gained `snapshot` / `simulate_batch` /
  `release_batch` (mirroring `JaxPool`) so torch users can drive MCTS over the Rust engine.
- **Factored action spaces end-to-end**: `BinPack-v0` takes `factored=True`, advertising a
  2-factor (`item → ems`) conditional action space (`action_factorization()` +
  `conditional_mask()`). `JaxPool` and `TorchEnvPool` gained `query_mask()` + `action_factorization()`.
  New showcases drive it with both algorithms in both frameworks: `examples/factored_ppo_{jax,torch}.py`
  (autoregressive PPO) and `--factored` on `alphazero_{jax,torch}.py` (factored head → flat prior);
  `benchmarks/bench_factored.py` measures the flat-vs-factored policy-step speedup. Shared helpers
  added to `networks.py`/`torch_common.py` (`FactoredActorCritic`, `factored_sample`,
  `factored_log_prob_entropy`, `factored_flat_logits`).
- **Typing**: `py.typed` marker and a hand-written `_rustpool.pyi` stub for the native module.
- **Docstrings / signatures** on the whole PyO3 surface (`MyEnvPool`, `snapshot`,
  `simulate_batch`, `release_batch`, `query_mask`, `action_factorization`, …) and a module
  docstring + `__version__`.
- `rustpool.RlPool` alias for `MyEnvPool`; `rustpool.list_envs()` backed by the Rust registry
  (`registered_tasks()`).
- Dual `LICENSE-MIT` / `LICENSE-APACHE`; `CITATION.cff`, `CONTRIBUTING.md`, this changelog.
- Documentation site (`mkdocs.yml`, `docs/`) and benchmarks (`benchmarks/`).
- Example smoke tests (`tests/test_examples_smoke.py`); CI now runs the test suites.

### Changed
- **BinPack env step is much faster** at large sizes: the EMS containment prune is now
  output-sensitive (only newly created fragments are tested — 19× faster at 256 × 512), and the
  volume-sorted EMS view is computed once per step instead of five times. Same outputs. The
  pool step drops from 37.7 to 10.6 ms at 256 × 512 / 64 envs, and no longer slows down
  superlinearly as a container fills.
- **Breaking — `BinPack-v0(factored=True)` no longer emits the flat action mask by default**
  (new kwarg `emit_flat_mask`, default `not factored`). `info["action_mask"]` / `EnvOutput.action_mask`
  is `[B, 0]` unless you pass `emit_flat_mask=True`; read masks through `query_mask`. With the
  mask off the pool step is a further ~3.8× faster at 256 × 512. Flat (non-factored) envs are
  unchanged. `MyEnvPool.action_mask_width()` reports the real width; `JaxPool` now sizes its
  output template from it.
- **Packaging**: runtime dependencies trimmed to `numpy` + `gymnasium`; JAX, PyTorch, Stoa,
  plotting and dev/test stacks moved to `optional-dependencies` extras
  (`jax`, `examples`, `torch`, `stoa`, `viz`, `docs`, `dev`). Added full `[project]` metadata
  (description, authors, SPDX `MIT OR Apache-2.0` license, classifiers, keywords) and Cargo
  package metadata.
- `import rustpool` no longer eagerly imports JAX/Stoa — the Stoa wrapper is imported lazily.

### Fixed
- README corrected (right import name, real registered envs, accurate build/publish steps);
  removed references to an unregistered env and stale testpypi/docker steps.
- Removed confused placeholder comments and a dead self-assignment in `src/envs/binpack/mod.rs`.
- `query_mask` now collects results positionally (by row index), fixing wrong masks / a leaked
  message when the same `env_id` appears in multiple rows; it also validates all `env_id`s before
  dispatching so a bad row can't leave partial requests in flight.
- The pool methods now raise a clean `RuntimeError("environment pool has been closed")` after
  `close()` instead of an opaque `PanicException`; `close()` no longer re-panics if a worker
  panicked (matches `Drop`).
- CI no longer hard-fails at collection without the optional stack: `test_stoa_wrapper.py` guards
  its `jax`/`stoa` imports; the GitHub `release.yml` now gates publishing on the test suite.
- Cleaned the gymnasium wrapper (deduplicated imports, removed a misleading
  `info["final_observation"]`, initialise `step_count`); removed an orphaned `src/test_serde.rs`;
  bumped the crate to Rust edition 2021; added `[project.urls]` and `torch` to the `examples`
  extra. Repo is `ruff`-clean.

### Compatibility
- No breaking changes to the native API: `rustpool.MyEnvPool`, `rustpool.make`, and
  `rustpool.envpool_api.StoaRustpoolWrapper` are unchanged, so existing consumers keep
  working.

## [0.1.6]
- The **native** factored-action primitive on the engine: the `RlEnv` trait's
  `action_factorization()` / `conditional_mask()` and `MyEnvPool.query_mask()` (the
  non-advancing batched transport). No built-in env used it yet, and the Python bridges
  (`JaxPool`/`TorchEnvPool`) and the factored `BinPack` env arrived later (see `[Unreleased]`).
  (See git history for the rest of the 0.1.x series.)

[Unreleased]: https://github.com/wiremind/rustpool/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/wiremind/rustpool/releases/tag/v0.1.6
