# Contributing to rustpool

Thanks for your interest! rustpool is a Rust extension (PyO3 + maturin) with a Python API.

## Development setup

```bash
uv venv && source .venv/bin/activate     # (with python -m venv .venv: plain pip install below)
uv pip install -e ".[examples,dev]"      # maturin, pytest, ruff, gymnasium, jax/flax/optax/mctx, torch
maturin develop                          # (re)build the Rust extension in-place after Rust edits
```

`maturin develop` recompiles `src/` into `rustpool/_rustpool.*.so`. Re-run it whenever you change
Rust code; pure-Python and example changes need no rebuild.

## Running the checks

```bash
cargo test                 # Rust unit tests (src/)
cargo fmt --check          # Rust formatting (required in CI)
cargo clippy --all-targets # advisory in CI: the tree is not warning-free yet
pytest                     # Python tests (tests/)
ruff check .               # Python lint
```

The example smoke tests (`tests/test_examples_smoke.py`) are skipped unless the optional
`examples` / `torch` stacks are installed; they run each example for a few steps on CPU.

## Adding an environment

Environments are pure Rust implementing the `RlEnv` trait (`src/core/rl_env.rs`):

1. Create `src/envs/<your_env>/mod.rs`. Implement `RlEnv` (`reset`, `step`, `observation_spec`,
   `action_spec`, `obs_keys`) on your env struct, plus a `#[serde(deny_unknown_fields)]`
   `Deserialize` config, a `parse_<env>_config(kwargs) -> PyResult<EnvFactory>` parser that reads
   it with `registry::parse_env_kwargs` and raises `ValueError` for out-of-range values, and a
   `register_env(&mut registry)` fn. Derive `Clone` on the env to get the
   `snapshot`/`simulate_batch` (MCTS) API for free.
2. Declare the module in `src/envs/mod.rs` (`pub mod <your_env>;`) and register it in
   `src/registry.rs::build_registry`.
3. Add it to the docs (`docs/environments.md`) and tests in `tests/`.
4. `maturin develop`, then `rustpool.list_envs()` should show your `task_id`.

Optional capabilities are all defaulted on the trait, so opt in only if you need them:
`query_state` (JSON escape hatch), `action_factorization` + `conditional_mask` (factored
autoregressive masks).

To build an env in a *separate crate* without touching rustpool, depend on `_rustpool` and use
`RlPool::new_with_options` (see the README "Writing your own environment" section).

## Coding conventions

- Match the surrounding style; keep the worker threads pure Rust (no Python object (`Py<PyAny>`) across channels —
  NumPy conversion happens centrally in `RlPool::recv`).
- Public PyO3 methods get `///` doc comments (they become Python docstrings) and, on `#[new]`,
  a `text_signature`. Constructor parameters are documented in the `RlPool` struct's doc
  comment (the class docstring), since PyO3 does not expose `#[new]`'s. Update
  `rustpool/_rustpool.pyi` when you change the Python-visible surface.
- Documentation lives in `docs/` (MkDocs + Material). Build it with the extension installed, so
  the API page can read the compiled module's docstrings:
  `maturin develop --extras docs && mkdocs build --strict` (output in the gitignored `site/`).
  `tests/test_docs_snippets.py` runs every Python snippet in `docs/`; Rust code on the guide pages
  is included from compiled files (`--8<--` sections in `src/` and `examples/rust/`).
- Keep `import rustpool` lightweight — do not import JAX/PyTorch/gymnasium at package import time.

## Releasing

Bump `[package].version` in `Cargo.toml` (the single source of truth: `pyproject.toml` reads it as
a dynamic version), update `CHANGELOG.md`, commit, then push a matching tag `vX.Y.Z` (that tag
only, never `--tags`). `.github/workflows/release.yml` then checks that the tag matches
`Cargo.toml`, runs the tests, builds manylinux2014 wheels for CPython 3.11 to 3.14 and the sdist,
tests each of them installed, and publishes to PyPI through Trusted Publishing.

## License

By contributing you agree your contributions are dual-licensed under MIT OR Apache-2.0.
