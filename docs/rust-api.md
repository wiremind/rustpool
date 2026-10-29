# Rust API

What an environment crate uses from `_rustpool` (rustpool's Rust crate). Each signature below
is checked against the source by the test suite. For the full documentation, run
`cargo doc --open` in a checkout.

## The environment trait

```rust
pub trait RlEnv: Send + Sync + RlEnvClone + 'static {
    fn reset(&mut self, seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>);
    fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>);
    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
    fn obs_keys(&self) -> Vec<String>;
    // Optional, with defaults:
    fn query_state(&self, _key: &str) -> Option<String>;
    fn conditional_mask(&self, _prefix: &[i32]) -> Vec<bool>;
    fn action_factorization(&self) -> Option<FactorizationSpec>;
}
```

`RlEnvClone` is implemented for every `RlEnv + Clone` type: derive `Clone` and the pool's search
calls work. The contract of each method is in
[Write an environment inside rustpool](guides/env-in-rustpool.md#1-implement-rlenv).

## Observations

```rust
pub enum ArrayData {
    Float32(Vec<f32>),
    Int32(Vec<i32>),
    Bool(Vec<bool>),
}
pub type GenericObs = Vec<ArrayData>;
```

One `ArrayData` per observation key, in `obs_keys()` order, flattened row-major to the shape in
`observation_spec()`.

## Factored action spaces

```rust
pub struct FactorizationSpec {
    pub name: String,
    pub factor_names: Vec<String>,
    pub factor_widths: Vec<usize>,
    pub pad_width: usize,
    pub flat_action_size: usize,
    pub conditional: bool,
}
```

`conditional_mask(prefix)` answers the mask over the next factor, given the factors already
chosen in `prefix` (`-1` = unset), `pad_width` wide. See [Factored action masks](concepts/factored-actions.md).

## Building a pool

```rust
pub type EnvFactory = Box<dyn Fn(u64) -> Box<dyn RlEnv> + Send + Sync>;

pub struct PoolOptions {
    pub seed: u64,
    pub num_threads: Option<usize>,
    pub dispatch: Dispatch,
    pub autoreset: AutoReset,
}
pub fn parse(seed: u64, num_threads: Option<usize>, dispatch: &str, autoreset: &str) -> PyResult<Self>

pub fn new_with_options(num_envs: usize, batch_size: usize, factory: EnvFactory, options: PoolOptions) -> PyResult<Self>
pub fn new_with_factory(num_envs: usize, batch_size: usize, factory: EnvFactory) -> PyResult<Self>
pub fn new_with_factory_and_threads(num_envs: usize, batch_size: usize, factory: EnvFactory, num_threads: Option<usize>) -> PyResult<Self>

pub enum Dispatch {
    Auto,
    PerEnv,
    PerWorker,
}
pub enum AutoReset {
    SameStep,
    NextStep,
}
```

- `PoolOptions::default()`: seed 0, `num_threads: None` (then `RUSTPOOL_NUM_THREADS`, then the
  available CPUs; always capped at `num_envs`), `Dispatch::Auto`, `AutoReset::SameStep`.
- `PoolOptions::parse` takes the Python strings (`"auto"`, `"per_env"`, `"per_worker"`;
  `"same_step"`, `"next_step"`) and raises `ValueError` on anything else, like `rustpool.make`.
- `RlPool::new_with_options` builds env `i` as `factory(options.seed + i)`, and the
  environment used for the specs as `factory(options.seed)`.
- `new_with_factory` and `new_with_factory_and_threads` are kept for existing crates: they are
  `new_with_options` with the default options (and the given thread count).

Export the pool to Python from your own `#[pymodule]` with `m.add_class::<RlPool>()`; see
[Bring your own crate](guides/own-crate.md).
