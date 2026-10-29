//! A complete external env crate in one file: the "Bring your own crate" guide in the docs
//! shows it piece by piece.
//!
//! In your own crate this is `src/lib.rs`, with `crate-type = ["cdylib", "rlib"]` and a
//! dependency on `_rustpool` (see the guide). Here it is a Cargo example built as a `cdylib`,
//! so `cargo test` checks that it compiles against the current engine and
//! `tests/test_external_crate.py` imports it as a real external module.

// --8<-- [start:env]
use _rustpool::{
    ArrayData, AutoReset, Dispatch, EnvFactory, GenericObs, PoolOptions, RlEnv, RlPool,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;

/// A corridor of `length` cells: start at a random cell, reach the right end.
/// Action 0 moves left, 1 moves right; moving left from cell 0 is masked out.
#[derive(Clone)] // Clone gives the pool snapshot / simulate_batch for free
struct Corridor {
    length: i32,
    max_steps: u32,
    pos: i32,
    steps: u32,
    rng: StdRng,
}

impl Corridor {
    fn new(length: i32, max_steps: u32, seed: u64) -> Self {
        Self {
            length,
            max_steps,
            pos: 0,
            steps: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn obs(&self) -> GenericObs {
        // One entry per observation key, in `obs_keys()` order, flattened.
        vec![ArrayData::Int32(vec![self.pos])]
    }

    fn mask(&self) -> Vec<bool> {
        vec![self.pos > 0, true] // [left allowed, right allowed]
    }
}

impl RlEnv for Corridor {
    fn reset(&mut self, seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        self.pos = self.rng.gen_range(0..self.length - 1);
        self.steps = 0;
        (self.obs(), 0.0, false, self.mask())
    }

    fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>) {
        self.pos = if action == 1 {
            self.pos + 1
        } else {
            (self.pos - 1).max(0)
        };
        self.steps += 1;
        let arrived = self.pos == self.length - 1;
        let done = arrived || self.steps >= self.max_steps;
        (
            self.obs(),
            if arrived { 1.0 } else { 0.0 },
            done,
            self.mask(),
        )
    }

    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let spec = PyDict::new(py);
        spec.set_item("pos", (vec![1], "int32"))?; // {key: (shape, dtype)}
        Ok(spec.into_any().unbind())
    }

    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok((2,).into_pyobject(py)?.into_any().unbind()) // (num_actions,)
    }

    fn obs_keys(&self) -> Vec<String> {
        vec!["pos".to_string()]
    }
}
// --8<-- [end:env]

// --8<-- [start:module]
/// `external_env.make(...)`: the same pool options as `rustpool.make`.
#[pyfunction]
#[pyo3(signature = (num_envs=4, batch_size=4, seed=0, *, num_threads=None,
                    dispatch="auto", autoreset="same_step", length=8, max_steps=50))]
#[allow(clippy::too_many_arguments)]
fn make(
    num_envs: usize,
    batch_size: usize,
    seed: u64,
    num_threads: Option<usize>,
    dispatch: &str,
    autoreset: &str,
    length: i32,
    max_steps: u32,
) -> PyResult<RlPool> {
    if length < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "length must be >= 2",
        ));
    }
    // Parses the strings exactly like rustpool.make (ValueError on a bad value).
    let options = PoolOptions::parse(seed, num_threads, dispatch, autoreset)?;
    // Env i is built as factory(seed + i), as in rustpool.make.
    let factory: EnvFactory =
        Box::new(move |env_seed| Box::new(Corridor::new(length, max_steps, env_seed)));
    RlPool::new_with_options(num_envs, batch_size, factory, options)
}

// --8<-- [start:options]
/// The options as a Rust value, without strings: fields you leave out keep their defaults.
pub fn fixed_options() -> PoolOptions {
    PoolOptions {
        seed: 42,
        autoreset: AutoReset::NextStep,
        dispatch: Dispatch::PerWorker,
        ..PoolOptions::default() // num_threads: None -> RUSTPOOL_NUM_THREADS, then the CPUs
    }
}
// RlPool::new_with_factory(num_envs, batch_size, factory) still works: it is
// new_with_options with PoolOptions::default() (seed 0, auto dispatch, same_step).
// --8<-- [end:options]

#[pymodule]
fn external_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RlPool>()?; // the pool's full Python API, under your module
    m.add_function(wrap_pyfunction!(make, m)?)?;
    Ok(())
}
// --8<-- [end:module]
