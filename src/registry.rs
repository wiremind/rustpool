// src/registry.rs

use crate::core::rl_env::RlEnv;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

pub type EnvFactory = Box<dyn Fn(u64) -> Box<dyn RlEnv> + Send + Sync>;
pub type EnvParser = fn(&Bound<'_, PyDict>) -> PyResult<EnvFactory>;

fn build_registry() -> HashMap<String, EnvParser> {
    let mut m = HashMap::new();
    crate::envs::binpack::register_env(&mut m);
    crate::envs::cartpole::register_env(&mut m);
    crate::envs::maze::register_env(&mut m);
    m
}

static ENV_REGISTRY: Lazy<HashMap<String, EnvParser>> = Lazy::new(build_registry);

/// Task ids registered in the built-in environment registry, sorted (e.g. `BinPack-v0`).
pub fn registered_task_ids() -> Vec<String> {
    let mut ids: Vec<String> = ENV_REGISTRY.keys().cloned().collect();
    ids.sort();
    ids
}

pub fn create_env_factory(task_id: &str, kwargs: &Bound<'_, PyDict>) -> PyResult<EnvFactory> {
    match ENV_REGISTRY.get(task_id) {
        Some(parser) => parser(kwargs),
        None => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown task_id: {}",
            task_id
        ))),
    }
}

/// Deserialise an environment's kwargs into its config struct (serde, through `pythonize`). A
/// serde error (an unknown key on a `#[serde(deny_unknown_fields)]` struct, a mistyped value)
/// becomes `TypeError("<task_id>: ...")`, as for a Python function given a bad keyword argument.
pub fn parse_env_kwargs<T: serde::de::DeserializeOwned>(
    task_id: &str,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<T> {
    pythonize::depythonize(kwargs.as_any())
        .map_err(|e| PyTypeError::new_err(format!("{task_id}: {e}")))
}
