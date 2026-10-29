use crate::core::types::GenericObs;
use pyo3::prelude::*;

/// Describes one chosen decomposition of an env's flat action space into ordered
/// factors (a "factored action space"). Generic: rustpool transports the per-factor
/// conditional masks (`conditional_mask`) without knowing what the factors mean; the
/// policy composes the chosen factor indices back into the flat action it commits via
/// the normal `step`. `conditional=true` => masks depend on the prefix of already-chosen
/// factors (autoregressive); `false` => independent per-factor masks.
#[derive(Clone, Debug)]
pub struct FactorizationSpec {
    pub name: String,
    pub factor_names: Vec<String>,
    pub factor_widths: Vec<usize>,
    pub pad_width: usize,
    pub flat_action_size: usize,
    pub conditional: bool,
}

// --8<-- [start:trait]
pub trait RlEnv: Send + Sync + RlEnvClone + 'static {
    fn reset(&mut self, seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>);
    fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>);
    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
    fn obs_keys(&self) -> Vec<String>;
    fn query_state(&self, _key: &str) -> Option<String> {
        None
    }
    /// Non-advancing conditional action mask given an env-defined integer `prefix`.
    /// Used by the batched, non-advancing `query_mask` / `query_state_mask` calls. Default
    /// empty so envs that don't support it keep compiling unchanged. `&self`: must not
    /// mutate the env.
    fn conditional_mask(&self, _prefix: &[i32]) -> Vec<bool> {
        Vec::new()
    }
    /// The env's chosen action-space factorization, surfaced to Python via
    /// `RlPool::action_factorization`. `None` (default) => flat action space (no
    /// factoring). Lets the policy build a factored/autoregressive head from the
    /// advertised spec instead of hard-coding the decomposition.
    fn action_factorization(&self) -> Option<FactorizationSpec> {
        None
    }
}
// --8<-- [end:trait]

pub trait RlEnvClone {
    fn clone_box(&self) -> Box<dyn RlEnv>;
}

impl<T: 'static + RlEnv + Clone> RlEnvClone for T {
    fn clone_box(&self) -> Box<dyn RlEnv> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn RlEnv> {
    fn clone(&self) -> Box<dyn RlEnv> {
        self.clone_box()
    }
}
