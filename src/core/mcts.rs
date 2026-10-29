use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::core::rl_env::RlEnv;

pub struct StateRegistry {
    // Arc so read-only queries (`conditional_masks`) can take cheap shared handles and run
    // OUTSIDE the lock; `get_clones` still deep-clones (simulate_batch mutates its copies).
    cloned_states: Mutex<HashMap<usize, Arc<Box<dyn RlEnv>>>>,
    next_state_id: Mutex<usize>,
}

impl StateRegistry {
    pub fn new() -> Self {
        Self {
            cloned_states: Mutex::new(HashMap::new()),
            next_state_id: Mutex::new(0),
        }
    }

    pub fn snapshot(&self, envs: Vec<Box<dyn RlEnv>>) -> Vec<i32> {
        let mut state_ids = Vec::with_capacity(envs.len());
        let mut cloned_states = self.cloned_states.lock().unwrap();
        let mut next_id = self.next_state_id.lock().unwrap();

        for env in envs {
            let state_id = *next_id;
            // Handles are surfaced to Python as i32 (and -1 is the release sentinel), so refuse
            // to allocate past i32::MAX rather than silently aliasing live/-1 handles.
            assert!(
                state_id < i32::MAX as usize,
                "StateRegistry exhausted the i32 handle space (>2^31-1 allocations); \
                 release more aggressively or restart the registry"
            );
            *next_id += 1;
            cloned_states.insert(state_id, Arc::new(env));
            state_ids.push(state_id as i32);
        }

        state_ids
    }

    pub fn get_clones(&self, state_ids: &[i32]) -> Result<Vec<Box<dyn RlEnv>>, String> {
        let cloned_states = self.cloned_states.lock().unwrap();
        let mut environments = Vec::with_capacity(state_ids.len());

        for &state_id in state_ids {
            let env = cloned_states
                .get(&(state_id as usize))
                .ok_or_else(|| format!("invalid state_id {} (unknown or released)", state_id))?;
            environments.push((**env).clone());
        }

        Ok(environments)
    }

    /// Non-advancing per-factor conditional masks of stored states: row `i` is
    /// `conditional_mask(prefixes[i])` of `state_ids[i]`, in input order (an id may repeat),
    /// computed in parallel without cloning any state. A negative `state_id` (the `release`
    /// sentinel, used for dead / finished search rows) yields an EMPTY row without a registry
    /// lookup; an unknown or already-released id >= 0 is an `Err`. The lock is held only while
    /// the shared handles are collected, so a concurrent `release` of the same id keeps that
    /// state alive until its mask is computed.
    pub fn conditional_masks(
        &self,
        state_ids: &[i32],
        prefixes: &[&[i32]],
    ) -> Result<Vec<Vec<bool>>, String> {
        use rayon::prelude::*;
        if state_ids.len() != prefixes.len() {
            return Err(format!(
                "prefixes rows ({}) must match state_ids length ({})",
                prefixes.len(),
                state_ids.len()
            ));
        }
        let envs: Vec<Option<Arc<Box<dyn RlEnv>>>> = {
            let cloned_states = self.cloned_states.lock().unwrap();
            let mut envs = Vec::with_capacity(state_ids.len());
            for &state_id in state_ids {
                if state_id < 0 {
                    envs.push(None);
                    continue;
                }
                let env = cloned_states.get(&(state_id as usize)).ok_or_else(|| {
                    format!("invalid state_id {} (unknown or released)", state_id)
                })?;
                envs.push(Some(Arc::clone(env)));
            }
            envs
        };
        Ok(envs
            .par_iter()
            .zip(prefixes.par_iter())
            .map(|(env, prefix)| match env {
                Some(env) => env.conditional_mask(prefix),
                None => Vec::new(),
            })
            .collect())
    }

    pub fn release(&self, state_ids: &[i32]) {
        let mut cloned_states = self.cloned_states.lock().unwrap();
        for &state_id in state_ids {
            cloned_states.remove(&(state_id as usize));
        }
    }

    /// Drop every stored state (the pool is closing).
    pub fn clear(&self) {
        self.cloned_states.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::GenericObs;
    use pyo3::prelude::*;

    /// Minimal env for registry tests: `step(a)` adds `a` to `value`; `conditional_mask(p)` is
    /// a one-hot of width 4 at `(value + p[0]) mod 4`. The Python-facing spec methods are never
    /// called here (and must not reference Python symbols, so the test binary links without
    /// libpython).
    #[derive(Clone)]
    struct CounterEnv {
        value: i32,
    }

    impl RlEnv for CounterEnv {
        fn reset(&mut self, _seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>) {
            self.value = 0;
            (Vec::new(), 0.0, false, Vec::new())
        }
        fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>) {
            self.value += action;
            (Vec::new(), 0.0, false, Vec::new())
        }
        fn observation_spec(&self, _py: Python<'_>) -> PyResult<Py<PyAny>> {
            unreachable!("not used by the registry tests")
        }
        fn action_spec(&self, _py: Python<'_>) -> PyResult<Py<PyAny>> {
            unreachable!("not used by the registry tests")
        }
        fn obs_keys(&self) -> Vec<String> {
            Vec::new()
        }
        fn conditional_mask(&self, prefix: &[i32]) -> Vec<bool> {
            let hot = (self.value + prefix[0]).rem_euclid(4) as usize;
            (0..4).map(|j| j == hot).collect()
        }
    }

    fn one_hot(hot: usize) -> Vec<bool> {
        (0..4).map(|j| j == hot).collect()
    }

    fn registry_with(values: &[i32]) -> (StateRegistry, Vec<i32>) {
        let registry = StateRegistry::new();
        let envs: Vec<Box<dyn RlEnv>> = values
            .iter()
            .map(|&value| Box::new(CounterEnv { value }) as Box<dyn RlEnv>)
            .collect();
        let ids = registry.snapshot(envs);
        (registry, ids)
    }

    #[test]
    fn conditional_masks_input_order_and_duplicates() {
        let (registry, ids) = registry_with(&[0, 1, 2]);
        let p0: &[i32] = &[0, -1, -1];
        let p1: &[i32] = &[1, -1, -1];
        let state_ids = [ids[2], ids[0], ids[2], ids[1]];
        let prefixes = [p0, p0, p1, p0];
        let masks = registry.conditional_masks(&state_ids, &prefixes).unwrap();
        assert_eq!(masks, vec![one_hot(2), one_hot(0), one_hot(3), one_hot(1)]);
    }

    #[test]
    fn conditional_masks_negative_ids_are_empty_without_lookup() {
        let (registry, ids) = registry_with(&[3]);
        let p: &[i32] = &[0, -1, -1];
        // -1 and i32::MIN are never looked up (an empty registry would otherwise Err).
        let masks = registry
            .conditional_masks(&[-1, ids[0], i32::MIN], &[p, p, p])
            .unwrap();
        assert_eq!(masks, vec![Vec::new(), one_hot(3), Vec::new()]);
        let empty = StateRegistry::new();
        assert_eq!(
            empty.conditional_masks(&[-1, -5], &[p, p]).unwrap(),
            vec![Vec::<bool>::new(), Vec::new()]
        );
    }

    #[test]
    fn conditional_masks_unknown_or_released_id_is_an_error() {
        let (registry, ids) = registry_with(&[0, 1]);
        let p: &[i32] = &[0, -1, -1];
        let err = registry
            .conditional_masks(&[ids[0], 12345], &[p, p])
            .unwrap_err();
        assert!(err.contains("invalid state_id 12345"), "{}", err);
        registry.release(&[ids[1]]);
        let err = registry.conditional_masks(&[ids[1]], &[p]).unwrap_err();
        assert!(
            err.contains(&format!("invalid state_id {}", ids[1])),
            "{}",
            err
        );
        // The surviving state is untouched.
        assert_eq!(
            registry.conditional_masks(&[ids[0]], &[p]).unwrap(),
            vec![one_hot(0)]
        );
    }

    #[test]
    fn conditional_masks_rejects_length_mismatch() {
        let (registry, ids) = registry_with(&[0]);
        let p: &[i32] = &[0, -1, -1];
        assert!(registry.conditional_masks(&[ids[0], ids[0]], &[p]).is_err());
    }

    #[test]
    fn conditional_masks_read_stored_state_and_do_not_mutate_it() {
        let (registry, ids) = registry_with(&[0]);
        let p: &[i32] = &[0, -1, -1];
        // A stepped clone (what simulate_batch stores) reports its own state...
        let mut child = registry.get_clones(&ids).unwrap().pop().unwrap();
        let _ = child.step(2);
        let child_id = registry.snapshot(vec![child])[0];
        let masks = registry
            .conditional_masks(&[ids[0], child_id], &[p, p])
            .unwrap();
        assert_eq!(masks, vec![one_hot(0), one_hot(2)]);
        // ...and querying is non-advancing: the parent still clones to value 0.
        let mut again = registry.get_clones(&ids).unwrap().pop().unwrap();
        let _ = again.step(1);
        let again_id = registry.snapshot(vec![again])[0];
        assert_eq!(
            registry.conditional_masks(&[again_id], &[p]).unwrap(),
            vec![one_hot(1)]
        );
    }
}
