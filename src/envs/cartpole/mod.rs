//! `CartPole-v1`: the classic cart-pole balancing task, a port of gymnasium's
//! `CartPoleEnv` (Euler integration, f64 state) with its 500-step `TimeLimit`.
//! A near-zero-compute env for measuring engine overhead.

use crate::core::rl_env::RlEnv;
use crate::core::types::{ArrayData, GenericObs};
use crate::registry::{parse_env_kwargs, EnvFactory, EnvParser};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

// gymnasium.envs.classic_control.cartpole.CartPoleEnv constants, derived the same way.
const GRAVITY: f64 = 9.8;
const MASSCART: f64 = 1.0;
const MASSPOLE: f64 = 0.1;
const TOTAL_MASS: f64 = MASSPOLE + MASSCART;
const LENGTH: f64 = 0.5; // half the pole's length
const POLEMASS_LENGTH: f64 = MASSPOLE * LENGTH;
const FORCE_MAG: f64 = 10.0;
const TAU: f64 = 0.02;
const THETA_THRESHOLD: f64 = 12.0 * 2.0 * std::f64::consts::PI / 360.0;
const X_THRESHOLD: f64 = 2.4;
const RESET_BOUND: f64 = 0.05;

// --8<-- [start:config]
fn default_max_episode_steps() -> u32 {
    500
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct CartPoleKwargs {
    #[serde(default = "default_max_episode_steps")]
    max_episode_steps: u32,
}

fn parse_cartpole_config(kwargs: &Bound<'_, PyDict>) -> PyResult<EnvFactory> {
    let cfg: CartPoleKwargs = parse_env_kwargs("CartPole-v1", kwargs)?;
    if cfg.max_episode_steps == 0 {
        return Err(PyValueError::new_err("max_episode_steps must be >= 1"));
    }
    Ok(Box::new(move |seed| {
        Box::new(CartPoleEnv::new(cfg.max_episode_steps, seed))
    }))
}

pub fn register_env(registry: &mut HashMap<String, EnvParser>) {
    registry.insert("CartPole-v1".to_string(), parse_cartpole_config);
}
// --8<-- [end:config]

#[derive(Clone)]
pub struct CartPoleEnv {
    max_episode_steps: u32,
    rng: StdRng,
    /// `(x, x_dot, theta, theta_dot)`, f64 like gymnasium's `self.state`.
    state: [f64; 4],
    steps: u32,
}

impl CartPoleEnv {
    const OBS_KEY: &'static str = "obs";

    pub fn new(max_episode_steps: u32, seed: u64) -> Self {
        Self {
            max_episode_steps,
            rng: StdRng::seed_from_u64(seed),
            state: [0.0; 4],
            steps: 0,
        }
    }

    /// One Euler step of gymnasium's dynamics; `action == 1` pushes right, anything else left.
    fn integrate(state: [f64; 4], action: i32) -> [f64; 4] {
        let [x, x_dot, theta, theta_dot] = state;
        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let temp = (force + POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) / TOTAL_MASS;
        let thetaacc = (GRAVITY * sintheta - costheta * temp)
            / (LENGTH * (4.0 / 3.0 - MASSPOLE * (costheta * costheta) / TOTAL_MASS));
        let xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;
        [
            x + TAU * x_dot,
            x_dot + TAU * xacc,
            theta + TAU * theta_dot,
            theta_dot + TAU * thetaacc,
        ]
    }

    fn obs(&self) -> GenericObs {
        vec![ArrayData::Float32(
            self.state.iter().map(|&v| v as f32).collect(),
        )]
    }
}

// --8<-- [start:impl]
impl RlEnv for CartPoleEnv {
    fn reset(&mut self, seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        for v in self.state.iter_mut() {
            *v = self.rng.gen_range(-RESET_BOUND..RESET_BOUND);
        }
        self.steps = 0;
        (self.obs(), 0.0, false, Vec::new())
    }

    fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>) {
        self.state = Self::integrate(self.state, action);
        self.steps += 1;
        let [x, _, theta, _] = self.state;
        let terminated = !(-X_THRESHOLD..=X_THRESHOLD).contains(&x)
            || !(-THETA_THRESHOLD..=THETA_THRESHOLD).contains(&theta);
        let truncated = self.steps >= self.max_episode_steps;
        // gymnasium pays 1.0 on every step, the terminating one included.
        (self.obs(), 1.0, terminated || truncated, Vec::new())
    }

    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let spec = PyDict::new(py);
        spec.set_item(Self::OBS_KEY, (vec![4], "float32"))?;
        Ok(spec.into_any().unbind())
    }

    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok((2,).into_pyobject(py)?.into_any().unbind())
    }

    fn obs_keys(&self) -> Vec<String> {
        vec![Self::OBS_KEY.to_string()]
    }

    /// `"state"`: the exact f64 `[x, x_dot, theta, theta_dot]` (round-trip JSON);
    /// `"steps"`: steps taken in the current episode. A test hook for dynamics parity.
    fn query_state(&self, key: &str) -> Option<String> {
        match key {
            "state" => serde_json::to_string(&self.state).ok(),
            "steps" => Some(self.steps.to_string()),
            _ => None,
        }
    }
}
// --8<-- [end:impl]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_bounds_and_determinism() {
        let mut a = CartPoleEnv::new(500, 3);
        let mut b = CartPoleEnv::new(500, 3);
        a.reset(Some(9));
        b.reset(Some(9));
        assert_eq!(a.state, b.state);
        assert!(a.state.iter().all(|v| v.abs() < RESET_BOUND));
    }

    #[test]
    fn test_pushing_one_way_terminates_before_truncation() {
        let mut env = CartPoleEnv::new(500, 0);
        env.reset(None);
        let mut steps = 0;
        loop {
            steps += 1;
            let (_, reward, done, mask) = env.step(1);
            assert_eq!(reward, 1.0);
            assert!(mask.is_empty());
            if done {
                break;
            }
        }
        assert!(steps < 500);
        let theta = env.state[2];
        assert!(theta.abs() > THETA_THRESHOLD || env.state[0].abs() > X_THRESHOLD);
    }

    #[test]
    fn test_truncation() {
        let mut env = CartPoleEnv::new(5, 0);
        env.reset(None);
        env.state = [0.0; 4];
        let dones: Vec<bool> = (0..5).map(|i| env.step(i % 2).2).collect();
        assert_eq!(dones, vec![false, false, false, false, true]);
    }
}
