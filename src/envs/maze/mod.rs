use crate::core::rl_env::RlEnv;
use crate::core::types::{ArrayData, GenericObs};
use crate::registry::{parse_env_kwargs, EnvFactory, EnvParser};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::TryFrom;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
pub struct MazeConfig {
    pub width: i32,
    pub height: i32,
    pub max_episode_steps: i32,
}

fn default_width() -> i32 {
    10
}
fn default_height() -> i32 {
    10
}
fn default_max_episode_steps() -> i32 {
    100
}

impl Default for MazeConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            max_episode_steps: default_max_episode_steps(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MazeKwargs {
    #[serde(default = "default_width")]
    width: i32,
    #[serde(default = "default_height")]
    height: i32,
    #[serde(default = "default_max_episode_steps")]
    max_episode_steps: i32,
}

fn parse_maze_config(kwargs: &Bound<'_, PyDict>) -> PyResult<EnvFactory> {
    let cfg: MazeKwargs = parse_env_kwargs("Maze-v0", kwargs)?;
    if cfg.width < 1 || cfg.height < 1 {
        return Err(PyValueError::new_err("width and height must be >= 1"));
    }
    if cfg.max_episode_steps < 1 {
        return Err(PyValueError::new_err("max_episode_steps must be >= 1"));
    }
    let config = MazeConfig {
        width: cfg.width,
        height: cfg.height,
        max_episode_steps: cfg.max_episode_steps,
    };

    Ok(Box::new(move |seed| Box::new(MazeEnv::new(config, seed))))
}

pub fn register_env(registry: &mut HashMap<String, EnvParser>) {
    registry.insert("Maze-v0".to_string(), parse_maze_config);
}

#[derive(Clone)]
pub struct MazeEnv {
    config: MazeConfig,
    rng: StdRng,
    agent_pos: (i32, i32),
    goal_pos: (i32, i32),
    steps: i32,
}

impl MazeEnv {
    const OBS_KEYS: [&'static str; 1] = ["agent_pos"];

    pub fn new(config: MazeConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
            agent_pos: (0, 0), // Initialized during reset
            goal_pos: (config.width - 1, config.height - 1),
            steps: 0,
        }
    }

    fn action_mask_for(&self, pos: (i32, i32)) -> [bool; 4] {
        let (x, y) = pos;
        [
            y > 0,                      // Up
            y < self.config.height - 1, // Down
            x > 0,                      // Left
            x < self.config.width - 1,  // Right
        ]
    }

    fn get_obs_and_mask(&self) -> (GenericObs, Vec<bool>) {
        // Observation: agent's position
        let agent_pos = vec![self.agent_pos.0 as f32, self.agent_pos.1 as f32];
        let action_mask = self.action_mask_for(self.agent_pos).to_vec();

        let mut obs = Vec::with_capacity(Self::OBS_KEYS.len());
        obs.push(ArrayData::Float32(agent_pos));

        (obs, action_mask)
    }
}

impl RlEnv for MazeEnv {
    fn reset(&mut self, seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>) {
        // Re-seed RNG if seed is provided
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        // Reset agent to a random position
        self.agent_pos = (
            self.rng.gen_range(0..self.config.width),
            self.rng.gen_range(0..self.config.height),
        );
        self.steps = 0;

        let (obs, action_mask) = self.get_obs_and_mask();
        (obs, 0.0, false, action_mask)
    }

    fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>) {
        self.steps += 1;

        let current_mask = self.action_mask_for(self.agent_pos);

        let action_idx = usize::try_from(action).ok();
        let valid_action =
            matches!(action_idx, Some(idx) if idx < current_mask.len() && current_mask[idx]);

        if valid_action {
            match action {
                0 => self.agent_pos.1 -= 1, // Up
                1 => self.agent_pos.1 += 1, // Down
                2 => self.agent_pos.0 -= 1, // Left
                3 => self.agent_pos.0 += 1, // Right
                _ => unreachable!(),
            }
        }

        let reached_goal = self.agent_pos == self.goal_pos;
        let truncated = self.steps >= self.config.max_episode_steps;
        let invalid_action = !valid_action;
        let done = reached_goal || truncated || invalid_action;
        let reward = if reached_goal { 1.0 } else { 0.0 };

        let (obs, action_mask) = self.get_obs_and_mask();
        (obs, reward, done, action_mask)
    }

    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obs_spec = PyDict::new(py);
        let float32_str = "float32";
        obs_spec.set_item("agent_pos", (vec![2], float32_str))?;
        Ok(obs_spec.into_any().unbind())
    }

    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let num_actions = 4; // Up, Down, Left, Right
        Ok((num_actions,).into_pyobject(py)?.into_any().unbind())
    }

    fn obs_keys(&self) -> Vec<String> {
        Self::OBS_KEYS.iter().map(|k| (*k).to_string()).collect()
    }
}
