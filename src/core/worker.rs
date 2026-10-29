use super::rl_env::RlEnv;
use super::types::GenericObs;
use crossbeam_channel::{Receiver, Sender};
use std::collections::HashSet;

/// Message from the calling thread to a worker.
pub enum WorkerAction {
    Step(usize, i32), // (env_id, action)
    /// Every (env_id, action) of one `send` that this worker owns (per-worker dispatch).
    StepBatch(Vec<(usize, i32)>),
    Reset(Option<Vec<(usize, u64)>>), // Optional list of (env_id, seed) pairs
    Snapshot(Vec<usize>),             // Request snapshots for specific env_ids
    Query(usize, String),
    QueryMask(usize, usize, Vec<i32>), // (env_id, row_idx, prefix) -> non-advancing conditional mask
    /// Every `QueryMask` row of one `query_mask` call that this worker owns (per-worker
    /// dispatch).
    QueryMaskBatch(Vec<(usize, usize, Vec<i32>)>),
    Close,
}

/// Message from a worker back to the calling thread.
pub enum WorkerMessage {
    StepResult {
        env_id: usize,
        obs: GenericObs,
        reward: f32,
        done: bool,
        action_mask: Vec<bool>,
    },
    SnapshotResult {
        env_id: usize,
        snapshot: Box<dyn RlEnv>,
    },
    QueryResult {
        env_id: usize,
        result: Option<String>,
    },
    MaskResult {
        env_id: usize,
        row_idx: usize,
        mask: Vec<bool>,
    },
    /// The results of one `StepBatch`, in its order.
    StepResults(Vec<StepOut>),
    /// The `(env_id, row_idx, mask)` answers to one `QueryMaskBatch`.
    MaskResults(Vec<(usize, usize, Vec<bool>)>),
}

/// One env's step result.
pub struct StepOut {
    pub env_id: usize,
    pub obs: GenericObs,
    pub reward: f32,
    pub done: bool,
    pub action_mask: Vec<bool>,
}

impl From<StepOut> for WorkerMessage {
    fn from(o: StepOut) -> Self {
        WorkerMessage::StepResult {
            env_id: o.env_id,
            obs: o.obs,
            reward: o.reward,
            done: o.done,
            action_mask: o.action_mask,
        }
    }
}

/// How `send` and `query_mask` hand work to the workers and how results come back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dispatch {
    /// Per call: `PerWorker` when the call has more rows than there are workers, else `PerEnv`
    /// (one message per row is cheaper when each worker gets at most about one row anyway).
    #[default]
    Auto,
    /// One channel message (and wake-up) per env each way.
    PerEnv,
    /// One message per worker each way per call: the worker handles all its rows of the call,
    /// then returns their results together. Channel operations per call drop from O(rows) to
    /// O(threads).
    PerWorker,
}

impl Dispatch {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Dispatch::Auto),
            "per_env" => Some(Dispatch::PerEnv),
            "per_worker" => Some(Dispatch::PerWorker),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dispatch::Auto => "auto",
            Dispatch::PerEnv => "per_env",
            Dispatch::PerWorker => "per_worker",
        }
    }

    /// Whether a call of `rows` rows over `num_threads` workers is sent per worker.
    pub fn batches(self, rows: usize, num_threads: usize) -> bool {
        match self {
            Dispatch::Auto => rows > num_threads,
            Dispatch::PerEnv => false,
            Dispatch::PerWorker => true,
        }
    }
}

/// When an episode ends, which call returns the reset observation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoReset {
    /// rustpool's default: the terminal step resets at once and returns the new episode's first
    /// observation with the terminal reward and `done=True` (one call per transition).
    #[default]
    SameStep,
    /// EnvPool's convention: the terminal step returns the terminal observation with
    /// `done=True`; the env's next step resets it (its action is ignored) and returns the
    /// first observation with reward 0 and `done=False` (one extra call per episode).
    NextStep,
}

impl AutoReset {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "same_step" => Some(AutoReset::SameStep),
            "next_step" => Some(AutoReset::NextStep),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            AutoReset::SameStep => "same_step",
            AutoReset::NextStep => "next_step",
        }
    }
}

/// Step one env. `same_step`: on a terminal step reset it at once and return the hybrid state,
/// the terminal reward and `done` with the next episode's first observation and action mask.
/// `next_step`: return the terminal state as is and mark the env in `pending_reset`, so that its
/// next step is the reset instead.
fn step_env(
    env: &mut Box<dyn RlEnv>,
    env_id: usize,
    action: i32,
    autoreset: AutoReset,
    pending_reset: &mut HashSet<usize>,
) -> StepOut {
    if autoreset == AutoReset::NextStep && pending_reset.remove(&env_id) {
        let (obs, _, _, action_mask) = env.reset(None);
        return StepOut {
            env_id,
            obs,
            reward: 0.0,
            done: false,
            action_mask,
        };
    }
    let (obs, reward, done, action_mask) = env.step(action);
    if done && autoreset == AutoReset::NextStep {
        pending_reset.insert(env_id);
    } else if done {
        let (reset_obs, _, _, reset_action_mask) = env.reset(None);
        return StepOut {
            env_id,
            obs: reset_obs,
            reward,
            done: true,
            action_mask: reset_action_mask,
        };
    }
    StepOut {
        env_id,
        obs,
        reward,
        done,
        action_mask,
    }
}

pub fn worker_loop(
    _shard_id: usize,
    mut envs: std::collections::HashMap<usize, Box<dyn RlEnv>>,
    action_rx: Receiver<WorkerAction>,
    state_tx: Sender<WorkerMessage>,
    autoreset: AutoReset,
) {
    // Envs whose last step ended an episode (`next_step` only): their next step resets them.
    let mut pending_reset: HashSet<usize> = HashSet::new();
    loop {
        match action_rx.recv() {
            Ok(WorkerAction::Step(target_env_id, action)) => {
                // Look up the specific environment in this shard
                if let Some(env) = envs.get_mut(&target_env_id) {
                    let out = step_env(env, target_env_id, action, autoreset, &mut pending_reset);
                    if state_tx.send(out.into()).is_err() {
                        break; // Stop if the main thread has disconnected
                    }
                }
                // If env_id not found in this shard, silently ignore
                // (this shouldn't happen with correct routing)
            }

            Ok(WorkerAction::StepBatch(batch)) => {
                let outs: Vec<StepOut> = batch
                    .into_iter()
                    .filter_map(|(env_id, action)| {
                        envs.get_mut(&env_id)
                            .map(|env| step_env(env, env_id, action, autoreset, &mut pending_reset))
                    })
                    .collect();
                if state_tx.send(WorkerMessage::StepResults(outs)).is_err() {
                    break;
                }
            }

            Ok(WorkerAction::Reset(seed_map_opt)) => {
                // Build a seed lookup for this shard's environments
                let seed_lookup: std::collections::HashMap<usize, u64> = seed_map_opt
                    .as_ref()
                    .map(|seeds| seeds.iter().cloned().collect())
                    .unwrap_or_default();

                // Reset all environments in this shard
                for (env_id, env) in envs.iter_mut() {
                    // Check if this environment has an explicit seed
                    let seed = seed_lookup.get(env_id).copied();
                    pending_reset.remove(env_id);
                    let (obs, _, done, action_mask) = env.reset(seed);

                    if state_tx
                        .send(WorkerMessage::StepResult {
                            env_id: *env_id,
                            obs,
                            reward: 0.0,
                            done,
                            action_mask,
                        })
                        .is_err()
                    {
                        return; // Stop if the main thread has disconnected
                    }
                }
            }

            Ok(WorkerAction::Snapshot(env_ids)) => {
                for env_id in env_ids {
                    if let Some(env) = envs.get(&env_id) {
                        let snapshot = env.clone();
                        if state_tx
                            .send(WorkerMessage::SnapshotResult { env_id, snapshot })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }

            Ok(WorkerAction::Query(env_id, key)) => {
                if let Some(env) = envs.get(&env_id) {
                    let result = env.query_state(&key);
                    if state_tx
                        .send(WorkerMessage::QueryResult { env_id, result })
                        .is_err()
                    {
                        return;
                    }
                }
            }

            Ok(WorkerAction::QueryMask(env_id, row_idx, prefix)) => {
                // Non-advancing (&self): compute the conditional mask, do not step.
                if let Some(env) = envs.get(&env_id) {
                    let mask = env.conditional_mask(&prefix);
                    if state_tx
                        .send(WorkerMessage::MaskResult {
                            env_id,
                            row_idx,
                            mask,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }

            Ok(WorkerAction::QueryMaskBatch(rows)) => {
                let masks = rows
                    .into_iter()
                    .filter_map(|(env_id, row_idx, prefix)| {
                        envs.get(&env_id)
                            .map(|env| (env_id, row_idx, env.conditional_mask(&prefix)))
                    })
                    .collect();
                if state_tx.send(WorkerMessage::MaskResults(masks)).is_err() {
                    return;
                }
            }

            Ok(WorkerAction::Close) | Err(_) => {
                break;
            }
        }
    }
}
