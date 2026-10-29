use crossbeam_channel::{unbounded, Receiver, Sender};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::thread;

pub mod core;
pub mod envs;
pub mod registry;

use crate::core::batching::{self, ObsSpecEntry};
use crate::core::mcts::StateRegistry;
pub use crate::core::rl_env::{FactorizationSpec, RlEnv};
use crate::core::threads;
pub use crate::core::types::{ArrayData, GenericObs};
use crate::core::worker::{worker_loop, StepOut, WorkerAction, WorkerMessage};
pub use crate::core::worker::{AutoReset, Dispatch};

pub type EnvFactory = Box<dyn Fn(u64) -> Box<dyn RlEnv> + Send + Sync>;

/// Pool options shared by both ways of building a pool: the Python constructor (`RlPool(...)`
/// / `rustpool.make`, envs from the built-in registry) and [`RlPool::new_with_options`]
/// (external env crates). `Default` is rustpool's defaults: seed 0, threads from
/// `RUSTPOOL_NUM_THREADS` / the CPUs, `Dispatch::Auto`, `AutoReset::SameStep`.
///
/// ```ignore
/// let opts = PoolOptions { seed: 42, autoreset: AutoReset::NextStep, ..Default::default() };
/// RlPool::new_with_options(num_envs, batch_size, factory, opts)
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolOptions {
    /// Env `i` is built as `factory(seed + i)` (wrapping at `u64::MAX`).
    pub seed: u64,
    /// Worker threads; `None` falls back to `RUSTPOOL_NUM_THREADS`, then to the available
    /// CPUs. Always capped at `num_envs`.
    pub num_threads: Option<usize>,
    pub dispatch: Dispatch,
    pub autoreset: AutoReset,
}

impl PoolOptions {
    /// Parse the string forms the Python constructor takes (`"auto"`, `"per_env"`,
    /// `"per_worker"`; `"same_step"`, `"next_step"`), raising `ValueError` on anything else.
    /// For an external crate's own `make(..., dispatch=..., autoreset=...)`.
    pub fn parse(
        seed: u64,
        num_threads: Option<usize>,
        dispatch: &str,
        autoreset: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            seed,
            num_threads,
            dispatch: parse_dispatch(dispatch)?,
            autoreset: parse_autoreset(autoreset)?,
        })
    }
}

struct StepResult {
    env_id: usize,
    obs: GenericObs,
    reward: f32,
    done: bool,
    action_mask: Vec<bool>,
}

fn parse_obs_spec(py: Python<'_>, spec_obj: &Py<PyAny>) -> PyResult<HashMap<String, ObsSpecEntry>> {
    let spec_dict = spec_obj.cast_bound::<PyDict>(py)?;
    let mut obs_spec = HashMap::with_capacity(spec_dict.len());

    for (key, value) in spec_dict.iter() {
        let key_str: String = key.extract()?;
        let (shape, dtype_raw): (Vec<usize>, String) = value.extract()?;
        let dtype = batching::ObsDtype::from_str(&dtype_raw).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unsupported dtype '{}' for key {}",
                dtype_raw, key_str
            ))
        })?;

        obs_spec.insert(key_str, ObsSpecEntry { shape, dtype });
    }

    Ok(obs_spec)
}

/// `query_mask` / `query_state_mask` on an env with no action factorization.
fn no_factorization_error() -> PyErr {
    PyValueError::new_err(
        "the pool's env has no action factorization (its conditional masks have width 0): \
         query_mask and query_state_mask need a factored action space",
    )
}

/// A pool of native Rust reinforcement-learning environments with an asynchronous,
/// EnvPool-style API plus a state-cloning substrate for tree search (AlphaZero/MCTS).
///
/// Each environment instance lives on a Rust worker thread (true parallelism, no GIL).
/// The pool is driven with a non-blocking ``send`` / blocking ``recv`` protocol:
/// ``recv`` returns as soon as ``batch_size`` environments are ready, so policy
/// inference overlaps with simulation. Observations come back as a ``dict`` of NumPy
/// arrays; rewards/dones as 1-D NumPy arrays; ``info`` carries ``env_id`` and (when the
/// env provides one) ``action_mask``.
///
/// Beyond EnvPool it also exposes ``snapshot`` / ``simulate_batch`` / ``release_batch``
/// (clone arbitrary states and fork them in parallel — the substrate a batched MCTS /
/// AlphaZero search drives) and ``query_mask`` / ``query_state_mask`` /
/// ``action_factorization`` (factored, autoregressive action masks, on live envs or on
/// stored states).
///
/// Prefer constructing via ``rustpool.make``, which also fills in the task's default
/// config. Exposed in Python as ``rustpool.RlPool``; ``MyEnvPool``, its old name, is a
/// deprecated alias.
///
/// Parameters
/// ----------
/// num_envs : int
///     Number of environments in the pool (default 4).
/// batch_size : int
///     How many ready environments ``recv`` waits for (``<= num_envs``, default 4).
///     ``batch_size == num_envs`` is the synchronous regime; smaller is asynchronous: ``recv``
///     returns the first ``batch_size`` envs to finish while the others keep stepping.
/// seed : int
///     Env ``i`` is built with seed ``seed + i`` (default 0).
/// task_id : str
///     A registered task, see ``rustpool.list_envs`` (default ``"BinPack-v0"``).
/// num_threads : int, optional
///     Keyword-only. Worker threads; when omitted, ``RUSTPOOL_NUM_THREADS``, then the CPUs
///     available to the process. Always capped at ``num_envs``. Env ``i`` lives on worker
///     ``i % num_threads``.
/// dispatch : {"auto", "per_env", "per_worker"}
///     Keyword-only. How ``send`` and ``query_mask`` reach the workers: one channel message
///     per env each way (``per_env``), or one per worker per call with that worker's results
///     returned together (``per_worker``). ``auto`` (default) picks ``per_worker`` when a call
///     has more rows than there are workers, else ``per_env``. Results are identical in every
///     mode.
/// autoreset : {"same_step", "next_step"}
///     Keyword-only. ``same_step`` (default): the terminal step resets the env at once and
///     returns the new episode's first observation with the terminal reward and
///     ``done=True``. ``next_step`` (EnvPool's and gymnasium's convention): the terminal step
///     returns the terminal observation with ``done=True``, and the env's next ``send`` is the
///     reset (its action is ignored; reward 0, ``done=False``).
/// **kwargs
///     The env's own config, deserialised in Rust: an unknown key or a mistyped value raises
///     ``TypeError``, an out-of-range value ``ValueError``; see the environment reference.
///
/// Example
/// -------
/// >>> import numpy as np, rustpool
/// >>> pool = rustpool.make("Maze-v0", num_envs=8, batch_size=8)
/// >>> pool.async_reset()
/// >>> obs, rew, done, info = pool.recv()
/// >>> actions = np.zeros(len(info["env_id"]), dtype=np.int32)
/// >>> pool.send(actions, info["env_id"])
/// >>> pool.close()
#[pyclass(subclass, module = "rustpool._rustpool")]
pub struct RlPool {
    /// Number of Rust worker threads this pool spawned (see ``num_threads`` in the constructor).
    #[pyo3(get)]
    num_threads: usize,
    action_txs: Vec<Sender<WorkerAction>>,
    state_rx: Receiver<WorkerMessage>,
    message_buffer: Mutex<VecDeque<WorkerMessage>>,
    _env_template: Box<dyn RlEnv>,
    obs_spec: HashMap<String, ObsSpecEntry>,
    obs_key_names: Vec<String>,
    /// Number of environments in the pool.
    #[pyo3(get)]
    num_envs: usize,
    /// How many ready environments ``recv`` returns per call.
    #[pyo3(get)]
    batch_size: usize,
    _worker_handles: Option<Vec<thread::JoinHandle<()>>>,
    registry: StateRegistry,
    dispatch: Dispatch,
    autoreset: AutoReset,
}

/// Move a step-result message (`StepResult` or `StepResults`) into `results` until it holds
/// `cap` entries; whatever does not fit comes back, to be buffered for the next `recv`.
fn take_step_results(
    msg: WorkerMessage,
    results: &mut Vec<StepResult>,
    cap: usize,
) -> Option<WorkerMessage> {
    let push = |o: StepOut, results: &mut Vec<StepResult>| {
        results.push(StepResult {
            env_id: o.env_id,
            obs: o.obs,
            reward: o.reward,
            done: o.done,
            action_mask: o.action_mask,
        })
    };
    match msg {
        WorkerMessage::StepResult {
            env_id,
            obs,
            reward,
            done,
            action_mask,
        } => {
            let out = StepOut {
                env_id,
                obs,
                reward,
                done,
                action_mask,
            };
            if results.len() >= cap {
                return Some(out.into());
            }
            push(out, results);
            None
        }
        WorkerMessage::StepResults(outs) => {
            let mut rest = outs.into_iter();
            while results.len() < cap {
                match rest.next() {
                    Some(o) => push(o, results),
                    None => return None,
                }
            }
            let rest: Vec<StepOut> = rest.collect();
            (!rest.is_empty()).then_some(WorkerMessage::StepResults(rest))
        }
        other => Some(other),
    }
}

fn is_step_result(msg: &WorkerMessage) -> bool {
    matches!(
        msg,
        WorkerMessage::StepResult { .. } | WorkerMessage::StepResults(_)
    )
}

fn parse_autoreset(value: &str) -> PyResult<AutoReset> {
    AutoReset::parse(value).ok_or_else(|| {
        PyValueError::new_err(format!(
            "autoreset must be 'same_step' or 'next_step', got {:?}",
            value
        ))
    })
}

fn parse_dispatch(value: &str) -> PyResult<Dispatch> {
    Dispatch::parse(value).ok_or_else(|| {
        PyValueError::new_err(format!(
            "dispatch must be 'auto', 'per_env' or 'per_worker', got {:?}",
            value
        ))
    })
}

#[pymethods]
impl RlPool {
    /// Construct a pool of ``num_envs`` environments of type ``task_id``.
    ///
    /// ``recv`` returns once ``batch_size`` (<= ``num_envs``) environments are ready.
    /// ``seed`` seeds env ``i`` deterministically as ``seed + i``. Extra ``**kwargs`` are
    /// forwarded to the env config (validated by serde); see ``rustpool.make`` and the
    /// per-env docs for accepted keys.
    ///
    /// ``num_threads`` (keyword-only) sets the number of Rust worker threads. When omitted it
    /// falls back to the ``RUSTPOOL_NUM_THREADS`` environment variable (read here, at
    /// construction), then to the number of CPUs available to the process. It is always
    /// capped at ``num_envs``. Note every pool spawns its own workers, so a process holding
    /// several pools should split its CPUs between them.
    ///
    /// ``dispatch`` (keyword-only) sets how ``send`` and ``query_mask`` reach the workers:
    /// ``"per_env"`` sends one channel message per env each way; ``"per_worker"`` one per
    /// worker thread per call, the results coming back together per worker; ``"auto"``
    /// (default) chooses per call: per worker when the call has more rows than there are
    /// worker threads, else per env. Results are identical in every mode.
    ///
    /// ``autoreset`` (keyword-only): ``"same_step"`` (default) resets an env on its terminal
    /// step and returns the next episode's first observation with ``done=True``;
    /// ``"next_step"`` follows EnvPool, returning the terminal observation and resetting on the
    /// env's next ``send`` (whose action is ignored; reward 0, ``done=False``).
    #[new]
    #[pyo3(signature = (num_envs=4, batch_size=4, seed=0, task_id="BinPack-v0", *, num_threads=None, dispatch="auto", autoreset="same_step", **kwargs),
           text_signature = "(num_envs=4, batch_size=4, seed=0, task_id='BinPack-v0', *, num_threads=None, dispatch='auto', autoreset='same_step', **kwargs)")]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        batch_size: usize,
        seed: u64,
        task_id: &str,
        num_threads: Option<usize>,
        dispatch: &str,
        autoreset: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let options = PoolOptions::parse(seed, num_threads, dispatch, autoreset)?;
        let factory = Python::attach(|py| {
            let empty_kwargs = PyDict::new(py);
            registry::create_env_factory(task_id, kwargs.unwrap_or(&empty_kwargs))
        })?;
        Self::new_with_options(num_envs, batch_size, factory, options)
    }

    /// ``"auto"``, ``"per_env"`` or ``"per_worker"``: how ``send`` and ``query_mask`` reach
    /// the workers (see the constructor).
    #[getter]
    pub fn dispatch(&self) -> &'static str {
        self.dispatch.name()
    }

    /// ``"same_step"`` or ``"next_step"``: which call returns an episode's reset observation
    /// (see the constructor).
    #[getter]
    pub fn autoreset(&self) -> &'static str {
        self.autoreset.name()
    }

    /// The action space as ``(num_actions,)`` for a flat discrete action space.
    pub fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self._env_template.action_spec(py)
    }

    /// Width of ``info["action_mask"]`` rows: ``num_actions`` when the env emits a flat mask;
    /// the factorization's ``pad_width`` when it emits the stage-0 factor mask instead
    /// (``BinPack-v0(factored=True)``, whose later stages come from ``query_mask``); ``0`` when
    /// it emits none (``BinPack-v0(emit_flat_mask=False)``). Wrappers that pre-declare output
    /// shapes must use this, not ``action_spec()``.
    pub fn action_mask_width(&self) -> usize {
        let mut probe = self._env_template.clone();
        probe.reset(None).3.len()
    }

    /// The observation space as ``{key: (shape, dtype_str)}`` (e.g. ``([2], "float32")``),
    /// matching the keys and per-array shapes returned in the ``obs`` dict by ``recv``.
    pub fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self._env_template.observation_spec(py)
    }

    /// The env's chosen action-space factorization as a dict (or None for a flat
    /// action space). Generic capability: the policy reads this to build a factored /
    /// autoregressive head + the per-factor mask-query loop (via query_mask).
    pub fn action_factorization(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self._env_template.action_factorization() {
            Some(spec) => {
                let num_factors = spec.factor_widths.len();
                let d = PyDict::new(py);
                d.set_item("name", spec.name)?;
                d.set_item("factor_names", spec.factor_names)?;
                d.set_item("factor_widths", spec.factor_widths)?;
                d.set_item("num_factors", num_factors)?;
                d.set_item("pad_width", spec.pad_width)?;
                d.set_item("flat_action_size", spec.flat_action_size)?;
                d.set_item("conditional", spec.conditional)?;
                Ok(d.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }

    /// Signal every worker thread to stop, join them, and free every stored state (all
    /// ``snapshot`` / ``simulate_batch`` handles become invalid). Idempotent; also called when a
    /// ``with`` block exits and when the pool is garbage-collected, but call it (or use
    /// ``with``) to release the threads and the states promptly. Afterwards every method that
    /// reaches the envs or the states raises ``RuntimeError``; ``release_batch`` and ``close``
    /// stay no-ops, and the specs and sizes stay readable.
    pub fn close(&mut self, py: Python) {
        if let Some(handles) = self._worker_handles.take() {
            for tx in &self.action_txs {
                let _ = tx.send(WorkerAction::Close);
            }
            py.detach(move || {
                for handle in handles {
                    // Tolerate a worker that panicked (matches the Drop impl); re-panicking here
                    // would mask the original error and break the documented idempotent teardown.
                    let _ = handle.join();
                }
            });
        }
        self.registry.clear();
    }

    /// ``with rustpool.make(...) as pool:`` returns the pool itself.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Closes the pool when the ``with`` block exits; exceptions propagate.
    #[pyo3(signature = (*_exc_info))]
    fn __exit__(&mut self, py: Python<'_>, _exc_info: &Bound<'_, pyo3::types::PyTuple>) -> bool {
        self.close(py);
        false
    }

    /// Dispatch one ``action`` per ``env_id`` to the worker threads (non-blocking).
    ///
    /// ``actions`` and ``env_ids`` are 1-D ``int32`` arrays of equal length; pair the
    /// ``env_id`` array returned by ``recv`` with the actions your policy chose. Results
    /// are collected later by ``recv``. When an episode ends the env resets itself, in the
    /// same call (``autoreset="same_step"``, the default) or on its next ``send``
    /// (``"next_step"``); see the class parameters. Every id is validated before anything is
    /// sent, so a bad id (out of range or negative) raises without stepping any env.
    #[pyo3(text_signature = "($self, actions, env_ids)")]
    pub fn send(
        &self,
        actions: PyReadonlyArray1<i32>,
        env_ids: PyReadonlyArray1<i32>,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let actions_slice = actions.as_slice()?;
        let env_ids_slice = env_ids.as_slice()?;

        if actions_slice.len() != env_ids_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "actions length ({}) must match env_ids length ({})",
                actions_slice.len(),
                env_ids_slice.len()
            )));
        }

        // Validate every env_id BEFORE dispatching: a bad id must not leave the actions before
        // it in flight (those envs would step once more than the caller knows).
        if let Some(bad) = env_ids_slice
            .iter()
            .find(|&&id| id < 0 || (id as usize) >= self.num_envs)
        {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "env_id {} is out of bounds for num_envs {}",
                bad, self.num_envs
            )));
        }
        if self.dispatch.batches(actions_slice.len(), self.num_threads) {
            let mut shards: Vec<Vec<(usize, i32)>> = (0..self.num_threads)
                .map(|_| Vec::with_capacity(actions_slice.len() / self.num_threads + 1))
                .collect();
            for (action, env_id) in actions_slice.iter().zip(env_ids_slice.iter()) {
                shards[(*env_id as usize) % self.num_threads].push((*env_id as usize, *action));
            }
            for (shard_id, batch) in shards.into_iter().enumerate() {
                if !batch.is_empty() {
                    self.action_txs[shard_id]
                        .send(WorkerAction::StepBatch(batch))
                        .unwrap();
                }
            }
        } else {
            for (action, env_id) in actions_slice.iter().zip(env_ids_slice.iter()) {
                let shard_id = (*env_id as usize) % self.num_threads;
                self.action_txs[shard_id]
                    .send(WorkerAction::Step(*env_id as usize, *action))
                    .unwrap();
            }
        }
        Ok(())
    }

    /// Block until ``batch_size`` environments are ready and return one batch.
    ///
    /// Returns ``(obs, reward, done, info)`` where ``obs`` is a ``dict`` of stacked NumPy
    /// arrays (leading axis ``batch_size``), ``reward`` is ``float32[batch_size]``,
    /// ``done`` is ``bool[batch_size]`` and ``info`` is a dict with ``env_id``
    /// (``int32[batch_size]``, sorted ascending) and, when the env exposes one,
    /// ``action_mask`` (``bool[batch_size, action_mask_width()]``). Releases the GIL while
    /// waiting. It waits for exactly ``batch_size`` results: with fewer environments sent (or
    /// reset) and not yet received, it blocks forever, and Ctrl-C does not interrupt it.
    #[pyo3(text_signature = "($self)")]
    pub fn recv<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.ensure_open()?;
        let mut results = Vec::with_capacity(self.batch_size);

        while results.len() < self.batch_size {
            // First check the buffer (step results set aside earlier: the leftovers of a
            // per-worker batch, or results another call received while waiting for its own).
            let buffered_msg = {
                let mut buffer = self.message_buffer.lock().unwrap();
                buffer
                    .iter()
                    .position(is_step_result)
                    .and_then(|idx| buffer.remove(idx))
            };
            if let Some(msg) = buffered_msg {
                if let Some(rest) = take_step_results(msg, &mut results, self.batch_size) {
                    self.message_buffer.lock().unwrap().push_front(rest);
                }
                continue;
            }

            // If nothing in buffer, read from channel
            let msg = py.detach(|| self.state_rx.recv().unwrap());
            if is_step_result(&msg) {
                if let Some(rest) = take_step_results(msg, &mut results, self.batch_size) {
                    self.message_buffer.lock().unwrap().push_back(rest);
                }
            } else {
                // Push to buffer and keep looking for step results
                self.message_buffer.lock().unwrap().push_back(msg);
            }
        }

        results.sort_by_key(|r| r.env_id);

        let mut reward_list = Vec::with_capacity(self.batch_size);
        let mut done_list = Vec::with_capacity(self.batch_size);
        let mut env_id_list = Vec::with_capacity(self.batch_size);
        let mut action_mask_list = Vec::with_capacity(self.batch_size);
        let mut obs_list = Vec::with_capacity(self.batch_size);

        for result in results {
            reward_list.push(result.reward);
            done_list.push(result.done);
            env_id_list.push(result.env_id as i32);
            action_mask_list.push(result.action_mask);
            obs_list.push(result.obs);
        }

        let info = pyo3::types::PyDict::new(py);
        info.set_item("env_id", env_id_list.as_slice().to_pyarray(py))?;

        if let Some(stacked_masks) = batching::stack_action_masks(py, &action_mask_list)? {
            info.set_item("action_mask", stacked_masks)?;
        }

        let stacked_obs = if obs_list.is_empty() {
            PyDict::new(py).into_any().unbind()
        } else {
            batching::stack_observations(py, &self.obs_spec, &self.obs_key_names, &obs_list)?
        };

        let result_tuple = (
            stacked_obs,
            reward_list.as_slice().to_pyarray(py),
            done_list.as_slice().to_pyarray(py),
            info,
        );
        Ok(result_tuple.into_pyobject(py)?.into_any().unbind())
    }

    /// Reset every environment (non-blocking); collect the initial observations with
    /// ``recv``. ``seeds`` is an optional ``int64[num_envs]`` array of per-env seeds for
    /// reproducible resets; ``None`` resets without reseeding.
    #[pyo3(signature = (seeds=None), text_signature = "($self, seeds=None)")]
    pub fn async_reset(&self, seeds: Option<PyReadonlyArray1<i64>>) -> PyResult<()> {
        self.ensure_open()?;
        match seeds {
            None => {
                for tx in &self.action_txs {
                    tx.send(WorkerAction::Reset(None)).unwrap();
                }
            }
            Some(seeds_array) => {
                let seeds_slice = seeds_array.as_slice()?;
                if seeds_slice.len() != self.num_envs {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "seeds length ({}) must match num_envs ({})",
                        seeds_slice.len(),
                        self.num_envs
                    )));
                }

                let mut shard_buffers: Vec<Vec<(usize, u64)>> =
                    (0..self.num_threads).map(|_| Vec::new()).collect();

                for (env_id, &seed) in seeds_slice.iter().enumerate() {
                    let shard_id = env_id % self.num_threads;
                    shard_buffers[shard_id].push((env_id, seed as u64));
                }

                for (shard_id, buffer) in shard_buffers.into_iter().enumerate() {
                    self.action_txs[shard_id]
                        .send(WorkerAction::Reset(Some(buffer)))
                        .unwrap();
                }
            }
        }
        Ok(())
    }

    /// Number of environments in the pool (``len(pool)``).
    pub fn __len__(&self) -> usize {
        self.num_envs
    }

    /// Clone the current state of each given environment into the state registry and
    /// return a list of integer ``state_id`` handles (one per ``env_id``). The handles
    /// are the roots a tree search forks from with ``simulate_batch``; free them with
    /// ``release_batch``. ``env_ids`` is an ``int32`` array; an id outside ``[0, num_envs)``
    /// raises ``IndexError`` before any env is cloned. Does not advance any env.
    #[pyo3(text_signature = "($self, env_ids)")]
    pub fn snapshot(&self, env_ids: PyReadonlyArray1<i32>) -> PyResult<Vec<i32>> {
        self.ensure_open()?;
        let env_ids_vec = env_ids
            .as_slice()?
            .iter()
            .map(|&id| self.check_env_id(id.into()))
            .collect::<PyResult<Vec<usize>>>()?;

        // Group by shard
        let mut shard_requests: Vec<Vec<usize>> =
            (0..self.num_threads).map(|_| Vec::new()).collect();
        for &env_id in &env_ids_vec {
            let shard_id = env_id % self.num_threads;
            shard_requests[shard_id].push(env_id);
        }

        // Send requests
        for (shard_id, ids) in shard_requests.into_iter().enumerate() {
            if !ids.is_empty() {
                self.action_txs[shard_id]
                    .send(WorkerAction::Snapshot(ids))
                    .unwrap();
            }
        }

        // Collect results. An env id may be requested more than once: the worker answers each
        // request with its own clone, so keep every clone and hand one to each request.
        let mut snapshots: HashMap<usize, Vec<Box<dyn RlEnv>>> = HashMap::new();
        let mut remaining = env_ids_vec.len();

        while remaining > 0 {
            // First check message_buffer
            let msg_opt = {
                let mut buffer = self.message_buffer.lock().unwrap();
                let mut found_idx = None;
                for (idx, msg) in buffer.iter().enumerate() {
                    if let WorkerMessage::SnapshotResult { env_id, .. } = msg {
                        if env_ids_vec.contains(env_id) {
                            found_idx = Some(idx);
                            break;
                        }
                    }
                }
                found_idx.and_then(|idx| buffer.remove(idx))
            };

            if let Some(WorkerMessage::SnapshotResult { env_id, snapshot }) = msg_opt {
                snapshots.entry(env_id).or_default().push(snapshot);
                remaining -= 1;
                continue;
            }

            // Receive from channel
            let msg = self.state_rx.recv().unwrap();
            match msg {
                WorkerMessage::SnapshotResult { env_id, snapshot } => {
                    if env_ids_vec.contains(&env_id) {
                        snapshots.entry(env_id).or_default().push(snapshot);
                        remaining -= 1;
                    } else {
                        // Not for us
                        self.message_buffer
                            .lock()
                            .unwrap()
                            .push_back(WorkerMessage::SnapshotResult { env_id, snapshot });
                    }
                }
                _ => {
                    self.message_buffer.lock().unwrap().push_back(msg);
                }
            }
        }

        let mut ordered_snapshots = Vec::with_capacity(env_ids_vec.len());
        for &env_id in &env_ids_vec {
            let snapshot = snapshots
                .get_mut(&env_id)
                .and_then(Vec::pop)
                .expect("one snapshot per requested env_id");
            ordered_snapshots.push(snapshot);
        }

        Ok(self.registry.snapshot(ordered_snapshots))
    }

    /// Escape hatch to read env-internal state natively: returns the parsed JSON the env
    /// exposes for ``key`` (a Python object), or ``None`` if the env does not support it.
    /// Buffers any interleaved async messages so it is safe to call mid-rollout. An
    /// ``env_id`` outside ``[0, num_envs)`` raises ``IndexError``.
    #[pyo3(signature = (env_id, key), text_signature = "($self, env_id, key)")]
    pub fn query_state<'py>(
        &self,
        py: Python<'py>,
        env_id: i64,
        key: String,
    ) -> PyResult<Py<PyAny>> {
        self.ensure_open()?;
        let env_id = self.check_env_id(env_id)?;

        let shard_id = env_id % self.num_threads;
        self.action_txs[shard_id]
            .send(WorkerAction::Query(env_id, key))
            .unwrap();

        loop {
            // First check message_buffer
            let msg_opt = {
                let mut buffer = self.message_buffer.lock().unwrap();
                let mut found_idx = None;
                for (idx, msg) in buffer.iter().enumerate() {
                    if let WorkerMessage::QueryResult {
                        env_id: res_env_id, ..
                    } = msg
                    {
                        if *res_env_id == env_id {
                            found_idx = Some(idx);
                            break;
                        }
                    }
                }
                found_idx.and_then(|idx| buffer.remove(idx))
            };

            if let Some(WorkerMessage::QueryResult { result, .. }) = msg_opt {
                return match result {
                    Some(json_str) => {
                        let json_module = py.import("json")?;
                        Ok(json_module.call_method1("loads", (json_str,))?.into())
                    }
                    None => Ok(py.None()),
                };
            }

            // Receive from channel
            let msg = py.detach(|| self.state_rx.recv().unwrap());
            match msg {
                WorkerMessage::QueryResult {
                    env_id: res_env_id,
                    result,
                } => {
                    if res_env_id == env_id {
                        return match result {
                            Some(json_str) => {
                                let json_module = py.import("json")?;
                                Ok(json_module.call_method1("loads", (json_str,))?.into())
                            }
                            None => Ok(py.None()),
                        };
                    } else {
                        // Not for us
                        self.message_buffer
                            .lock()
                            .unwrap()
                            .push_back(WorkerMessage::QueryResult {
                                env_id: res_env_id,
                                result,
                            });
                    }
                }
                _ => {
                    // Buffer anything else
                    self.message_buffer.lock().unwrap().push_back(msg);
                }
            }
        }
    }

    /// Batched, **non-advancing** conditional-mask query on the live envs: it travels to the
    /// workers like ``send`` / ``recv`` (``dispatch`` applies) but steps nothing.
    ///
    /// ``prefixes`` is ``int32[n, k]`` (the factors chosen so far; ``-1`` = unset) and
    /// ``env_ids`` is ``int32[n]``; ids may repeat. Row ``i`` is that env's mask over the next
    /// factor given ``prefixes[i]``. Returns a ``bool[n, W]`` array in the input order
    /// (``[0, pad_width]`` for ``n == 0``). Each ``True`` leaves at least one valid completion,
    /// so sampling factor by factor never dead-ends. An env with no action factorization (it
    /// answers empty masks) raises ``ValueError``.
    #[pyo3(signature = (prefixes, env_ids))]
    pub fn query_mask<'py>(
        &self,
        py: Python<'py>,
        prefixes: PyReadonlyArray2<i32>,
        env_ids: PyReadonlyArray1<i32>,
    ) -> PyResult<Py<PyAny>> {
        self.ensure_open()?;
        let env_ids_slice = env_ids.as_slice()?;
        let prefixes_arr = prefixes.as_array();
        let n = prefixes_arr.shape()[0];
        if n != env_ids_slice.len() {
            return Err(PyValueError::new_err(format!(
                "prefixes rows ({}) must match env_ids length ({})",
                n,
                env_ids_slice.len()
            )));
        }

        // Validate every env_id BEFORE dispatching, so a bad row can't leave half the batch's
        // requests in flight (whose results would later pollute the message buffer).
        if let Some(bad) = env_ids_slice
            .iter()
            .find(|&&id| id < 0 || (id as usize) >= self.num_envs)
        {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "env_id {} is out of bounds for num_envs {}",
                bad, self.num_envs
            )));
        }

        // Dispatch the non-advancing requests, each tagged with its row index so duplicate
        // env_ids stay distinct on the way back: one message per row, or one per worker.
        if self.dispatch.batches(n, self.num_threads) {
            let mut shards: Vec<Vec<(usize, usize, Vec<i32>)>> =
                (0..self.num_threads).map(|_| Vec::new()).collect();
            for (i, env_id) in env_ids_slice.iter().enumerate() {
                let row: Vec<i32> = prefixes_arr.row(i).iter().copied().collect();
                shards[(*env_id as usize) % self.num_threads].push((*env_id as usize, i, row));
            }
            for (shard_id, rows) in shards.into_iter().enumerate() {
                if !rows.is_empty() {
                    self.action_txs[shard_id]
                        .send(WorkerAction::QueryMaskBatch(rows))
                        .unwrap();
                }
            }
        } else {
            for (i, env_id) in env_ids_slice.iter().enumerate() {
                let row: Vec<i32> = prefixes_arr.row(i).iter().copied().collect();
                let shard_id = (*env_id as usize) % self.num_threads;
                self.action_txs[shard_id]
                    .send(WorkerAction::QueryMask(*env_id as usize, i, row))
                    .unwrap();
            }
        }

        // Collect results POSITIONALLY by row index (not by env_id) so duplicate env_ids map to
        // the right rows; re-buffer anything else so query_mask and recv interleave safely.
        let mut results: Vec<Option<Vec<bool>>> = vec![None; n];
        let mut filled = 0usize;
        // A row of this call fills its slot; anything else (a row of another call) is set
        // aside and re-buffered.
        let mut set_aside: Vec<WorkerMessage> = Vec::new();
        let mut other: Vec<WorkerMessage> = Vec::new();
        let mut place = |env_id: usize, row_idx: usize, mask: Vec<bool>, filled: &mut usize| {
            if row_idx < n && results[row_idx].is_none() {
                results[row_idx] = Some(mask);
                *filled += 1;
            } else {
                set_aside.push(WorkerMessage::MaskResult {
                    env_id,
                    row_idx,
                    mask,
                });
            }
        };
        while filled < n {
            let buffered = {
                let mut buffer = self.message_buffer.lock().unwrap();
                let found = buffer.iter().position(|msg| match msg {
                    // A filled slot means the row belongs to another call: `place` sets it
                    // aside (re-buffered only at the end, so it is not picked up again).
                    WorkerMessage::MaskResult { row_idx, .. } => *row_idx < n,
                    WorkerMessage::MaskResults(_) => true,
                    _ => false,
                });
                found.and_then(|idx| buffer.remove(idx))
            };
            let msg = match buffered {
                Some(msg) => msg,
                None => py.detach(|| self.state_rx.recv().unwrap()),
            };
            match msg {
                WorkerMessage::MaskResult {
                    env_id,
                    row_idx,
                    mask,
                } => place(env_id, row_idx, mask, &mut filled),
                WorkerMessage::MaskResults(rows) => {
                    for (env_id, row_idx, mask) in rows {
                        place(env_id, row_idx, mask, &mut filled);
                    }
                }
                msg => other.push(msg),
            }
        }
        if !set_aside.is_empty() || !other.is_empty() {
            let mut buffer = self.message_buffer.lock().unwrap();
            buffer.extend(set_aside);
            buffer.extend(other);
        }

        let masks: Vec<Vec<bool>> = results.into_iter().map(|m| m.unwrap_or_default()).collect();

        match batching::stack_action_masks(py, &masks)? {
            Some(stacked) if !masks[0].is_empty() => Ok(stacked),
            _ => match self.pad_width() {
                Some(w) if n == 0 => batching::stack_fixed_width_masks(py, &[], w),
                _ => Err(no_factorization_error()),
            },
        }
    }

    /// ``query_mask`` for **stored states**: row ``i`` is the non-advancing conditional mask
    /// of registry handle ``state_ids[i]`` (from ``snapshot`` / ``simulate_batch``) given
    /// ``prefixes[i]`` (``-1`` = unset). ``state_ids`` is ``[n]`` i32 and ``prefixes`` is
    /// ``[n, k]`` i32; returns a ``[n, W]`` bool array in INPUT order (an id may repeat). Lets a
    /// tree search read per-factor masks at any node without the flat mask.
    ///
    /// * ``W`` is ``width`` when given, else the env's ``action_factorization()["pad_width"]``,
    ///   else the width of the first non-empty row. When none is known (an env with no
    ///   conditional masks, e.g. a flat action mode) it raises ``ValueError``.
    /// * ``state_id < 0`` (the ``release`` sentinel, used for dead / finished search rows)
    ///   answers an all-false row without a registry lookup or any mask work.
    /// * Every other row must come back exactly ``W`` wide, else ``ValueError`` (e.g. an env
    ///   outside its factored action mode returns EMPTY masks; they are never padded).
    /// * Unknown or already-released ids raise ``ValueError``.
    /// * Reads the stored states without cloning or mutating them and never touches the
    ///   worker channels (safe to interleave with in-flight ``send``/``recv``); runs in
    ///   parallel (Rayon) with the GIL released.
    #[pyo3(signature = (state_ids, prefixes, width=None))]
    pub fn query_state_mask<'py>(
        &self,
        py: Python<'py>,
        state_ids: PyReadonlyArray1<i32>,
        prefixes: PyReadonlyArray2<i32>,
        width: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        self.ensure_open()?;
        let ids: Vec<i32> = state_ids.as_slice()?.to_vec();
        let prefixes_arr = prefixes.as_array();
        let n = prefixes_arr.shape()[0];
        if n != ids.len() {
            return Err(PyValueError::new_err(format!(
                "prefixes rows ({}) must match state_ids length ({})",
                n,
                ids.len()
            )));
        }
        if width == Some(0) {
            return Err(PyValueError::new_err("width must be positive"));
        }
        let rows: Vec<Vec<i32>> = (0..n)
            .map(|i| prefixes_arr.row(i).iter().copied().collect())
            .collect();
        let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
        let registry = &self.registry;
        let masks = py
            .detach(|| registry.conditional_masks(&ids, &row_refs))
            .map_err(PyValueError::new_err)?;

        let w = match width
            .or_else(|| self.pad_width())
            .or_else(|| masks.iter().map(Vec::len).find(|&len| len > 0))
        {
            Some(w) => w,
            None => return Err(no_factorization_error()),
        };
        let mut out: Vec<Vec<bool>> = Vec::with_capacity(n);
        for (i, mask) in masks.into_iter().enumerate() {
            if ids[i] < 0 {
                out.push(vec![false; w]);
            } else if mask.len() != w {
                return Err(PyValueError::new_err(format!(
                    "conditional mask of state_id {} (row {}) has width {} (expected {}); is \
                     the env in a factored action mode?",
                    ids[i],
                    i,
                    mask.len(),
                    w
                )));
            } else {
                out.push(mask);
            }
        }
        batching::stack_fixed_width_masks(py, &out, w)
    }

    /// Fork-and-step a batch of stored states **without mutating the parents**.
    ///
    /// For each ``(state_id, action)`` pair, clone the stored state, apply one ``step`` in
    /// parallel (Rayon, GIL released), store each resulting child as a **new** ``state_id``,
    /// and return ``(obs, reward, done, info)`` shaped exactly like ``recv`` — except
    /// ``info["state_id"]`` holds the children's ``int32`` handles. A parent ``state_id``
    /// can be expanded any number of times (persistent / immutable-node semantics), which
    /// is what a batched MCTS / AlphaZero search needs. ``state_ids`` and ``actions`` are
    /// equal-length ``int32`` arrays. Free children you no longer need with ``release_batch``.
    /// There is no auto-reset here: a child whose step ends the episode is stored as the
    /// terminal state, with ``done=True``.
    #[pyo3(text_signature = "($self, state_ids, actions)")]
    pub fn simulate_batch<'py>(
        &self,
        py: Python<'py>,
        state_ids: PyReadonlyArray1<i32>,
        actions: PyReadonlyArray1<i32>,
    ) -> PyResult<Py<PyAny>> {
        self.ensure_open()?;
        let state_ids_slice = state_ids.as_slice()?;
        let actions_slice = actions.as_slice()?;

        if state_ids_slice.len() != actions_slice.len() {
            return Err(PyValueError::new_err(
                "state_ids and actions must have same length",
            ));
        }

        // Phase A: Read (Short Lock)
        let envs = self
            .registry
            .get_clones(state_ids_slice)
            .map_err(PyValueError::new_err)?;
        let environments_to_step: Vec<(Box<dyn RlEnv>, i32)> = envs
            .into_iter()
            .zip(actions_slice.iter().copied())
            .collect();

        // Phase B: Compute (Parallel via Rayon)
        let results: Vec<(Box<dyn RlEnv>, GenericObs, f32, bool, Vec<bool>)> = py.detach(|| {
            environments_to_step
                .into_par_iter()
                .map(|(mut env, action)| {
                    let (obs, reward, done, action_mask) = env.step(action);
                    (env, obs, reward, done, action_mask)
                })
                .collect()
        });

        // Phase C: Write (Short Lock)
        let mut envs = Vec::with_capacity(results.len());
        let mut reward_list = Vec::with_capacity(results.len());
        let mut done_list = Vec::with_capacity(results.len());
        let mut action_mask_list = Vec::with_capacity(results.len());
        let mut obs_list = Vec::with_capacity(results.len());

        for (env, obs, reward, done, action_mask) in results {
            envs.push(env);
            reward_list.push(reward);
            done_list.push(done);
            action_mask_list.push(action_mask);
            obs_list.push(obs);
        }

        let new_state_ids = self.registry.snapshot(envs);

        let info = PyDict::new(py);
        info.set_item("state_id", new_state_ids.as_slice().to_pyarray(py))?;

        if let Some(stacked_masks) = batching::stack_action_masks(py, &action_mask_list)? {
            info.set_item("action_mask", stacked_masks)?;
        }

        let stacked_obs =
            batching::stack_observations(py, &self.obs_spec, &self.obs_key_names, &obs_list)?;

        let result_tuple = (
            stacked_obs,
            reward_list.as_slice().to_pyarray(py),
            done_list.as_slice().to_pyarray(py),
            info,
        );
        Ok(result_tuple.into_pyobject(py)?.into_any().unbind())
    }

    /// Free the registry entries for the given ``int32`` ``state_id`` handles. MCTS creates
    /// huge numbers of nodes; release pruned/finished handles or the registry leaks memory.
    /// Unknown ids (e.g. ``-1`` sentinels for unused tree slots) are ignored.
    #[pyo3(text_signature = "($self, state_ids)")]
    pub fn release_batch(&self, state_ids: PyReadonlyArray1<i32>) -> PyResult<()> {
        self.registry.release(state_ids.as_slice()?);
        Ok(())
    }
}

impl RlPool {
    /// `env_id` as an index, or `IndexError` when it is negative or `>= num_envs`. Checked on
    /// the signed value, so `-1` is reported as `-1`.
    fn check_env_id(&self, env_id: i64) -> PyResult<usize> {
        if env_id < 0 || env_id as u64 >= self.num_envs as u64 {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "env_id {} is out of bounds for num_envs {}",
                env_id, self.num_envs
            )));
        }
        Ok(env_id as usize)
    }

    /// The advertised factorization's `pad_width`, if the env has one.
    fn pad_width(&self) -> Option<usize> {
        self._env_template
            .action_factorization()
            .map(|spec| spec.pad_width)
            .filter(|&w| w > 0)
    }

    /// Return a clean Python error if the worker threads have been shut down (post-`close`),
    /// instead of letting a channel `send`/`recv` panic (which surfaces as a PanicException).
    fn ensure_open(&self) -> PyResult<()> {
        if self._worker_handles.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "environment pool has been closed",
            ));
        }
        Ok(())
    }

    /// Build a pool from a user-supplied env factory with every pool option — the entry point
    /// for **external env crates**, with the same options as the Python constructor. Depend on
    /// the `_rustpool` crate, implement [`RlEnv`] for your env, and pass a
    /// `Box<dyn Fn(u64) -> Box<dyn RlEnv>>` here; expose the returned `RlPool` from your own
    /// `#[pymodule]` to get the full async + MCTS + factored-action engine. As in the Python
    /// constructor, env `i` is built as `factory(options.seed + i)`. See the README "Writing your
    /// own environment" section.
    pub fn new_with_options(
        num_envs: usize,
        batch_size: usize,
        factory: EnvFactory,
        options: PoolOptions,
    ) -> PyResult<Self> {
        let PoolOptions {
            seed,
            num_threads,
            dispatch,
            autoreset,
        } = options;
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be >= 1"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be >= 1"));
        }
        if batch_size > num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch_size ({}) cannot be greater than num_envs ({})",
                batch_size, num_envs
            )));
        }

        let (env_template, obs_spec, obs_key_names) = Python::attach(|py| {
            let env_template = factory(seed);
            let obs_spec_obj = env_template.observation_spec(py)?;
            let obs_spec = parse_obs_spec(py, &obs_spec_obj)?;
            let obs_key_names = env_template.obs_keys();
            PyResult::Ok((env_template, obs_spec, obs_key_names))
        })?;

        let num_threads =
            threads::pool_num_threads(num_envs, num_threads).map_err(PyValueError::new_err)?;

        let mut shards: Vec<HashMap<usize, Box<dyn RlEnv>>> =
            (0..num_threads).map(|_| HashMap::new()).collect();

        for env_id in 0..num_envs {
            let shard_id = env_id % num_threads;
            let env = factory(seed.wrapping_add(env_id as u64));
            shards[shard_id].insert(env_id, env);
        }

        let (state_tx, state_rx) = unbounded();
        let mut action_txs = Vec::with_capacity(num_threads);
        let mut worker_handles = Vec::with_capacity(num_threads);

        for shard_id in 0..num_threads {
            let (action_tx, action_rx) = unbounded();
            action_txs.push(action_tx);
            let state_tx_clone = state_tx.clone();
            let shard_envs = shards.remove(0);

            let handle = thread::spawn(move || {
                worker_loop(shard_id, shard_envs, action_rx, state_tx_clone, autoreset);
            });
            worker_handles.push(handle);
        }

        Ok(Self {
            num_threads,
            action_txs,
            state_rx,
            message_buffer: Mutex::new(VecDeque::new()),
            _env_template: env_template,
            obs_spec,
            obs_key_names,
            num_envs,
            batch_size,
            _worker_handles: Some(worker_handles),
            registry: StateRegistry::new(),
            dispatch,
            autoreset,
        })
    }

    /// [`RlPool::new_with_options`] with the default options (seed 0, so env `i` is
    /// `factory(i)`; threads from `RUSTPOOL_NUM_THREADS` / the CPUs; `dispatch="auto"`;
    /// `autoreset="same_step"`). Kept for existing external crates.
    pub fn new_with_factory(
        num_envs: usize,
        batch_size: usize,
        factory: EnvFactory,
    ) -> PyResult<Self> {
        Self::new_with_options(num_envs, batch_size, factory, PoolOptions::default())
    }

    /// [`RlPool::new_with_factory`] with an explicit worker-thread count. Kept for existing
    /// external crates; [`RlPool::new_with_options`] takes every option.
    pub fn new_with_factory_and_threads(
        num_envs: usize,
        batch_size: usize,
        factory: EnvFactory,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let options = PoolOptions {
            num_threads,
            ..PoolOptions::default()
        };
        Self::new_with_options(num_envs, batch_size, factory, options)
    }
}

/// The pool's name before 0.2.0, kept so existing external crates still compile.
#[deprecated(since = "0.2.0", note = "renamed to `RlPool`")]
pub type MyEnvPool = RlPool;

impl Drop for RlPool {
    fn drop(&mut self) {
        if let Some(handles) = self._worker_handles.take() {
            for tx in &self.action_txs {
                let _ = tx.send(WorkerAction::Close);
            }
            for handle in handles {
                let _ = handle.join();
            }
        }
    }
}

// --- MODULE REGISTRATION ---

/// The task ids registered in the built-in environment registry, sorted:
/// ``["BinPack-v0", "CartPole-v1", "Maze-v0"]``. Pass one as ``task_id`` to ``rustpool.make``.
#[pyfunction]
fn registered_tasks() -> Vec<String> {
    registry::registered_task_ids()
}

/// Test hook (not public API): build a pool the way an external env crate does, through
/// [`RlPool::new_with_options`] with a factory closure, here wrapping a built-in env so the
/// external-crate path can be tested from Python without compiling a second extension module.
#[pyfunction]
#[pyo3(signature = (task_id, num_envs=4, batch_size=4, *, seed=0, num_threads=None, dispatch="auto", autoreset="same_step", **kwargs))]
#[allow(clippy::too_many_arguments)]
fn _pool_from_factory(
    py: Python<'_>,
    task_id: &str,
    num_envs: usize,
    batch_size: usize,
    seed: u64,
    num_threads: Option<usize>,
    dispatch: &str,
    autoreset: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<RlPool> {
    let options = PoolOptions::parse(seed, num_threads, dispatch, autoreset)?;
    let empty_kwargs = PyDict::new(py);
    let inner = registry::create_env_factory(task_id, kwargs.unwrap_or(&empty_kwargs))?;
    let factory: EnvFactory = Box::new(move |s| inner(s));
    RlPool::new_with_options(num_envs, batch_size, factory, options)
}

/// Compiled core of rustpool: a parallel RL environment-execution engine with an
/// asynchronous EnvPool-style API and a state-cloning substrate for batched tree search.
///
/// This native module is an implementation detail; import the public API from the
/// ``rustpool`` package instead (``rustpool.make``, ``rustpool.RlPool``).
#[pymodule]
fn _rustpool(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RlPool>()?;
    m.add("MyEnvPool", m.getattr("RlPool")?)?; // deprecated alias, see `rustpool.MyEnvPool`
    m.add_function(wrap_pyfunction!(registered_tasks, m)?)?;
    m.add_function(wrap_pyfunction!(_pool_from_factory, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
