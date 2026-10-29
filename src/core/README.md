# Core Module Overview

This directory contains the Rust “core” runtime for `rustpool`. Each module is focused on a single responsibility and is designed to be reused by the PyO3 binding layer in `src/lib.rs`.

## Modules

### batching.rs
Utilities for **batching** observations and action masks coming from multiple environments.

- **ObsDtype / ObsSpecEntry**: Describe observation dtypes and shapes used for validation.
- **stack_action_masks**: Validates mask lengths and stacks them into a 2D NumPy array.
- **stack_fixed_width_masks**: Stacks rows already validated to one width (``query_state_mask``).
- **stack_observations**: Validates observation shapes/types and stacks per-key arrays into a dict of batched NumPy arrays.

### mcts.rs
State snapshot management used by MCTS-style simulation.

- **StateRegistry**: Stores cloned environment states and issues integer `state_id`s. Provides:
  - `snapshot(envs) -> Vec<i32>` to register states and return IDs
  - `get_clones(ids)` to retrieve clones for simulation
  - `conditional_masks(ids, prefixes)` to read factored masks of stored states without cloning
  - `release(ids)` to drop stored states, and `clear()` when the pool closes

### rl_env.rs
Trait definitions for environments used by the runtime.

- **RlEnv**: Core environment interface (reset/step/specs/obs keys).
- **RlEnvClone**: Enables cloning boxed trait objects.

### worker.rs
Threaded worker loop and message types for stepping environments in parallel.

- **WorkerAction / WorkerMessage**: Commands/results sent across channels: one message per env
  (`Step`, `QueryMask`) or one per worker per call (`StepBatch`, `QueryMaskBatch`), chosen by
  `Dispatch`.
- **AutoReset**: `same_step` or `next_step` auto-reset, applied in the worker.
- **worker_loop**: Executes actions and returns step, snapshot, query and mask results.

### threads.rs
Worker-thread count: the `num_threads` argument, else `RUSTPOOL_NUM_THREADS`, else the CPUs the
process may use, capped at `num_envs`.

### types.rs
Core data types used across the runtime.

- **GenericObs / ArrayData**: Typed observation containers used by batching and envs.

## How it fits together

1. `lib.rs` (PyO3 binding layer) dispatches work to `worker.rs` and uses `batching.rs` to stack results.
2. `worker.rs` owns per-thread environment shards and sends `WorkerMessage` results back.
3. `mcts.rs` stores cloned states for fast, parallel simulation.
4. `rl_env.rs` and `types.rs` define the shared interfaces and data containers.
