# Environments

Built-in tasks (see `rustpool.list_envs()`). Construct with
`rustpool.make(task_id, num_envs=..., batch_size=..., **kwargs)`: the pool options and the
environment's config below go in the same call. The config is deserialised by serde in Rust: an
unknown key or a mistyped value raises `TypeError`, an out-of-range value `ValueError`.

## `Maze-v0`

Grid navigation: the agent starts at a uniformly random cell (possibly the goal itself, in which
case the next step pays `+1` whatever the action) and must reach the fixed corner goal
`(width-1, height-1)`. Sparse reward `+1` at the goal. 4 actions (0 up, 1 down, 2 left, 3 right)
with a boundary mask; an out-of-bounds move ends the episode without moving, so masking
matters. A fast learning/sanity task.

| kwarg | default | meaning |
| --- | --- | --- |
| `width`, `height` | 10, 10 | grid size |
| `max_episode_steps` | 100 | truncation horizon |

Observation: `{"agent_pos": float32[2]}`. Action space: `(4,)`.

## `CartPole-v1`

The classic cart-pole balancing task: a port of gymnasium's `CartPoleEnv` (same constants,
Euler integration on an f64 state, uniform `±0.05` reset), with gymnasium's 500-step time limit
built in. Reward `1.0` on every step; `done` when the pole leans past 12° or the cart leaves
`±2.4`, or at the step limit (rustpool has a single `done`; truncation is not flagged
separately). Two actions (0 = push left, 1 = push right), no action mask. A near-zero-compute
env for measuring engine overhead; next states match gymnasium's to within 1e-6 over 10,000 random
transitions in the tests (`query_state(env_id, "state")` returns the exact f64 state).

| kwarg | default | meaning |
| --- | --- | --- |
| `max_episode_steps` | 500 | truncation horizon |

Observation: `{"obs": float32[4]}` = `(x, x_dot, theta, theta_dot)`. Action space: `(2,)`.

## `BinPack-v0`

3D container loading via Empty-Maximal-Spaces (EMS): place items into one or more 20-ft
containers, choosing `(item, orientation, container, ems)` per step (orientation and container
only when `orientations > 1` / `num_containers > 1`). Dense reward = placed-item volume / total
capacity (`num_containers` containers; episodic return ≈ fraction filled); sparse reward credits
the final packed fraction.

| kwarg | default | meaning |
| --- | --- | --- |
| `max_items` | 25 | max items **per container** (`max_items * num_containers` item slots) |
| `max_ems` | 50 | max empty-maximal-spaces tracked **per container** |
| `num_containers` | 1 | containers `C`; one instance is generated per container |
| `orientations` | 1 | orientations per item `O` ∈ {1, 2, 6}: 2 = rotation about the vertical axis, 6 = all axis-aligned |
| `shuffle_items` | `False` | shuffle the generated items (with their ground-truth locations) at every reset; otherwise items come in generation order, one container's instance after another, so at `C > 1` a slot reveals its instance |
| `factored` | `False` | expose the factored action chain (`action_factorization()` / `query_mask`) |
| `emit_flat_mask` | `not factored` | emit the flat mask in `info["action_mask"]`; when off, a factored env emits the stage-0 (item) mask padded to `pad_width` instead, a flat one `[B, 0]` |
| `reward_type` | `"dense"` | `"dense"` or `"sparse"` |
| `split_eps`, `prob_split_one_item`, `split_num_same_items` | — | instance generator controls |

Observation: a dict of `items` / `items_mask` / `placed_mask` / `items_location` /
`ground_truth_locations` (one row per item slot), `ems` / `ems_mask` (`num_containers` blocks of
`max_ems` rows, container `c` at rows `c*max_ems..`), plus `items_container` (when
`num_containers > 1`) and `items_orientation` (when `orientations > 1`), int32, `-1` while
unplaced. Action space: `(N * O * C * max_ems,)` with `N = max_items * C` item slots; the flat
action is the mixed radix `((item * O + o) * C + c) * max_ems + ems` (`item * max_ems + ems` at the
defaults), with a validity mask in `info["action_mask"]`. At the defaults (`C = O = 1`) the env is
bit-identical to the single-container env (pinned by `tests/test_binpack_anchor.py`).

## Writing your own

Environments are Rust types implementing the `RlEnv` trait, either
[inside rustpool](guides/env-in-rustpool.md) (a new task of `rustpool.make`) or
[in your own crate](guides/own-crate.md). Any `Clone` environment gets `snapshot` /
`simulate_batch` / `release_batch` for free.
