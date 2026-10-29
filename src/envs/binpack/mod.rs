use crate::core::rl_env::{FactorizationSpec, RlEnv};
use crate::core::types::{ArrayData, GenericObs};
use crate::registry::{parse_env_kwargs, EnvFactory, EnvParser};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

mod bench;

// --- Constants (20ft Container in mm) ---
const CONT_W: i32 = 5870;
const CONT_D: i32 = 2330;
const CONT_H: i32 = 2200;

// --- Data Structures (Pure Rust) ---

#[derive(Debug, Clone, Copy)]
pub struct BoxItem {
    pub w: i32,
    pub d: i32,
    pub h: i32,
}

impl BoxItem {
    fn volume(&self) -> f32 {
        ((self.w as i64) * (self.d as i64) * (self.h as i64)) as f32
    }

    /// The item's `(w, d, h)` under orientation `o`. `0..2` are the two upright orientations
    /// (rotation about the vertical axis), `0..6` all axis-aligned ones; `orientations = O`
    /// exposes `0..O`.
    fn oriented(&self, o: usize) -> (i32, i32, i32) {
        let (w, d, h) = (self.w, self.d, self.h);
        match o {
            0 => (w, d, h),
            1 => (d, w, h),
            2 => (w, h, d),
            3 => (h, w, d),
            4 => (d, h, w),
            _ => (h, d, w),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Space {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
    pub d: i32,
    pub h: i32,
}

impl Space {
    fn volume(&self) -> f32 {
        ((self.w as i64) * (self.d as i64) * (self.h as i64)) as f32
    }

    fn can_fit(&self, w: i32, d: i32, h: i32) -> bool {
        w <= self.w && d <= self.d && h <= self.h
    }

    fn contains(&self, inner: &Space) -> bool {
        inner.x >= self.x
            && inner.y >= self.y
            && inner.z >= self.z
            && inner.x + inner.w <= self.x + self.w
            && inner.y + inner.d <= self.y + self.d
            && inner.z + inner.h <= self.z + self.h
    }

    pub fn intersect(&self, other: &Space) -> bool {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let z1 = self.z.max(other.z);

        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.d).min(other.y + other.d);
        let z2 = (self.z + self.h).min(other.z + other.h);

        (x2 > x1) && (y2 > y1) && (z2 > z1)
    }
}

// --- Configuration ---

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
pub struct BinPackConfig {
    pub max_items: usize,
    pub max_ems: usize,
    pub split_eps: f32,
    pub prob_split_one_item: f64,
    pub split_num_same_items: usize,
    /// Opt-in: advertise a 2-factor (item -> ems) autoregressive action space via
    /// `action_factorization()` / `conditional_mask()`. The flat `step` encoding is unchanged.
    pub factored: bool,
    /// Emit the flat `max_items * max_ems` action mask in `info["action_mask"]`. A factored
    /// consumer reads masks via `query_mask` instead and never touches it, so turning this
    /// off removes both its O(items*ems) construction and its per-step transport; a factored
    /// env then emits the stage-0 (item) mask padded to `pad_width` in its place. From
    /// Python it defaults to `!factored` (see `BinPackKwargs`); pass it explicitly to override,
    /// e.g. `factored=True, emit_flat_mask=True` for tree search over the flat action space.
    pub emit_flat_mask: bool,
    /// Number of identical containers `C`. Items: `max_items` generated per container
    /// (`max_items * C` slots); free spaces: a separate top-`max_ems` EMS list per container.
    pub num_containers: usize,
    /// Allowed orientations per item `O` in {1, 2, 6} (see `BoxItem::oriented`).
    pub orientations: usize,
    /// Shuffle the generated items (and their ground-truth locations with them) with the env
    /// RNG at every reset. Off by default: items then come in generation order, one
    /// container's instance after the other, so at `C > 1` an item's slot reveals which
    /// container's perfect packing it belongs to.
    pub shuffle_items: bool,
}

fn default_max_items() -> usize {
    25
}
fn default_max_ems() -> usize {
    50
}
fn default_split_eps() -> f32 {
    0.3
}
fn default_prob_split_one_item() -> f64 {
    0.7
}
fn default_split_num_same_items() -> usize {
    5
}
fn default_factored() -> bool {
    false
}
fn default_emit_flat_mask() -> bool {
    true
}
fn default_one() -> usize {
    1
}

impl Default for BinPackConfig {
    fn default() -> Self {
        Self {
            max_items: default_max_items(),
            max_ems: default_max_ems(),
            split_eps: default_split_eps(),
            prob_split_one_item: default_prob_split_one_item(),
            split_num_same_items: default_split_num_same_items(),
            factored: default_factored(),
            emit_flat_mask: default_emit_flat_mask(),
            num_containers: 1,
            orientations: 1,
            shuffle_items: false,
        }
    }
}

impl BinPackConfig {
    /// Item slots: `max_items` per container.
    pub fn item_slots(&self) -> usize {
        self.max_items * self.num_containers
    }

    /// Flat action count `items * O * C * max_ems`; the flat action is the mixed-radix
    /// `((item * O + o) * C + c) * max_ems + ems` (`item * max_ems + ems` at `O = C = 1`).
    pub fn num_actions(&self) -> usize {
        self.item_slots() * self.orientations * self.num_containers * self.max_ems
    }

    fn decode(&self, action: usize) -> (usize, usize, usize, usize) {
        let (me, c_n, o_n) = (self.max_ems, self.num_containers, self.orientations);
        let ems = action % me;
        let rest = action / me;
        let c = rest % c_n;
        let rest = rest / c_n;
        (rest / o_n, rest % o_n, c, ems)
    }

    fn encode(&self, item: usize, o: usize, c: usize, ems: usize) -> usize {
        ((item * self.orientations + o) * self.num_containers + c) * self.max_ems + ems
    }
}

fn default_reward_type() -> String {
    "dense".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BinPackKwargs {
    #[serde(default = "default_max_items")]
    max_items: usize,
    #[serde(default = "default_max_ems")]
    max_ems: usize,
    #[serde(default = "default_split_eps")]
    split_eps: f32,
    #[serde(default = "default_prob_split_one_item")]
    prob_split_one_item: f64,
    #[serde(default = "default_split_num_same_items")]
    split_num_same_items: usize,
    #[serde(default = "default_reward_type")]
    reward_type: String,
    #[serde(default = "default_factored")]
    factored: bool,
    /// `None` (not passed) resolves to `!factored`: a factored env skips the flat mask unless
    /// the caller asks for it.
    #[serde(default)]
    emit_flat_mask: Option<bool>,
    #[serde(default = "default_one")]
    num_containers: usize,
    #[serde(default = "default_one")]
    orientations: usize,
    #[serde(default)]
    shuffle_items: bool,
}

fn parse_bin_pack_config(kwargs: &Bound<'_, PyDict>) -> PyResult<EnvFactory> {
    let args: BinPackKwargs = parse_env_kwargs("BinPack-v0", kwargs)?;
    if args.max_ems < 1 {
        return Err(PyValueError::new_err("max_ems must be >= 1"));
    }
    if !(0.0..=1.0).contains(&args.prob_split_one_item) {
        return Err(PyValueError::new_err(
            "prob_split_one_item must be in [0, 1]",
        ));
    }
    if !(0.0..=0.5).contains(&args.split_eps) {
        return Err(PyValueError::new_err("split_eps must be in [0, 0.5]"));
    }
    if args.num_containers < 1 {
        return Err(PyValueError::new_err("num_containers must be >= 1"));
    }
    if ![1, 2, 6].contains(&args.orientations) {
        return Err(PyValueError::new_err("orientations must be 1, 2 or 6"));
    }

    let config = BinPackConfig {
        max_items: args.max_items,
        max_ems: args.max_ems,
        split_eps: args.split_eps,
        prob_split_one_item: args.prob_split_one_item,
        split_num_same_items: args.split_num_same_items,
        factored: args.factored,
        emit_flat_mask: args.emit_flat_mask.unwrap_or(!args.factored),
        num_containers: args.num_containers,
        orientations: args.orientations,
        shuffle_items: args.shuffle_items,
    };

    let reward_fn_type = match args.reward_type.to_lowercase().as_str() {
        "dense" => RewardFnType::Dense,
        "sparse" => RewardFnType::Sparse,
        _ => {
            return Err(PyValueError::new_err(
                "reward_type must be either 'dense' or 'sparse'",
            ))
        }
    };

    Ok(Box::new(move |seed| {
        Box::new(BinPackEnv::new(config, seed, reward_fn_type))
    }))
}

pub fn register_env(registry: &mut HashMap<String, EnvParser>) {
    registry.insert("BinPack-v0".to_string(), parse_bin_pack_config);
}

// --- Rewards ---

#[derive(Debug, Clone, Copy)]
pub enum RewardFnType {
    Dense,
    Sparse,
}

trait RewardFn: Send + Sync + RewardFnClone + 'static {
    // Changed to take Core instead of Env
    fn calculate(
        &self,
        core: &BinPackCore,
        action_successful: bool,
        is_done: bool,
        placed_item_volume: f32,
    ) -> f32;
}

trait RewardFnClone {
    fn clone_reward(&self) -> Box<dyn RewardFn>;
}

impl<T: 'static + RewardFn + Clone> RewardFnClone for T {
    fn clone_reward(&self) -> Box<dyn RewardFn> {
        Box::new(self.clone())
    }
}

/// Total packing capacity: `C` containers (exactly the single-container volume at `C = 1`).
fn capacity(core: &BinPackCore) -> f32 {
    (CONT_W as i64 * CONT_D as i64 * CONT_H as i64) as f32 * core.config.num_containers as f32
}

#[derive(Clone)]
struct DenseReward;
impl RewardFn for DenseReward {
    fn calculate(
        &self,
        core: &BinPackCore,
        action_successful: bool,
        _is_done: bool,
        placed_item_volume: f32,
    ) -> f32 {
        if action_successful {
            placed_item_volume / capacity(core)
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
struct SparseReward;
impl RewardFn for SparseReward {
    fn calculate(
        &self,
        core: &BinPackCore,
        _action_successful: bool,
        is_done: bool,
        _placed_item_volume: f32,
    ) -> f32 {
        if is_done {
            let total_packed_volume: f32 = core
                .items
                .iter()
                .zip(&core.placed_mask)
                .filter(|(_, &placed)| placed)
                .map(|(item, _)| item.volume())
                .sum();
            total_packed_volume / capacity(core)
        } else {
            0.0
        }
    }
}

// --- Generator (Pure Rust) ---

#[derive(Clone)]
struct InstanceGenerator {
    config: BinPackConfig,
    rng: StdRng,
}

impl InstanceGenerator {
    fn new(config: BinPackConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn generate(&mut self) -> (Vec<BoxItem>, Vec<[i32; 3]>) {
        if self.config.max_items == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut active_spaces = vec![Space {
            x: 0,
            y: 0,
            z: 0,
            w: CONT_W,
            d: CONT_D,
            h: CONT_H,
        }];

        while active_spaces.len() < self.config.max_items {
            let candidate_indices: Vec<usize> = active_spaces
                .iter()
                .enumerate()
                .filter(|(_, s)| s.w > 0 && s.d > 0 && s.h > 0)
                .map(|(i, _)| i)
                .collect();

            if candidate_indices.is_empty() {
                break;
            }

            let rand_idx = self.rng.gen_range(0..candidate_indices.len());
            let index_to_remove = candidate_indices[rand_idx];
            let space_to_split = active_spaces.remove(index_to_remove);

            let min_dim = 100;
            let mut axes = Vec::new();
            if space_to_split.w > min_dim {
                axes.push('w');
            }
            if space_to_split.d > min_dim {
                axes.push('d');
            }
            if space_to_split.h > min_dim {
                axes.push('h');
            }

            if axes.is_empty() {
                active_spaces.push(space_to_split);
                if active_spaces
                    .iter()
                    .all(|s| s.w <= min_dim && s.d <= min_dim && s.h <= min_dim)
                {
                    break;
                }
                continue;
            }

            let axis = *axes.choose(&mut self.rng).unwrap();

            let new_spaces = if self.rng.gen_bool(self.config.prob_split_one_item) {
                self.split_once(&space_to_split, axis)
            } else {
                let max_pieces = self.config.max_items - active_spaces.len();
                let upper = std::cmp::min(self.config.split_num_same_items, max_pieces);

                if upper < 2 {
                    self.split_once(&space_to_split, axis)
                } else {
                    let pieces = self.rng.gen_range(2..=upper);
                    self.split_multiple(&space_to_split, axis, pieces)
                }
            };

            active_spaces.extend(new_spaces);
        }

        let valid_spaces: Vec<_> = active_spaces
            .into_iter()
            .filter(|s| s.w > 0 && s.d > 0 && s.h > 0)
            .collect();

        let items = valid_spaces
            .iter()
            .map(|s| BoxItem {
                w: s.w,
                d: s.d,
                h: s.h,
            })
            .collect();

        let locations = valid_spaces.iter().map(|s| [s.x, s.y, s.z]).collect();

        (items, locations)
    }

    fn split_once(&mut self, space: &Space, axis: char) -> Vec<Space> {
        let (length, mut s1, mut s2) = match axis {
            'w' => (space.w, *space, *space),
            'd' => (space.d, *space, *space),
            _ => (space.h, *space, *space),
        };

        let min_split = (self.config.split_eps * length as f32) as i32;
        let max_split = length - min_split;

        if min_split >= max_split {
            return vec![*space];
        }

        let split_pos = self.rng.gen_range(min_split..max_split);

        match axis {
            'w' => {
                s1.w = split_pos;
                s2.x += split_pos;
                s2.w -= split_pos;
            }
            'd' => {
                s1.d = split_pos;
                s2.y += split_pos;
                s2.d -= split_pos;
            }
            _ => {
                s1.h = split_pos;
                s2.z += split_pos;
                s2.h -= split_pos;
            }
        }
        vec![s1, s2]
    }

    fn split_multiple(&mut self, space: &Space, axis: char, num_splits: usize) -> Vec<Space> {
        let mut new_spaces = Vec::with_capacity(num_splits);

        let (length, current_space) = match axis {
            'w' => (space.w, *space),
            'd' => (space.d, *space),
            _ => (space.h, *space),
        };

        let step = length / num_splits as i32;
        if step == 0 {
            return vec![*space];
        }

        for i in 0..num_splits {
            let mut new_space = current_space;
            match axis {
                'w' => {
                    new_space.x = space.x + i as i32 * step;
                    new_space.w = if i == num_splits - 1 {
                        length - (i as i32 * step)
                    } else {
                        step
                    };
                }
                'd' => {
                    new_space.y = space.y + i as i32 * step;
                    new_space.d = if i == num_splits - 1 {
                        length - (i as i32 * step)
                    } else {
                        step
                    };
                }
                _ => {
                    new_space.z = space.z + i as i32 * step;
                    new_space.h = if i == num_splits - 1 {
                        length - (i as i32 * step)
                    } else {
                        step
                    };
                }
            }
            new_spaces.push(new_space);
        }
        new_spaces
    }
}

// --- BinPackCore: Pure Rust Environment Logic ---
// Separated from the PyO3 wrapper to allow testing without linking Python.

fn full_container() -> Space {
    Space {
        x: 0,
        y: 0,
        z: 0,
        w: CONT_W,
        d: CONT_D,
        h: CONT_H,
    }
}

/// One stage of the factored action chain. Stages with a single option (`O = 1`, `C = 1`)
/// are not active, so at the defaults the chain is `item -> ems`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Factor {
    Item,
    Orientation,
    Container,
    Ems,
}

#[derive(Clone)]
pub struct BinPackCore {
    pub config: BinPackConfig,
    generator: InstanceGenerator,
    // Public fields for direct access in tests/wrapper
    pub items: Vec<BoxItem>,
    pub placed_mask: Vec<bool>,
    /// Placement corner inside the item's container.
    pub items_location: Vec<[i32; 3]>,
    /// Container / orientation of each placed item (`-1` while unplaced).
    pub items_container: Vec<i32>,
    pub items_orientation: Vec<i32>,
    pub ground_truth_locations: Vec<[i32; 3]>,
    /// Free spaces (EMS) per container.
    pub free_spaces: Vec<Vec<Space>>,
    /// Per container: volume-descending, `max_ems`-truncated view of its `free_spaces`.
    /// Invalidated exactly where that container's `free_spaces` is written, so a cache hit is
    /// identical to a recompute.
    sorted_cache: Vec<Option<Vec<Space>>>,
}

impl BinPackCore {
    pub fn new(config: BinPackConfig, seed: u64) -> Self {
        Self {
            config,
            generator: InstanceGenerator::new(config, seed),
            items: Vec::new(),
            placed_mask: Vec::new(),
            items_location: Vec::new(),
            items_container: Vec::new(),
            items_orientation: Vec::new(),
            ground_truth_locations: Vec::new(),
            // Valid empty containers, as `reset` builds them: before the first reset the env is
            // an instance with no items (a step is an invalid action that ends the episode, and
            // a snapshot is safe to fork) instead of a state that panics when touched.
            free_spaces: vec![vec![full_container()]; config.num_containers],
            sorted_cache: vec![None; config.num_containers],
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) -> Vec<bool> {
        // Re-seed the generator if seed is provided
        if let Some(s) = seed {
            self.generator = InstanceGenerator::new(self.config, s);
        }

        // One generated instance per container, drawn in sequence from the same stream (at
        // C = 1 exactly the single-container draw).
        self.items.clear();
        self.ground_truth_locations.clear();
        for _ in 0..self.config.num_containers {
            let (items, locations) = self.generator.generate();
            self.items.extend(items);
            self.ground_truth_locations.extend(locations);
        }
        if self.config.shuffle_items {
            // One permutation for both, drawn from the generator's stream after the instances.
            let mut perm: Vec<usize> = (0..self.items.len()).collect();
            perm.shuffle(&mut self.generator.rng);
            self.items = perm.iter().map(|&i| self.items[i]).collect();
            self.ground_truth_locations = perm
                .iter()
                .map(|&i| self.ground_truth_locations[i])
                .collect();
        }
        let n = self.items.len();
        self.placed_mask = vec![false; n];
        self.items_location = vec![[0, 0, 0]; n];
        self.items_container = vec![-1; n];
        self.items_orientation = vec![-1; n];
        self.free_spaces = vec![vec![full_container()]; self.config.num_containers];
        self.sorted_cache = vec![None; self.config.num_containers];

        self.get_action_mask()
    }

    /// Place `item` with orientation `o` into EMS `ems` of container `c`; returns the flat mask.
    pub fn step(&mut self, item: usize, o: usize, c: usize, ems: usize) -> (bool, f32, Vec<bool>) {
        let (placed_successfully, placed_volume) = self.place_item(item, o, c, ems);
        let mask = self.get_action_mask();
        (placed_successfully, placed_volume, mask)
    }

    fn fits(&self, item: usize, o: usize, s: &Space) -> bool {
        let (w, d, h) = self.items[item].oriented(o);
        s.can_fit(w, d, h)
    }

    /// The flat mask, `num_actions` entries in `BinPackConfig::encode` order.
    fn get_action_mask(&mut self) -> Vec<bool> {
        let cfg = self.config;
        let mut action_mask = vec![false; cfg.num_actions()];
        self.ensure_sorted();
        for item_idx in 0..self.items.len() {
            if self.placed_mask[item_idx] {
                continue;
            }
            for o in 0..cfg.orientations {
                let (w, d, h) = self.items[item_idx].oriented(o);
                for c in 0..cfg.num_containers {
                    let ems = self.sorted_cache[c].as_deref().unwrap();
                    let base = cfg.encode(item_idx, o, c, 0);
                    for (ems_idx, space) in ems.iter().enumerate() {
                        if space.can_fit(w, d, h) {
                            action_mask[base + ems_idx] = true;
                        }
                    }
                }
            }
        }
        action_mask
    }

    fn compute_sorted(free: &[Space], max_ems: usize) -> Vec<Space> {
        let mut s = free.to_vec();
        s.sort_by(|a, b| b.volume().partial_cmp(&a.volume()).unwrap());
        s.truncate(max_ems);
        s
    }

    /// Existence test for "some unplaced item fits some top-`max_ems` free space" (any
    /// orientation, any container). Same predicate as `get_action_mask().iter().any(..)`, but
    /// it stops at the first hit instead of filling the whole mask to then reduce it.
    fn any_legal_action(&mut self) -> bool {
        self.ensure_sorted();
        (0..self.items.len()).any(|i| !self.placed_mask[i] && self.item_fits_anywhere(i))
    }

    /// The `action_mask` emitted when the flat mask is off, plus "no legal action left".
    /// Factored: the stage-0 (item) mask padded to the spec's `pad_width` -- the first factor
    /// a factored policy samples, so it needs only the K-1 later stages from `query_mask`.
    /// Flat (`emit_flat_mask=False`): empty, the consumer derives its mask itself.
    /// Either way "no legal action" equals the flat rule: an item is in the stage-0 mask iff
    /// some (orientation, container, EMS) completes it.
    fn unflat_mask(&mut self) -> (Vec<bool>, bool) {
        self.ensure_sorted();
        if !self.config.factored {
            return (Vec::new(), !self.any_legal_action());
        }
        let mut stage0 = self.factor_mask(&[]);
        let none = !stage0.iter().any(|&v| v);
        let pad = self
            .factors()
            .iter()
            .map(|&f| self.factor_width(f))
            .max()
            .unwrap();
        stage0.resize(pad, false);
        (stage0, none)
    }

    /// The top-`max_ems` free spaces of container `c`, largest-volume first (the order
    /// `step`/obs index into).
    fn sorted_ems(&self, c: usize) -> Vec<Space> {
        self.with_sorted(c, |s| s.to_vec())
    }

    /// Borrowing `&self` accessor -- no clone on a cache hit.
    fn with_sorted<R>(&self, c: usize, f: impl FnOnce(&[Space]) -> R) -> R {
        match &self.sorted_cache[c] {
            Some(s) => f(s),
            None => f(&Self::compute_sorted(
                &self.free_spaces[c],
                self.config.max_ems,
            )),
        }
    }

    /// Warm every container's cache. One sort per touched container per step: `update_ems`
    /// invalidates the container it wrote, and every later reader in the step hits.
    fn ensure_sorted(&mut self) {
        for c in 0..self.config.num_containers {
            if self.sorted_cache[c].is_none() {
                self.sorted_cache[c] = Some(Self::compute_sorted(
                    &self.free_spaces[c],
                    self.config.max_ems,
                ));
            }
        }
    }

    // --- Factored action chain: item -> [orientation] -> [container] -> ems ------------
    // Every mask is an existence witness: a choice is valid iff it leaves >= 1 valid
    // completion, so an autoregressive sampler never dead-ends.

    fn fits_in(&self, item: usize, o: usize, c: usize) -> bool {
        let (w, d, h) = self.items[item].oriented(o);
        self.with_sorted(c, |ems| ems.iter().any(|s| s.can_fit(w, d, h)))
    }

    fn orientation_fits(&self, item: usize, o: usize) -> bool {
        (0..self.config.num_containers).any(|c| self.fits_in(item, o, c))
    }

    fn item_fits_anywhere(&self, item: usize) -> bool {
        (0..self.config.orientations).any(|o| self.orientation_fits(item, o))
    }

    fn placeable(&self, item: usize) -> bool {
        item < self.items.len() && !self.placed_mask[item]
    }

    /// The active stages, in chain order.
    fn factors(&self) -> Vec<Factor> {
        let mut f = vec![Factor::Item];
        if self.config.orientations > 1 {
            f.push(Factor::Orientation);
        }
        if self.config.num_containers > 1 {
            f.push(Factor::Container);
        }
        f.push(Factor::Ems);
        f
    }

    fn factor_width(&self, f: Factor) -> usize {
        match f {
            Factor::Item => self.config.item_slots(),
            Factor::Orientation => self.config.orientations,
            Factor::Container => self.config.num_containers,
            Factor::Ems => self.config.max_ems,
        }
    }

    /// Mask over the next active stage given the already-chosen prefix `chosen` (one value per
    /// earlier active stage). Inactive stages are fixed at 0.
    fn factor_mask(&self, chosen: &[usize]) -> Vec<bool> {
        let factors = self.factors();
        let Some(&next) = factors.get(chosen.len()) else {
            return Vec::new();
        };
        let (mut item, mut o, mut c) = (0usize, 0usize, 0usize);
        for (f, &v) in factors.iter().zip(chosen) {
            match f {
                Factor::Item => item = v,
                Factor::Orientation => o = v,
                Factor::Container => c = v,
                Factor::Ems => {}
            }
        }
        let width = self.factor_width(next);
        if next != Factor::Item && !self.placeable(item) {
            return vec![false; width];
        }
        let o_ok = o < self.config.orientations;
        let c_ok = c < self.config.num_containers;
        match next {
            Factor::Item => (0..width)
                .map(|i| self.placeable(i) && self.item_fits_anywhere(i))
                .collect(),
            Factor::Orientation => (0..width)
                .map(|oo| self.orientation_fits(item, oo))
                .collect(),
            Factor::Container => (0..width)
                .map(|cc| o_ok && self.fits_in(item, o, cc))
                .collect(),
            Factor::Ems => {
                let mut mask = vec![false; width];
                if o_ok && c_ok {
                    self.with_sorted(c, |ems| {
                        for (j, s) in ems.iter().enumerate() {
                            mask[j] = self.fits(item, o, s);
                        }
                    });
                }
                mask
            }
        }
    }

    fn place_item(&mut self, item_idx: usize, o: usize, c: usize, ems_idx: usize) -> (bool, f32) {
        if c >= self.config.num_containers || o >= self.config.orientations {
            return (false, 0.0);
        }
        // `ems_idx` is always `< max_ems` (it comes from `action % max_ems`), so indexing the
        // truncated cache picks the same element the full sort would have.
        self.ensure_sorted();
        let sorted_ems = self.sorted_cache[c].as_deref().unwrap();

        if item_idx >= self.items.len() || ems_idx >= sorted_ems.len() || self.placed_mask[item_idx]
        {
            return (false, 0.0);
        }

        let (w, d, h) = self.items[item_idx].oriented(o);
        let target_space = sorted_ems[ems_idx];

        if target_space.can_fit(w, d, h) {
            let placed_item_space = Space {
                x: target_space.x,
                y: target_space.y,
                z: target_space.z,
                w,
                d,
                h,
            };

            self.placed_mask[item_idx] = true;
            self.items_location[item_idx] = [target_space.x, target_space.y, target_space.z];
            self.items_container[item_idx] = c as i32;
            self.items_orientation[item_idx] = o as i32;

            self.update_ems(c, placed_item_space);

            (true, placed_item_space.volume())
        } else {
            (false, 0.0)
        }
    }

    fn update_ems(&mut self, c: usize, placed: Space) {
        let free = &self.free_spaces[c];
        let mut new_ems_list = Vec::with_capacity(free.len() * 2);
        // Parallel flags: `true` = newly created fragment, `false` = untouched carry-over.
        let mut is_frag: Vec<bool> = Vec::with_capacity(free.len() * 2);

        for space in free.iter() {
            if space.intersect(&placed) {
                if placed.x > space.x {
                    let mut frag = *space;
                    frag.w = placed.x - space.x;
                    new_ems_list.push(frag);
                    is_frag.push(true);
                }
                if placed.x + placed.w < space.x + space.w {
                    let mut frag = *space;
                    frag.x = placed.x + placed.w;
                    frag.w = (space.x + space.w) - frag.x;
                    new_ems_list.push(frag);
                    is_frag.push(true);
                }
                if placed.y > space.y {
                    let mut frag = *space;
                    frag.d = placed.y - space.y;
                    new_ems_list.push(frag);
                    is_frag.push(true);
                }
                if placed.y + placed.d < space.y + space.d {
                    let mut frag = *space;
                    frag.y = placed.y + placed.d;
                    frag.d = (space.y + space.d) - frag.y;
                    new_ems_list.push(frag);
                    is_frag.push(true);
                }
                if placed.z > space.z {
                    let mut frag = *space;
                    frag.h = placed.z - space.z;
                    new_ems_list.push(frag);
                    is_frag.push(true);
                }
                if placed.z + placed.h < space.z + space.h {
                    let mut frag = *space;
                    frag.z = placed.z + placed.h;
                    frag.h = (space.z + space.h) - frag.z;
                    new_ems_list.push(frag);
                    is_frag.push(true);
                }
            } else {
                new_ems_list.push(*space);
                is_frag.push(false);
            }
        }

        let mut maximal_spaces = Vec::with_capacity(new_ems_list.len());

        for (i, s1) in new_ems_list.iter().enumerate() {
            // An untouched space can never be contained in anything in the new list. It
            // survived the previous prune, so no other untouched space contains it; and every
            // fragment F is a sub-box of some space S that intersected `placed`, so U subset-of F
            // would give U subset-of S -- which that same prune already ruled out. Only fragments
            // need the scan, which makes the prune output-sensitive in |fragments| rather
            // than quadratic in |free_spaces|.
            if !is_frag[i] {
                maximal_spaces.push(*s1);
                continue;
            }
            let mut is_redundant = false;
            for (j, s2) in new_ems_list.iter().enumerate() {
                if i == j {
                    continue;
                }

                if s2.contains(s1) {
                    // FIX: Handle identical spaces.
                    // If s1 == s2, both contains() checks return true.
                    // We must ensure we only remove one of them.
                    // Here, if s1 == s2 and i < j, we treat s1 as unique (not redundant)
                    // and let s2 be removed when the outer loop reaches it.
                    if s1 == s2 && i < j {
                        continue;
                    }

                    is_redundant = true;
                    break;
                }
            }
            if !is_redundant {
                maximal_spaces.push(*s1);
            }
        }

        self.free_spaces[c] = maximal_spaces;
        self.sorted_cache[c] = None;
    }
}

// --- BinPackEnv: The Python Wrapper ---

#[pyclass(skip_from_py_object)]
pub struct BinPackEnv {
    core: BinPackCore,
    reward_fn: Box<dyn RewardFn>,
}

impl Clone for BinPackEnv {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            reward_fn: self.reward_fn.clone_reward(),
        }
    }
}

impl BinPackEnv {
    const OBS_KEYS: [&'static str; 7] = [
        "items",
        "ems",
        "items_mask",
        "placed_mask",
        "ems_mask",
        "items_location",
        "ground_truth_locations",
    ];
    /// Extra keys, present only when the feature is on (so the defaults keep today's obs).
    const CONTAINER_KEY: &'static str = "items_container";
    const ORIENTATION_KEY: &'static str = "items_orientation";

    pub fn new(config: BinPackConfig, seed: u64, reward_type: RewardFnType) -> Self {
        let reward_fn: Box<dyn RewardFn> = match reward_type {
            RewardFnType::Dense => Box::new(DenseReward),
            RewardFnType::Sparse => Box::new(SparseReward),
        };
        Self {
            core: BinPackCore::new(config, seed),
            reward_fn,
        }
    }

    fn build_obs(&self, action_mask: Vec<bool>) -> (GenericObs, Vec<bool>) {
        let cfg = self.core.config;
        let slots = cfg.item_slots();

        let norm_x = |v: i32| (v as f32) / (CONT_W as f32);
        let norm_y = |v: i32| (v as f32) / (CONT_D as f32);
        let norm_z = |v: i32| (v as f32) / (CONT_H as f32);

        // 1. Items
        let mut items = vec![0.0; slots * 3];
        for i in 0..self.core.items.len() {
            items[i * 3] = norm_x(self.core.items[i].w);
            items[i * 3 + 1] = norm_y(self.core.items[i].d);
            items[i * 3 + 2] = norm_z(self.core.items[i].h);
        }

        // 2. EMS: one block of `max_ems` rows per container (container c = rows c*max_ems..)
        let mut ems = vec![0.0; cfg.num_containers * cfg.max_ems * 6];
        let mut ems_mask = vec![false; cfg.num_containers * cfg.max_ems];
        for c in 0..cfg.num_containers {
            let ems_for_obs = self.core.sorted_ems(c);
            for (k, s) in ems_for_obs.iter().enumerate() {
                let i = c * cfg.max_ems + k;
                ems[i * 6] = norm_x(s.x);
                ems[i * 6 + 1] = norm_y(s.y);
                ems[i * 6 + 2] = norm_z(s.z);
                ems[i * 6 + 3] = norm_x(s.x + s.w);
                ems[i * 6 + 4] = norm_y(s.y + s.d);
                ems[i * 6 + 5] = norm_z(s.z + s.h);
                ems_mask[i] = true;
            }
        }

        // 3. Items Mask
        let mut items_mask = vec![false; slots];
        for i in 0..self.core.items.len() {
            items_mask[i] = true;
        }

        // 4. Placed Mask
        let mut placed_mask = vec![false; slots];
        placed_mask[..self.core.placed_mask.len()].copy_from_slice(&self.core.placed_mask);

        // 5. EMS Mask: filled with the EMS blocks above.

        // 6. Items Location (inside the item's container)
        let mut items_location = vec![0.0; slots * 3];
        for i in 0..self.core.items_location.len() {
            items_location[i * 3] = norm_x(self.core.items_location[i][0]);
            items_location[i * 3 + 1] = norm_y(self.core.items_location[i][1]);
            items_location[i * 3 + 2] = norm_z(self.core.items_location[i][2]);
        }

        // 7. Ground Truth
        let mut ground_truth_locations = vec![0.0; slots * 3];
        for i in 0..self.core.ground_truth_locations.len() {
            ground_truth_locations[i * 3] = norm_x(self.core.ground_truth_locations[i][0]);
            ground_truth_locations[i * 3 + 1] = norm_y(self.core.ground_truth_locations[i][1]);
            ground_truth_locations[i * 3 + 2] = norm_z(self.core.ground_truth_locations[i][2]);
        }

        let mut obs = Vec::with_capacity(Self::OBS_KEYS.len() + 2);
        obs.push(ArrayData::Float32(items));
        obs.push(ArrayData::Float32(ems));
        obs.push(ArrayData::Bool(items_mask));
        obs.push(ArrayData::Bool(placed_mask));
        obs.push(ArrayData::Bool(ems_mask));
        obs.push(ArrayData::Float32(items_location));
        obs.push(ArrayData::Float32(ground_truth_locations));

        // 8./9. Container / orientation of each placed item (-1 unplaced), only when on.
        let per_slot = |v: &[i32]| {
            let mut out = vec![-1i32; slots];
            out[..v.len()].copy_from_slice(v);
            ArrayData::Int32(out)
        };
        if cfg.num_containers > 1 {
            obs.push(per_slot(&self.core.items_container));
        }
        if cfg.orientations > 1 {
            obs.push(per_slot(&self.core.items_orientation));
        }

        (obs, action_mask)
    }
}

// --- Trait Implementation for Registry ---

impl RlEnv for BinPackEnv {
    fn reset(&mut self, seed: Option<u64>) -> (GenericObs, f32, bool, Vec<bool>) {
        let flat_mask = self.core.reset(seed);
        let (action_mask, no_more_moves) = if self.core.config.emit_flat_mask {
            let any = flat_mask.iter().any(|&v| v);
            (flat_mask, !any)
        } else {
            self.core.unflat_mask()
        };
        let (obs, action_mask) = self.build_obs(action_mask);

        let all_placed = self.core.placed_mask.iter().all(|&v| v);
        let done = all_placed || no_more_moves;

        (obs, 0.0, done, action_mask)
    }

    fn step(&mut self, action: i32) -> (GenericObs, f32, bool, Vec<bool>) {
        let (item_idx, o, c, ems_idx) = self.core.config.decode(action.max(0) as usize);

        let (placed_successfully, placed_volume) = self.core.place_item(item_idx, o, c, ems_idx);
        let (next_action_mask, no_more_moves) = if self.core.config.emit_flat_mask {
            let m = self.core.get_action_mask();
            let any = m.iter().any(|&v| v);
            (m, !any)
        } else {
            self.core.unflat_mask()
        };
        let (obs, next_action_mask) = self.build_obs(next_action_mask);

        let all_placed = self.core.placed_mask.iter().all(|&v| v);
        let done = all_placed || no_more_moves || !placed_successfully;

        let reward = self
            .reward_fn
            .calculate(&self.core, placed_successfully, done, placed_volume);

        (obs, reward, done, next_action_mask)
    }

    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let obs_spec = PyDict::new(py);
        let cfg = self.core.config;
        let max_items = cfg.item_slots();
        let max_ems = cfg.num_containers * cfg.max_ems;

        let float32_str = "float32";
        let bool_str = "bool";

        obs_spec.set_item("items", (vec![max_items, 3], float32_str))?;
        obs_spec.set_item("ems", (vec![max_ems, 6], float32_str))?;
        obs_spec.set_item("items_mask", (vec![max_items], bool_str))?;
        obs_spec.set_item("placed_mask", (vec![max_items], bool_str))?;
        obs_spec.set_item("ems_mask", (vec![max_ems], bool_str))?;
        obs_spec.set_item("items_location", (vec![max_items, 3], float32_str))?;
        obs_spec.set_item("ground_truth_locations", (vec![max_items, 3], float32_str))?;
        if cfg.num_containers > 1 {
            obs_spec.set_item(Self::CONTAINER_KEY, (vec![max_items], "int32"))?;
        }
        if cfg.orientations > 1 {
            obs_spec.set_item(Self::ORIENTATION_KEY, (vec![max_items], "int32"))?;
        }

        Ok(obs_spec.into_any().unbind())
    }

    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok((self.core.config.num_actions(),)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    fn obs_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = Self::OBS_KEYS.iter().map(|k| (*k).to_string()).collect();
        if self.core.config.num_containers > 1 {
            keys.push(Self::CONTAINER_KEY.to_string());
        }
        if self.core.config.orientations > 1 {
            keys.push(Self::ORIENTATION_KEY.to_string());
        }
        keys
    }

    fn action_factorization(&self) -> Option<FactorizationSpec> {
        if !self.core.config.factored {
            return None;
        }
        let factors = self.core.factors();
        let names: Vec<String> = factors
            .iter()
            .map(|f| match f {
                Factor::Item => "item",
                Factor::Orientation => "orientation",
                Factor::Container => "container",
                Factor::Ems => "ems",
            })
            .map(str::to_string)
            .collect();
        let widths: Vec<usize> = factors.iter().map(|&f| self.core.factor_width(f)).collect();
        Some(FactorizationSpec {
            name: format!("binpack_{}", names.join("_")),
            factor_names: names,
            pad_width: *widths.iter().max().unwrap(),
            flat_action_size: widths.iter().product(),
            factor_widths: widths,
            conditional: true,
        })
    }

    /// `"bench:flat" | "bench:chain" | "bench:place"[":reps"]`: mask and placement cost on
    /// the frozen state (see `bench.rs`); the state is not advanced.
    fn query_state(&self, key: &str) -> Option<String> {
        self.core.bench_query(key)
    }

    fn conditional_mask(&self, prefix: &[i32]) -> Vec<bool> {
        if !self.core.config.factored {
            return Vec::new();
        }
        // Factor index = number of already-chosen (non -1) prefix entries, in chain order.
        let chosen: Vec<usize> = prefix
            .iter()
            .copied()
            .filter(|&v| v >= 0)
            .map(|v| v as usize)
            .collect();
        let mut mask = self.core.factor_mask(&chosen);
        // Pad to the spec's pad_width so the batched query_mask transport stacks uniformly.
        let pad = self
            .core
            .factors()
            .iter()
            .map(|&f| self.core.factor_width(f))
            .max()
            .unwrap();
        mask.resize(pad, false);
        mask
    }
}

/// Uniformly pick one `true` index of a boolean mask (None if all false).
#[cfg(test)]
fn sample_true_index(mask: &[bool], rng: &mut StdRng) -> Option<usize> {
    let idxs: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    if idxs.is_empty() {
        None
    } else {
        Some(idxs[rng.gen_range(0..idxs.len())])
    }
}

// --- Python Interface Methods ---

#[pymethods]
impl BinPackEnv {
    // `reset`/`step` are intentionally not exposed to Python here: the worker threads call
    // them through the `RlEnv` trait (returning pure-Rust data), and NumPy conversion happens
    // centrally in `RlPool::recv`. Only the spec accessors are useful directly on an env.

    fn observation_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        RlEnv::observation_spec(self, py)
    }

    fn action_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        RlEnv::action_spec(self, py)
    }
}

// --- Tests (Pure Rust) ---

#[cfg(test)]
mod tests {
    use super::*;

    fn get_space_from_item(item: &BoxItem, loc: &[i32; 3]) -> Space {
        Space {
            x: loc[0],
            y: loc[1],
            z: loc[2],
            w: item.w,
            d: item.d,
            h: item.h,
        }
    }

    #[test]
    fn test_generated_instances_validity() {
        let config = BinPackConfig {
            max_items: 50,
            max_ems: 100,
            split_eps: 0.1,
            prob_split_one_item: 0.5,
            split_num_same_items: 5,
            factored: false,
            emit_flat_mask: true,
            num_containers: 1,
            orientations: 1,
            shuffle_items: false,
        };

        // Loop 100 times as requested
        for seed in 0..100 {
            let mut generator = InstanceGenerator::new(config, seed);
            let (items, locations) = generator.generate();
            let n = items.len();

            assert!(n > 0, "Seed {} generated 0 items", seed);

            // 1. Check Volume Conservation
            let total_item_vol: i64 = items
                .iter()
                .map(|i| i.w as i64 * i.d as i64 * i.h as i64)
                .sum();
            let container_vol = CONT_W as i64 * CONT_D as i64 * CONT_H as i64;
            assert_eq!(
                total_item_vol, container_vol,
                "Seed {}: Total item volume does not match container volume",
                seed
            );

            // 2. Check Bounds and Overlaps
            for i in 0..n {
                let s1 = get_space_from_item(&items[i], &locations[i]);

                assert!(
                    s1.x >= 0 && s1.y >= 0 && s1.z >= 0,
                    "Seed {}: Item {} negative coords",
                    seed,
                    i
                );
                assert!(s1.x + s1.w <= CONT_W, "Seed {}: Item {} exceeds X", seed, i);
                assert!(s1.y + s1.d <= CONT_D, "Seed {}: Item {} exceeds Y", seed, i);
                assert!(s1.z + s1.h <= CONT_H, "Seed {}: Item {} exceeds Z", seed, i);

                for j in (i + 1)..n {
                    let s2 = get_space_from_item(&items[j], &locations[j]);

                    if s1.intersect(&s2) {
                        panic!(
                            "Seed {}: Generator created overlap between Item {} and {}",
                            seed, i, j
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_greedy_episode_no_overlaps() {
        let config = BinPackConfig {
            max_items: 50,
            max_ems: 100,
            split_eps: 0.1,
            prob_split_one_item: 0.5,
            split_num_same_items: 5,
            factored: false,
            emit_flat_mask: true,
            num_containers: 1,
            orientations: 1,
            shuffle_items: false,
        };

        // Loop 100 times as requested
        for seed in 0..100 {
            // Use BinPackCore directly to avoid Python linking in tests
            let mut core = BinPackCore::new(config, seed);
            let mut action_mask = core.reset(None);
            let mut done = false;
            let mut steps = 0;

            while !done {
                steps += 1;

                // Greedy Strategy: Find first valid action
                let action_flat = action_mask.iter().position(|&x| x);

                if let Some(a) = action_flat {
                    let (item_idx, o, c, ems_idx) = config.decode(a);

                    let (placed_success, _, next_mask) = core.step(item_idx, o, c, ems_idx);
                    action_mask = next_mask;

                    if !placed_success {
                        panic!(
                            "Seed {}: Step {}: Action masked valid but failed",
                            seed, steps
                        );
                    }

                    // Check for overlaps among ALL currently placed items
                    let n = core.items.len();
                    for i in 0..n {
                        if !core.placed_mask[i] {
                            continue;
                        }
                        let s1 = get_space_from_item(&core.items[i], &core.items_location[i]);

                        for j in (i + 1)..n {
                            if !core.placed_mask[j] {
                                continue;
                            }
                            let s2 = get_space_from_item(&core.items[j], &core.items_location[j]);

                            if s1.intersect(&s2) {
                                panic!(
                                    "Seed {}: Step {}: Overlap detected between placed items {} and {}", 
                                    seed, steps, i, j
                                );
                            }
                        }
                    }

                    let all_placed = core.placed_mask.iter().all(|&v| v);
                    let no_more_moves = !action_mask.iter().any(|&v| v);
                    done = all_placed || no_more_moves;
                } else {
                    done = true;
                }

                if steps > config.max_items * 2 {
                    panic!("Seed {}: Episode did not terminate", seed);
                }
            }
        }
    }

    fn multi_config(c: usize, o: usize, factored: bool) -> BinPackConfig {
        BinPackConfig {
            max_items: 12,
            max_ems: 16,
            num_containers: c,
            orientations: o,
            factored,
            ..BinPackConfig::default()
        }
    }

    /// `shuffle_items` permutes items and their ground-truth locations together: same
    /// (item, location) pairs as generation order, a different order, deterministic per seed.
    #[test]
    fn test_shuffle_items_permutes_pairs() {
        let pairs = |core: &BinPackCore| -> Vec<(i32, i32, i32, [i32; 3])> {
            core.items
                .iter()
                .zip(&core.ground_truth_locations)
                .map(|(it, loc)| (it.w, it.d, it.h, *loc))
                .collect()
        };
        let plain_cfg = multi_config(3, 1, false);
        let shuffled_cfg = BinPackConfig {
            shuffle_items: true,
            ..plain_cfg
        };
        for seed in 0..20 {
            let mut plain = BinPackCore::new(plain_cfg, seed);
            let mut shuffled = BinPackCore::new(shuffled_cfg, seed);
            let mut again = BinPackCore::new(shuffled_cfg, seed);
            plain.reset(None);
            shuffled.reset(None);
            again.reset(None);
            let (p, s) = (pairs(&plain), pairs(&shuffled));
            assert_eq!(s, pairs(&again), "seed {seed}: not deterministic");
            assert_ne!(p, s, "seed {seed}: order unchanged");
            let (mut ps, mut ss) = (p.clone(), s.clone());
            ps.sort();
            ss.sort();
            assert_eq!(ps, ss, "seed {seed}: different (item, location) pairs");
        }
    }

    /// Random legal episodes with C containers and O orientations: every flat-mask action
    /// places, placements stay inside their container without overlapping (with the rotated
    /// dims), and dense rewards sum to the placed volume over the total capacity.
    #[test]
    fn test_multi_container_orientation_episodes() {
        for (c_n, o_n) in [(2, 1), (1, 2), (3, 6), (4, 2)] {
            let config = multi_config(c_n, o_n, false);
            for seed in 0..20u64 {
                let mut env = BinPackEnv::new(config, seed, RewardFnType::Dense);
                let (_, _, mut done, mut mask) = env.reset(None);
                assert_eq!(mask.len(), config.num_actions());
                assert!(env.core.items.len() <= config.item_slots());
                let mut rng = StdRng::seed_from_u64(seed + 99);
                let mut reward_sum = 0.0f64;
                while !done {
                    let a = sample_true_index(&mask, &mut rng).expect("not done => legal");
                    let (_, r, d, m) = env.step(a as i32);
                    let (item, o, c, _) = config.decode(a);
                    assert!(
                        env.core.placed_mask[item],
                        "legal action {} did not place",
                        a
                    );
                    assert_eq!(env.core.items_container[item], c as i32);
                    assert_eq!(env.core.items_orientation[item], o as i32);
                    reward_sum += r as f64;
                    done = d;
                    mask = m;
                }
                let core = &env.core;
                let placed: Vec<usize> = (0..core.items.len())
                    .filter(|&i| core.placed_mask[i])
                    .collect();
                let boxes: Vec<(i32, Space)> = placed
                    .iter()
                    .map(|&i| {
                        let (w, d, h) = core.items[i].oriented(core.items_orientation[i] as usize);
                        let [x, y, z] = core.items_location[i];
                        (core.items_container[i], Space { x, y, z, w, d, h })
                    })
                    .collect();
                for (k, (ck, sk)) in boxes.iter().enumerate() {
                    assert!(full_container().contains(sk), "out of container");
                    for (cj, sj) in &boxes[k + 1..] {
                        assert!(ck != cj || !sk.intersect(sj), "overlap in container {}", ck);
                    }
                }
                let vol: f64 = placed.iter().map(|&i| core.items[i].volume() as f64).sum();
                let cap = capacity(core) as f64;
                assert!(
                    (reward_sum - vol / cap).abs() < 1e-4,
                    "{} vs {}",
                    reward_sum,
                    vol / cap
                );
            }
        }
    }

    /// The factored chain (item -> [orientation] -> [container] -> ems) is exact: the product
    /// of the chain's conditional masks equals the flat mask entry for entry, at every state.
    #[test]
    fn test_factor_chain_matches_flat_mask() {
        for (c_n, o_n) in [(1, 1), (2, 1), (1, 6), (3, 2)] {
            let config = BinPackConfig {
                emit_flat_mask: true,
                ..multi_config(c_n, o_n, true)
            };
            let mut env = BinPackEnv::new(config, 3, RewardFnType::Dense);
            let (_, _, mut done, mut flat) = env.reset(None);
            let spec = env.action_factorization().unwrap();
            assert_eq!(spec.flat_action_size, config.num_actions());
            let mut rng = StdRng::seed_from_u64(7);
            while !done {
                let mut rebuilt = vec![false; config.num_actions()];
                let item_mask = env.core.factor_mask(&[]);
                for i in 0..config.item_slots() {
                    if !item_mask[i] {
                        continue;
                    }
                    let o_mask = if o_n > 1 {
                        env.core.factor_mask(&[i])
                    } else {
                        vec![true]
                    };
                    for o in 0..o_n {
                        if !o_mask[o] {
                            continue;
                        }
                        let pre: Vec<usize> = if o_n > 1 { vec![i, o] } else { vec![i] };
                        let c_mask = if c_n > 1 {
                            env.core.factor_mask(&pre)
                        } else {
                            vec![true]
                        };
                        for c in 0..c_n {
                            if !c_mask[c] {
                                continue;
                            }
                            let mut full = pre.clone();
                            if c_n > 1 {
                                full.push(c);
                            }
                            for (e, &ok) in env.core.factor_mask(&full).iter().enumerate() {
                                rebuilt[config.encode(i, o, c, e)] = ok;
                            }
                        }
                    }
                }
                assert_eq!(rebuilt, flat, "chain != flat mask (C={}, O={})", c_n, o_n);
                let a = sample_true_index(&flat, &mut rng).unwrap();
                let (_, _, d, m) = env.step(a as i32);
                done = d;
                flat = m;
            }
        }
    }
}
