//! Bench-only queries on a frozen state: `query_state(env_id, "bench:<kind>[:reps]")`.
//!
//! - `bench:flat`: the flat action mask (`get_action_mask`) -- what a flat env builds per step.
//! - `bench:chain`: the factored chain for one committed action -- stage 0, then each later
//!   stage given the first valid choice so far (what a factored-only env computes per step).
//! - `bench:place`: placing that same action -- `place_item` (incl. the EMS update) and the
//!   re-sort of the touched container, i.e. the env's own work besides the mask.
//!
//! Each returns `{"ns", "ns_min", "ns_all", "calls", "reps", ...}`: wall time per repetition
//! (on a clone of the state, sorted-EMS cache warm, so the state is never advanced) and the
//! number of `can_fit` tests. Calls are counted by counting twins of the mask functions that
//! run only here, so the production path carries no counter; every reply says whether the
//! twins reproduced the production masks (`"consistent"`).
//!
//! `reps` is 1..=`MAX_REPS` (default `DEFAULT_REPS`); any other value, an unknown kind, an
//! extra `:` segment, or a query before the env's first reset answers `None`. An internal benchmarking hook, outside semver: the keys
//! and the reply's fields may change in any release.

use super::*;
use serde_json::{json, Value};
use std::hint::black_box;
use std::time::Instant;

const DEFAULT_REPS: usize = 5;
/// Upper bound on `reps`: each repetition re-runs the measured work on the calling worker.
const MAX_REPS: usize = 1000;

fn timing(ns: &[u64]) -> Value {
    let mean = ns.iter().sum::<u64>() as f64 / ns.len().max(1) as f64;
    json!({"ns": mean, "ns_min": ns.iter().min(), "ns_all": ns, "reps": ns.len()})
}

fn merge(mut a: Value, b: Value) -> Value {
    if let (Some(a), Value::Object(b)) = (a.as_object_mut(), b) {
        a.extend(b);
    }
    a
}

impl BinPackCore {
    /// `can_fit` tests of the flat mask: every (unplaced item, orientation, container, EMS).
    fn count_flat_calls(&self) -> u64 {
        let ems: u64 = (0..self.config.num_containers)
            .map(|c| self.with_sorted(c, |s| s.len() as u64))
            .sum();
        let unplaced = self.placed_mask.iter().filter(|&&p| !p).count() as u64;
        unplaced * self.config.orientations as u64 * ems
    }

    fn count_fits_in(&self, item: usize, o: usize, c: usize) -> (bool, u64) {
        self.with_sorted(c, |ems| {
            let mut n = 0;
            for s in ems {
                n += 1;
                if self.fits(item, o, s) {
                    return (true, n);
                }
            }
            (false, n)
        })
    }

    fn count_orientation_fits(&self, item: usize, o: usize) -> (bool, u64) {
        let mut n = 0;
        for c in 0..self.config.num_containers {
            let (ok, k) = self.count_fits_in(item, o, c);
            n += k;
            if ok {
                return (true, n);
            }
        }
        (false, n)
    }

    fn count_item_fits_anywhere(&self, item: usize) -> (bool, u64) {
        let mut n = 0;
        for o in 0..self.config.orientations {
            let (ok, k) = self.count_orientation_fits(item, o);
            n += k;
            if ok {
                return (true, n);
            }
        }
        (false, n)
    }

    /// Counting twin of `factor_mask`: the same mask plus its `can_fit` tests.
    fn count_factor_mask(&self, chosen: &[usize]) -> (Vec<bool>, u64) {
        let factors = self.factors();
        let Some(&next) = factors.get(chosen.len()) else {
            return (Vec::new(), 0);
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
            return (vec![false; width], 0);
        }
        let mut n = 0;
        let mask = match next {
            Factor::Item => (0..width)
                .map(|i| {
                    if !self.placeable(i) {
                        return false;
                    }
                    let (ok, k) = self.count_item_fits_anywhere(i);
                    n += k;
                    ok
                })
                .collect(),
            Factor::Orientation => (0..width)
                .map(|oo| {
                    let (ok, k) = self.count_orientation_fits(item, oo);
                    n += k;
                    ok
                })
                .collect(),
            Factor::Container => (0..width)
                .map(|cc| {
                    if o >= self.config.orientations {
                        return false;
                    }
                    let (ok, k) = self.count_fits_in(item, o, cc);
                    n += k;
                    ok
                })
                .collect(),
            Factor::Ems => {
                let mut mask = vec![false; width];
                if o < self.config.orientations && c < self.config.num_containers {
                    self.with_sorted(c, |ems| {
                        for (j, s) in ems.iter().enumerate() {
                            n += 1;
                            mask[j] = self.fits(item, o, s);
                        }
                    });
                }
                mask
            }
        };
        (mask, n)
    }

    /// The first valid choice at every stage (empty if no item can be placed).
    fn first_valid_path(&self) -> Vec<usize> {
        let mut path = Vec::new();
        for _ in 0..self.factors().len() {
            match self.factor_mask(&path).iter().position(|&v| v) {
                Some(v) => path.push(v),
                None => return Vec::new(),
            }
        }
        path
    }

    /// `(item, orientation, container, ems)` of a full chain path.
    fn path_action(&self, path: &[usize]) -> (usize, usize, usize, usize) {
        let (mut item, mut o, mut c, mut e) = (0, 0, 0, 0);
        for (f, &v) in self.factors().iter().zip(path) {
            match f {
                Factor::Item => item = v,
                Factor::Orientation => o = v,
                Factor::Container => c = v,
                Factor::Ems => e = v,
            }
        }
        (item, o, c, e)
    }

    pub(super) fn bench_query(&self, key: &str) -> Option<String> {
        let mut parts = key.split(':');
        if parts.next() != Some("bench") {
            return None;
        }
        let kind = parts.next()?;
        let reps = match parts.next() {
            Some(r) => r
                .parse::<usize>()
                .ok()
                .filter(|r| (1..=MAX_REPS).contains(r))?,
            None => DEFAULT_REPS,
        };
        if parts.next().is_some() {
            return None;
        }
        if self.items.is_empty() {
            return None; // no instance yet (before the first reset): nothing to measure
        }
        let mut core = self.clone();
        core.ensure_sorted();
        let value = match kind {
            "flat" => {
                let mask = core.get_action_mask();
                let ns: Vec<u64> = (0..reps)
                    .map(|_| {
                        let t = Instant::now();
                        black_box(core.get_action_mask());
                        t.elapsed().as_nanos() as u64
                    })
                    .collect();
                merge(
                    timing(&ns),
                    json!({"calls": core.count_flat_calls(), "mask_len": mask.len(),
                           "valid": mask.iter().filter(|&&v| v).count()}),
                )
            }
            "chain" => {
                let path = core.first_valid_path();
                // Stage 0 always; the later stages only along a valid path.
                let stages = if path.is_empty() { 1 } else { path.len() };
                let mut stage_calls = Vec::with_capacity(stages);
                let mut consistent = true;
                for k in 0..stages {
                    let (mask, calls) = core.count_factor_mask(&path[..k]);
                    consistent &= mask == core.factor_mask(&path[..k]);
                    stage_calls.push(calls);
                }
                let mut stage_ns = vec![0u64; stages];
                let ns: Vec<u64> = (0..reps)
                    .map(|_| {
                        let t0 = Instant::now();
                        for (k, s) in stage_ns.iter_mut().enumerate() {
                            let t = Instant::now();
                            black_box(core.factor_mask(&path[..k]));
                            *s += t.elapsed().as_nanos() as u64;
                        }
                        t0.elapsed().as_nanos() as u64
                    })
                    .collect();
                let names: Vec<&str> = core
                    .factors()
                    .iter()
                    .map(|f| match f {
                        Factor::Item => "item",
                        Factor::Orientation => "orientation",
                        Factor::Container => "container",
                        Factor::Ems => "ems",
                    })
                    .collect();
                let stage_ns: Vec<f64> = stage_ns.iter().map(|&s| s as f64 / reps as f64).collect();
                merge(
                    timing(&ns),
                    json!({"calls": stage_calls.iter().sum::<u64>(), "stage_calls": stage_calls,
                           "stage_ns": stage_ns, "path": path, "factors": names,
                           "consistent": consistent}),
                )
            }
            "place" => {
                let path = core.first_valid_path();
                if path.is_empty() {
                    json!({"ns": null, "sort_ns": null, "action": null, "reps": 0})
                } else {
                    let (item, o, c, e) = core.path_action(&path);
                    let (mut place_ns, mut sort_ns) = (Vec::new(), Vec::new());
                    let mut placed = true;
                    for _ in 0..reps {
                        let mut s = core.clone();
                        let t = Instant::now();
                        placed &= black_box(s.place_item(item, o, c, e)).0;
                        place_ns.push(t.elapsed().as_nanos() as u64);
                        let t = Instant::now();
                        s.ensure_sorted();
                        sort_ns.push(t.elapsed().as_nanos() as u64);
                    }
                    let sort = timing(&sort_ns);
                    merge(
                        timing(&place_ns),
                        json!({"sort_ns": sort["ns"], "sort_ns_min": sort["ns_min"],
                               "action": core.config.encode(item, o, c, e), "placed": placed}),
                    )
                }
            }
            _ => return None,
        };
        Some(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(c: usize, o: usize) -> BinPackConfig {
        BinPackConfig {
            max_items: 10,
            max_ems: 12,
            num_containers: c,
            orientations: o,
            factored: true,
            ..BinPackConfig::default()
        }
    }

    /// The counting twins reproduce the production masks along random episodes, and the
    /// flat count is the brute-force product.
    #[test]
    fn test_counting_twins_match() {
        for (c, o) in [(1, 1), (3, 1), (2, 2)] {
            for seed in 0..5 {
                let mut core = BinPackCore::new(cfg(c, o), seed);
                core.reset(None);
                let mut rng = StdRng::seed_from_u64(seed);
                loop {
                    for key in ["bench:flat:2", "bench:chain:2", "bench:place:2"] {
                        let v: Value =
                            serde_json::from_str(&core.bench_query(key).unwrap()).unwrap();
                        assert_ne!(v["consistent"], json!(false), "{key} c={c} o={o}");
                    }
                    let flat = core.get_action_mask();
                    let calls = core.count_flat_calls();
                    let brute: u64 = (0..core.items.len())
                        .filter(|&i| !core.placed_mask[i])
                        .map(|_| {
                            (0..c)
                                .map(|cc| core.sorted_ems(cc).len() as u64)
                                .sum::<u64>()
                                * o as u64
                        })
                        .sum();
                    assert_eq!(calls, brute);
                    let Some(a) = sample_true_index(&flat, &mut rng) else {
                        break;
                    };
                    let (i, oo, cc, e) = core.config.decode(a);
                    core.step(i, oo, cc, e);
                }
            }
        }
    }

    #[test]
    fn test_bench_keys() {
        let mut core = BinPackCore::new(cfg(2, 1), 1);
        core.reset(None);
        assert!(core.bench_query("bench:nope").is_none());
        for key in [
            "bench:flat:0",
            "bench:flat:1001",
            "bench:flat:2:x",
            "bench:flat:-1",
        ] {
            assert!(core.bench_query(key).is_none(), "{key}");
        }
        assert!(core
            .bench_query(&format!("bench:flat:{MAX_REPS}"))
            .is_some());
        assert!(core.bench_query("bench:flat:0").is_none());
        let v: Value = serde_json::from_str(&core.bench_query("bench:chain:3").unwrap()).unwrap();
        assert_eq!(v["reps"], json!(3));
        assert_eq!(v["factors"], json!(["item", "container", "ems"]));
        assert_eq!(v["stage_calls"].as_array().unwrap().len(), 3);
        let p: Value = serde_json::from_str(&core.bench_query("bench:place").unwrap()).unwrap();
        assert_eq!(p["placed"], json!(true));
        assert_eq!(p["reps"], json!(DEFAULT_REPS));
    }
}
