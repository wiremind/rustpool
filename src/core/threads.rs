//! Worker-thread count for a pool.
//!
//! By default every pool spawns `std::thread::available_parallelism()` worker threads (capped
//! at `num_envs`). On Linux that is the whole cgroup CPU quota / affinity mask of the process,
//! so a process that builds several pools (e.g. one per GPU) oversubscribes the machine with
//! `n_pools x n_cpus` threads. The count can be overridden per pool:
//!
//! 1. an explicit `num_threads` (the Python keyword, or [`crate::RlPool::new_with_factory_and_threads`]),
//! 2. else the `RUSTPOOL_NUM_THREADS` environment variable, read at pool construction,
//! 3. else `available_parallelism()` (the historical behaviour; 4 if it cannot be queried).
//!
//! The result is always capped at `num_envs`: a worker owns a disjoint shard of envs
//! (`env_id % num_threads`), so threads beyond `num_envs` would own nothing and only idle.

use std::env::VarError;

/// Environment variable read at pool construction to set the worker-thread count.
pub const NUM_THREADS_ENV_VAR: &str = "RUSTPOOL_NUM_THREADS";

/// Fallback when `available_parallelism()` cannot be queried (unchanged historical value).
const FALLBACK_PARALLELISM: usize = 4;

/// Parse a thread count from the env-var value. Empty / whitespace-only counts as unset.
fn parse_env_value(raw: &str) -> Result<Option<usize>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match trimmed.parse::<usize>() {
        Ok(n) if n > 0 => Ok(Some(n)),
        _ => Err(format!(
            "{} must be a positive integer, got {:?}",
            NUM_THREADS_ENV_VAR, raw
        )),
    }
}

/// Pure resolution: `explicit` > `env_value` > `default_parallelism`, capped at `num_envs`.
///
/// `env_value` is the raw value of `RUSTPOOL_NUM_THREADS` (`None` = unset). An explicit
/// `Some(0)` or an unparsable / non-positive env value is an error, never silently ignored.
pub fn resolve_num_threads(
    num_envs: usize,
    explicit: Option<usize>,
    env_value: Option<&str>,
    default_parallelism: usize,
) -> Result<usize, String> {
    let requested = match explicit {
        Some(0) => return Err("num_threads must be a positive integer, got 0".to_string()),
        Some(n) => n,
        None => match env_value.map(parse_env_value).transpose()?.flatten() {
            Some(n) => n,
            None => default_parallelism,
        },
    };
    Ok(requested.min(num_envs))
}

/// Resolve the worker-thread count for a pool of `num_envs` envs, reading
/// `RUSTPOOL_NUM_THREADS` from the process environment and falling back to
/// `available_parallelism()`.
pub fn pool_num_threads(num_envs: usize, explicit: Option<usize>) -> Result<usize, String> {
    let env_value = match std::env::var(NUM_THREADS_ENV_VAR) {
        Ok(v) => Some(v),
        Err(VarError::NotPresent) => None,
        Err(VarError::NotUnicode(v)) => {
            return Err(format!(
                "{} must be a positive integer, got {:?}",
                NUM_THREADS_ENV_VAR, v
            ))
        }
    };
    let default_parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(FALLBACK_PARALLELISM);
    resolve_num_threads(
        num_envs,
        explicit,
        env_value.as_deref(),
        default_parallelism,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_uses_default_parallelism_capped_at_num_envs() {
        assert_eq!(resolve_num_threads(64, None, None, 16), Ok(16));
        assert_eq!(resolve_num_threads(8, None, None, 16), Ok(8));
        // Unchanged edge case: an empty pool gets no workers.
        assert_eq!(resolve_num_threads(0, None, None, 16), Ok(0));
    }

    #[test]
    fn env_value_overrides_default() {
        assert_eq!(resolve_num_threads(64, None, Some("4"), 16), Ok(4));
        assert_eq!(resolve_num_threads(64, None, Some(" 30 "), 16), Ok(30));
        // Still capped at num_envs.
        assert_eq!(resolve_num_threads(8, None, Some("30"), 16), Ok(8));
    }

    #[test]
    fn empty_env_value_counts_as_unset() {
        assert_eq!(resolve_num_threads(64, None, Some(""), 16), Ok(16));
        assert_eq!(resolve_num_threads(64, None, Some("  "), 16), Ok(16));
    }

    #[test]
    fn invalid_env_value_is_an_error() {
        for bad in ["0", "-2", "abc", "2.5", "auto"] {
            let err = resolve_num_threads(64, None, Some(bad), 16).unwrap_err();
            assert!(err.contains(NUM_THREADS_ENV_VAR), "{}", err);
            assert!(err.contains("positive integer"), "{}", err);
        }
    }

    #[test]
    fn explicit_wins_over_env_value() {
        assert_eq!(resolve_num_threads(64, Some(2), Some("30"), 16), Ok(2));
        // An explicit value bypasses even an invalid env var.
        assert_eq!(resolve_num_threads(64, Some(2), Some("abc"), 16), Ok(2));
        assert_eq!(resolve_num_threads(4, Some(32), None, 16), Ok(4));
    }

    #[test]
    fn explicit_zero_is_an_error() {
        assert!(resolve_num_threads(64, Some(0), None, 16).is_err());
    }
}
