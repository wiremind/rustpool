"""Worker-thread count: ``num_threads=`` keyword and ``RUSTPOOL_NUM_THREADS`` env var."""

import os

import numpy as np
import pytest

import rustpool

ENV_VAR = "RUSTPOOL_NUM_THREADS"


def _make(**kwargs):
    return rustpool.make("Maze-v0", num_envs=8, batch_size=8, seed=0, **kwargs)


def _roundtrip(pool):
    pool.async_reset()
    _, _, _, info = pool.recv()
    pool.send(np.zeros(len(info["env_id"]), dtype=np.int32), info["env_id"])
    _, _, _, info = pool.recv()
    assert sorted(info["env_id"].tolist()) == list(range(8))
    pool.close()


def _cgroup_cpu_quota() -> bool:
    """True if a cgroup v2 CPU quota may limit this process (then the exact count is unknown)."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            return f.read().split()[0] != "max"
    except OSError:
        return False


def test_default_is_available_cpus_capped_at_num_envs(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    pool = _make()
    if hasattr(os, "sched_getaffinity") and not _cgroup_cpu_quota():
        # Without a quota, available_parallelism() is the affinity mask (conftest pins it).
        assert pool.num_threads == min(8, len(os.sched_getaffinity(0)))
    else:
        assert 1 <= pool.num_threads <= min(8, os.cpu_count() or 8)
    _roundtrip(pool)


def test_env_var_sets_worker_count(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "3")
    pool = _make()
    assert pool.num_threads == 3
    _roundtrip(pool)


def test_env_var_is_read_per_construction(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "2")
    first = _make()
    monkeypatch.setenv(ENV_VAR, "5")
    second = _make()
    assert (first.num_threads, second.num_threads) == (2, 5)
    first.close()
    second.close()


def test_env_var_capped_at_num_envs(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "64")
    pool = _make()
    assert pool.num_threads == 8
    pool.close()


def test_empty_env_var_counts_as_unset(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    default = _make()
    monkeypatch.setenv(ENV_VAR, "")
    empty = _make()
    assert empty.num_threads == default.num_threads
    default.close()
    empty.close()


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5"])
def test_invalid_env_var_raises(monkeypatch, bad):
    monkeypatch.setenv(ENV_VAR, bad)
    with pytest.raises(ValueError, match=ENV_VAR):
        _make()


def test_keyword_wins_over_env_var(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "5")
    pool = _make(num_threads=2)
    assert pool.num_threads == 2
    _roundtrip(pool)


def test_keyword_zero_raises(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="num_threads"):
        _make(num_threads=0)


@pytest.mark.parametrize("bad", [-1, 1.5, "2"])
def test_keyword_must_be_an_int(monkeypatch, bad):
    # Option<usize> extraction: a negative int overflows, a float / str is a type error.
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises((OverflowError, TypeError)):
        _make(num_threads=bad)


def test_keyword_is_keyword_only():
    # Positional args stop at task_id; a 5th positional must not land in num_threads.
    with pytest.raises(TypeError):
        rustpool.RlPool(8, 8, 0, "Maze-v0", 2)
