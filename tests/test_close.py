"""After close(): every call that reaches the envs or the stored states raises RuntimeError."""

import numpy as np
import pytest

import rustpool

N = 2


def _closed_pool():
    pool = rustpool.make("BinPack-v0", num_envs=N, batch_size=N, factored=True)
    pool.async_reset()
    pool.recv()
    state_ids = np.asarray(pool.snapshot(np.arange(N, dtype=np.int32)), np.int32)
    pool.close()
    return pool, state_ids


CALLS = {
    "send": lambda p, s: p.send(np.zeros(N, np.int32), np.arange(N, dtype=np.int32)),
    "recv": lambda p, s: p.recv(),
    "async_reset": lambda p, s: p.async_reset(),
    "snapshot": lambda p, s: p.snapshot(np.arange(N, dtype=np.int32)),
    "query_state": lambda p, s: p.query_state(0, "steps"),
    "query_mask": lambda p, s: p.query_mask(np.full((N, 1), -1, np.int32), np.arange(N, dtype=np.int32)),
    "simulate_batch": lambda p, s: p.simulate_batch(s, np.zeros(N, np.int32)),
    "query_state_mask": lambda p, s: p.query_state_mask(s, np.full((N, 1), -1, np.int32)),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_calls_after_close_raise_runtime_error(name):
    pool, state_ids = _closed_pool()
    with pytest.raises(RuntimeError, match="closed"):
        CALLS[name](pool, state_ids)


def test_close_is_idempotent_and_keeps_the_pool_describable():
    pool, state_ids = _closed_pool()
    pool.close()
    pool.release_batch(state_ids)  # a no-op: close() already freed the states
    assert len(pool) == pool.num_envs == N
    assert set(pool.observation_spec()) >= {"items", "ems"}
    assert pool.action_factorization()["pad_width"] > 0


def test_with_block_closes_the_pool_even_when_the_body_raises():
    with pytest.raises(KeyError), rustpool.make("Maze-v0", num_envs=N, batch_size=N) as pool:
        raise KeyError("boom")
    with pytest.raises(RuntimeError, match="closed"):
        pool.async_reset()
