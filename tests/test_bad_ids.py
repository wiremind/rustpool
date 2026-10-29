"""Bad ids: env ids are positions (IndexError), state ids are registry keys (ValueError)."""

import numpy as np
import pytest

import rustpool

N = 4


def _calls(pool, bad):
    ids = np.array([0, bad], np.int32)
    return {
        "send": lambda: pool.send(np.zeros(2, np.int32), ids),
        "snapshot": lambda: pool.snapshot(ids),
        "query_state": lambda: pool.query_state(bad, "steps"),
        "query_mask": lambda: pool.query_mask(np.full((2, 1), -1, np.int32), ids),
    }


@pytest.mark.parametrize("method", ["send", "snapshot", "query_state", "query_mask"])
@pytest.mark.parametrize("bad", [-1, N])
def test_bad_env_id_raises_index_error_and_leaves_the_pool_usable(method, bad):
    with rustpool.make("BinPack-v0", num_envs=N, batch_size=N, factored=True) as pool:
        pool.async_reset()
        pool.recv()
        with pytest.raises(IndexError, match=rf"env_id {bad} is out of bounds for num_envs {N}"):
            _calls(pool, bad)[method]()
        # Nothing was dispatched: a full round still returns every env exactly once.
        pool.send(np.zeros(N, np.int32), np.arange(N, dtype=np.int32))
        *_, info = pool.recv()
        assert sorted(info["env_id"].tolist()) == list(range(N))


def test_unknown_or_released_state_id_raises_value_error():
    with rustpool.make("Maze-v0", num_envs=2, batch_size=2) as pool:
        pool.async_reset()
        pool.recv()
        sid = pool.snapshot(np.array([0], np.int32))[0]
        pool.release_batch(np.array([sid], np.int32))
        for state_id in (sid, 10**6):
            with pytest.raises(ValueError, match=rf"invalid state_id {state_id} \(unknown or released\)"):
                pool.simulate_batch(np.array([state_id], np.int32), np.zeros(1, np.int32))
