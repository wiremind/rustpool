"""Calls made before the first async_reset must not kill a worker (BinPack used to panic)."""

import numpy as np
import pytest

import rustpool

KWARGS = {"BinPack-v0": {"factored": True, "emit_flat_mask": True, "max_items": 4, "max_ems": 8}}


@pytest.mark.parametrize("task", rustpool.list_envs())
def test_step_snapshot_and_queries_before_the_first_reset(task):
    with rustpool.make(task, num_envs=2, batch_size=2, **KWARGS.get(task, {})) as pool:
        ids = np.arange(2, dtype=np.int32)
        roots = np.asarray(pool.snapshot(ids), np.int32)
        pool.simulate_batch(roots, np.zeros(2, np.int32))
        pool.query_state(0, "state")
        pool.send(np.zeros(2, np.int32), ids)
        pool.recv()
        # The pool still works normally afterwards.
        pool.async_reset()
        _, _, _, info = pool.recv()
        assert sorted(info["env_id"].tolist()) == [0, 1]


def test_binpack_before_the_first_reset_is_an_empty_instance():
    with rustpool.make("BinPack-v0", num_envs=1, batch_size=1, **KWARGS["BinPack-v0"]) as pool:
        assert pool.query_state(0, "bench:flat:1") is None  # nothing to measure yet
        pool.send(np.zeros(1, np.int32), np.zeros(1, np.int32))
        _, reward, done, _ = pool.recv()
        assert done[0] and reward[0] == 0.0  # no item to place: the episode ends at once
