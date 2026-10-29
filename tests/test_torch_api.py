"""Direct unit tests for rustpool.torch_api.TorchEnvPool (zero-copy + MCTS substrate)."""

import numpy as np
import pytest

import rustpool

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from rustpool.torch_api import TorchEnvPool, to_torch  # noqa: E402


def test_reset_step_dtypes_and_int_cast():
    pool = TorchEnvPool("Maze-v0", num_envs=4, device="cpu", seed=0)
    try:
        obs, reward, done, info = pool.reset()
        assert obs["agent_pos"].shape == (4, 2)
        assert reward.dtype == torch.float32
        assert done.dtype == torch.bool
        assert info["action_mask"].dtype == torch.bool
        # int64 actions must be accepted (cast to int32 at the boundary, not a TypeError).
        obs, reward, done, info = pool.step(torch.zeros(4, dtype=torch.int64))
        assert reward.shape == (4,)
    finally:
        pool.close()


def test_to_torch_is_zero_copy():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = to_torch(arr)
    tensor[0, 0] = 99.0
    assert arr[0, 0] == 99.0  # shares the buffer


def test_mcts_substrate_roundtrip_and_determinism():
    pool = TorchEnvPool("Maze-v0", num_envs=4, device="cpu", seed=0)
    try:
        pool.reset()
        state_ids = pool.snapshot()
        assert state_ids.dtype == torch.int32 and state_ids.shape == (4,)

        obs1, _, _, info1 = pool.simulate_batch(state_ids, torch.zeros(4, dtype=torch.int64))
        assert info1["state_id"].dtype == torch.int32
        # Re-simulating the same parents + action is deterministic (clone includes RNG state).
        obs1b, _, _, _ = pool.simulate_batch(state_ids, torch.zeros(4, dtype=torch.int64))
        assert torch.equal(obs1["agent_pos"], obs1b["agent_pos"])
        # Children are fresh handles, distinct from the parents.
        assert not torch.equal(info1["state_id"], state_ids)

        pool.release_batch(state_ids)
        pool.release_batch(info1["state_id"])
    finally:
        pool.close()


@pytest.mark.torch
def test_torch_seeds_snapshot_and_factored_queries_match_the_pool():
    torch = pytest.importorskip("torch")
    from rustpool.torch_api import TorchEnvPool

    kwargs = {"factored": True, "max_items": 6, "max_ems": 8}
    pool = TorchEnvPool("BinPack-v0", num_envs=3, device="cpu", **kwargs)
    ref = rustpool.make("BinPack-v0", num_envs=3, batch_size=3, **kwargs)
    try:
        seeds = torch.tensor([4, 5, 6])
        obs, _, _, _ = pool.reset(seeds=seeds)
        again, _, _, _ = pool.reset(seeds=seeds)
        assert torch.equal(obs["items"], again["items"])
        assert obs["items"].device.type == "cpu"
        ids = pool.snapshot(torch.tensor([0, 2]))
        assert ids.dtype == torch.int32 and ids.shape == (2,)
        assert pool.action_factorization() == ref.action_factorization()
        ref.async_reset(seeds.numpy().astype(np.int64))
        ref.recv()
        prefixes = np.full((3, 1), -1, np.int32)
        np.testing.assert_array_equal(
            pool.query_mask(torch.from_numpy(prefixes)).numpy(),
            ref.query_mask(prefixes, np.arange(3, dtype=np.int32)))
    finally:
        pool.close()
        ref.close()
