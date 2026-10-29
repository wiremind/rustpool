"""Small contracts: empty batches, the compiled module's docstrings, lazy imports, large-batch
observation stacking, and dispatch identity under both auto-reset modes with async batches."""

import subprocess
import sys

import numpy as np
import pytest

import rustpool
from rustpool import _rustpool


def test_empty_batches():
    with rustpool.make("BinPack-v0", num_envs=2, batch_size=2, factored=True) as pool:
        pool.async_reset()
        pool.recv()
        empty = np.zeros(0, np.int32)
        assert pool.snapshot(empty) == []
        obs, reward, done, info = pool.simulate_batch(empty, empty)
        assert reward.shape == done.shape == info["state_id"].shape == (0,)
        assert obs["items"].shape == (0, *pool.observation_spec()["items"][0])
        pool.release_batch(empty)
        pool.send(empty, empty)  # a no-op: the next full round is unaffected
        pool.send(np.zeros(2, np.int32), np.arange(2, dtype=np.int32))
        assert sorted(pool.recv()[3]["env_id"].tolist()) == [0, 1]


def test_module_and_test_hook_docstrings():
    assert _rustpool.__doc__.startswith("Compiled core of rustpool")
    assert _rustpool._pool_from_factory.__doc__.startswith("Test hook")


def test_import_and_a_pool_pull_no_optional_stack():
    code = (
        "import sys, rustpool; rustpool.make('Maze-v0', num_envs=2).close(); "
        "heavy = {'jax', 'torch', 'gymnasium', 'stoa', 'chex', 'flax'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_large_batch_stacking_matches_small_batches():
    """512 BinPack envs stack above the parallel threshold (2^18 elements); every env's
    observation equals the same env's in a pool returning batches of 8."""
    big = rustpool.make("BinPack-v0", num_envs=512, batch_size=512, seed=5)
    small = rustpool.make("BinPack-v0", num_envs=512, batch_size=8, seed=5)
    try:
        big.async_reset()
        obs_big, _, _, info_big = big.recv()
        assert sum(np.prod(v.shape) for v in obs_big.values()) > 1 << 18
        small.async_reset()
        per_env = {}
        for _ in range(512 // 8):
            obs, _, _, info = small.recv()
            for row, e in enumerate(info["env_id"]):
                per_env[int(e)] = {k: v[row] for k, v in obs.items()}
        for row, e in enumerate(info_big["env_id"]):
            for k, v in obs_big.items():
                np.testing.assert_array_equal(v[row], per_env[int(e)][k])
    finally:
        big.close()
        small.close()


@pytest.mark.parametrize("autoreset", ["same_step", "next_step"])
def test_dispatch_modes_agree_with_async_batches(autoreset):
    """per_env and per_worker give every env the same trajectory, with 16 envs in batches of 4."""
    def run(dispatch):
        pool = rustpool.make("CartPole-v1", num_envs=16, batch_size=4, num_threads=4, seed=2,
                             dispatch=dispatch, autoreset=autoreset)
        per_env = {e: [] for e in range(16)}
        pool.async_reset()
        for step in range(30):
            for _ in range(4):
                obs, reward, done, info = pool.recv()
                for row, e in enumerate(info["env_id"]):
                    per_env[int(e)].append((obs["obs"][row].tolist(), float(reward[row]), bool(done[row])))
            pool.send(np.array([(e + step // 5) % 2 for e in range(16)], np.int32),
                      np.arange(16, dtype=np.int32))
        pool.close()
        return per_env

    assert run("per_env") == run("per_worker")
