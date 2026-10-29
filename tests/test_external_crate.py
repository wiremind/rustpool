"""A real external env crate (``examples/rust/external_env.rs``, the docs' "Bring your own
crate" guide): built with Cargo as its own extension module, imported, and driven through
rustpool's pool API, the pool options and both bridges. Its pool class is not
``rustpool.RlPool``: the same Rust type registered by another module."""


import numpy as np
import pytest

import rustpool


@pytest.fixture(scope="module")
def ext(external_env):
    return external_env  # built and imported by the session fixture in conftest.py


def test_pool_is_a_foreign_class_with_the_full_api(ext):
    pool = ext.make(num_envs=6, batch_size=6, seed=3, num_threads=2)
    assert not isinstance(pool, rustpool.RlPool)
    assert (pool.num_envs, pool.num_threads, pool.dispatch, pool.autoreset) == (
        6, 2, "auto", "same_step")
    assert pool.observation_spec() == {"pos": ([1], "int32")}
    assert tuple(pool.action_spec()) == (2,) and pool.action_mask_width() == 2
    pool.async_reset()
    obs, _, _, info = pool.recv()
    assert obs["pos"].shape == (6, 1) and info["action_mask"].shape == (6, 2)
    np.testing.assert_array_equal(info["action_mask"][:, 0], obs["pos"][:, 0] > 0)
    returns = 0.0
    for _ in range(40):  # always move right: every episode reaches the end
        pool.send(np.ones(6, np.int32), info["env_id"])
        obs, rew, done, info = pool.recv()
        returns += float(rew.sum())
    assert returns > 0
    pool.close()


def test_seed_plus_index_and_options(ext):
    def first_pos(seed):
        pool = ext.make(num_envs=4, batch_size=4, seed=seed, num_threads=1)
        pool.async_reset()
        pos = pool.recv()[0]["pos"][:, 0].copy()
        pool.close()
        return pos

    np.testing.assert_array_equal(first_pos(10)[1:], first_pos(11)[:-1])
    pool = ext.make(num_envs=2, batch_size=2, dispatch="per_worker", autoreset="next_step")
    assert (pool.dispatch, pool.autoreset) == ("per_worker", "next_step")
    pool.close()
    with pytest.raises(ValueError, match="autoreset"):
        ext.make(autoreset="later")
    with pytest.raises(ValueError, match="length"):
        ext.make(length=1)


def test_next_step_autoreset_in_the_external_crate(ext):
    pool = ext.make(num_envs=1, batch_size=1, num_threads=1, autoreset="next_step", length=3)
    pool.async_reset()
    _, _, _, info = pool.recv()
    pending, resets = False, 0
    for _ in range(30):
        pool.send(np.ones(1, np.int32), info["env_id"])
        obs, rew, done, info = pool.recv()
        if pending:
            assert rew[0] == 0.0 and not done[0]
            resets += 1
        elif done[0]:
            assert obs["pos"][0, 0] == 2  # the terminal cell, not a reset one
        pending = bool(done[0])
    pool.close()
    assert resets > 0


def test_search_substrate_in_the_external_crate(ext):
    pool = ext.make(num_envs=3, batch_size=3, seed=1)
    pool.async_reset()
    obs, _, _, info = pool.recv()
    roots = np.asarray(pool.snapshot(info["env_id"]), np.int32)
    _, _, _, child = pool.simulate_batch(roots, np.ones(3, np.int32))
    kids = np.asarray(child["state_id"], np.int32)
    o2, _, _, _ = pool.simulate_batch(roots, np.ones(3, np.int32))  # parents unchanged
    np.testing.assert_array_equal(o2["pos"], np.minimum(obs["pos"] + 1, 7))
    pool.release_batch(kids)
    pool.release_batch(roots)
    pool.close()


@pytest.mark.jax
def test_jaxpool_wraps_the_external_pool(ext):
    jax = pytest.importorskip("jax")
    from rustpool.jax_api import JaxPool

    pool = JaxPool.from_pools(ext.make(num_envs=4, batch_size=4))
    out = jax.jit(pool.reset)(jax.random.key(0))
    out = jax.jit(pool.step)(jax.numpy.ones(4, jax.numpy.int32))
    assert out.obs["pos"].shape == (4, 1) and out.action_mask.shape == (4, 2)
    pool.close()


@pytest.mark.torch
def test_torchenvpool_wraps_the_external_pool(ext):
    torch = pytest.importorskip("torch")
    from rustpool.torch_api import TorchEnvPool

    pool = TorchEnvPool.from_pool(ext.make(num_envs=4, batch_size=4))
    obs, _, _, info = pool.reset()
    obs, _, _, info = pool.step(torch.ones(4, dtype=torch.int64))
    assert obs["pos"].shape == (4, 1) and info["action_mask"].dtype == torch.bool
    pool.close()
