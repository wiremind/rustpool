"""Pools built the way external env crates build them (``RlPool::new_with_options``) take the
same pool options as ``rustpool.make``.

``_rustpool._pool_from_factory`` builds a pool the way an external crate does: a factory
closure passed to ``new_with_options`` (here wrapping a built-in env, so no second extension
module has to be compiled). Every option must behave as in ``rustpool.make``.
"""

import numpy as np
import pytest

import rustpool
from rustpool import _rustpool

external_pool = _rustpool._pool_from_factory


def _rollout(pool, steps=300):
    """Synchronous random rollout; returns the stacked obs, rewards and dones."""
    rng = np.random.default_rng(0)
    pool.async_reset()
    obs, _, _, info = pool.recv()
    o, r, d = [obs["obs"]], [], []
    for _ in range(steps):
        pool.send(rng.integers(0, 2, size=pool.num_envs).astype(np.int32), info["env_id"])
        obs, rew, done, info = pool.recv()
        o.append(obs["obs"])
        r.append(rew)
        d.append(done)
    pool.close()
    return np.stack(o), np.stack(r), np.stack(d)


@pytest.mark.parametrize("dispatch", ["auto", "per_env", "per_worker"])
@pytest.mark.parametrize("autoreset", ["same_step", "next_step"])
def test_every_option_matches_route_a(dispatch, autoreset):
    opts = dict(seed=11, num_threads=3, dispatch=dispatch, autoreset=autoreset)
    b = external_pool("CartPole-v1", 8, 8, **opts)
    assert (b.num_threads, b.dispatch, b.autoreset) == (3, dispatch, autoreset)
    a = rustpool.make("CartPole-v1", num_envs=8, batch_size=8, **opts)
    for x, y in zip(_rollout(a), _rollout(b), strict=True):
        np.testing.assert_array_equal(x, y)


def test_factory_receives_seed_plus_index():
    """Env i is factory(seed + i) in both routes: a pool seeded s + 1 is the pool seeded s
    shifted by one env."""
    def first_obs(seed):
        pool = external_pool("CartPole-v1", 4, 4, seed=seed, num_threads=1)
        pool.async_reset()
        obs = pool.recv()[0]["obs"]
        pool.close()
        return obs

    o5, o6 = first_obs(5), first_obs(6)
    np.testing.assert_array_equal(o5[1:], o6[:-1])
    assert not np.array_equal(o5, o6)


def test_seed_reproducibility():
    x = _rollout(external_pool("CartPole-v1", 6, 6, seed=42, num_threads=2))
    y = _rollout(external_pool("CartPole-v1", 6, 6, seed=42, num_threads=2))
    z = _rollout(external_pool("CartPole-v1", 6, 6, seed=43, num_threads=2))
    for u, v in zip(x, y, strict=True):
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(x[0], z[0])


def test_default_options_match_make_with_seed_0():
    """The Rust-side defaults (PoolOptions::default: seed 0, so env i is factory(i), auto
    dispatch, same_step) give exactly make()'s seed-0 pool."""
    b = external_pool("CartPole-v1", 4, 4)
    assert (b.dispatch, b.autoreset) == ("auto", "same_step")
    a = rustpool.make("CartPole-v1", num_envs=4, batch_size=4, seed=0)
    for x, y in zip(_rollout(a, 50), _rollout(b, 50), strict=True):
        np.testing.assert_array_equal(x, y)


def test_async_batching_and_env_kwargs_pass_through():
    pool = external_pool("BinPack-v0", 6, 2, seed=1, num_threads=2, max_items=8, max_ems=16)
    pool.async_reset()
    obs, _, _, info = pool.recv()
    assert len(info["env_id"]) == 2
    assert pool.observation_spec() == rustpool.make(
        "BinPack-v0", num_envs=1, max_items=8, max_ems=16).observation_spec()
    pool.close()


@pytest.mark.parametrize("kw, match", [
    (dict(dispatch="per_core"), "dispatch"),
    (dict(autoreset="later"), "autoreset"),
    (dict(num_threads=0), "num_threads|RUSTPOOL_NUM_THREADS|thread"),
])
def test_bad_options_raise_like_route_a(kw, match):
    with pytest.raises(ValueError, match=match):
        external_pool("CartPole-v1", 4, 4, **kw)


def test_batch_size_above_num_envs_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        external_pool("CartPole-v1", 2, 4)
