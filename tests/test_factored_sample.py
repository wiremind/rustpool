"""The examples' factored samplers take factor 0 from ``initial_mask`` (K-1 queries), and
``flatten_obs`` never feeds BinPack's ground-truth solution to a policy."""

import pathlib
import sys

import numpy as np
import pytest

import rustpool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))

KW = dict(max_items=6, max_ems=8, factored=True, num_containers=2, seed=3)
B = 5


def _counting_query(pool, ids, as_array):
    calls = []

    def query(prefix):
        calls.append(np.asarray(prefix).copy())
        return as_array(pool.query_mask(np.asarray(prefix, np.int32), ids))

    return query, calls


@pytest.mark.jax
def test_jax_factored_sample_initial_mask():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp
    from networks import FactoredActorCritic, factored_sample, flatten_obs

    pool = rustpool.make("BinPack-v0", num_envs=B, batch_size=B, **KW)
    try:
        spec = pool.action_factorization()
        widths, K = spec["factor_widths"], spec["num_factors"]
        assert K == 3  # item -> container -> ems
        ids = np.arange(B, dtype=np.int32)
        pool.async_reset()
        obs, _, _, info = pool.recv()
        obs = {k: jnp.asarray(v) for k, v in obs.items()}
        assert "ground_truth_locations" in obs
        net = FactoredActorCritic(widths)
        params = net.init(jax.random.key(0), obs)
        key = jax.random.key(1)

        query, calls = _counting_query(pool, ids, jnp.asarray)
        ref = factored_sample(net, params, obs, query, widths, key)
        assert len(calls) == K
        query, calls = _counting_query(pool, ids, jnp.asarray)
        got = factored_sample(net, params, obs, query, widths, key,
                              initial_mask=jnp.asarray(info["action_mask"]))
        assert len(calls) == K - 1
        for a, b in zip(ref, got, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        # ground_truth_locations is dropped; every other key is kept.
        full = sum(int(np.prod(v.shape[1:])) for v in obs.values())
        gt = int(np.prod(obs["ground_truth_locations"].shape[1:]))
        assert flatten_obs(obs).shape == (B, full - gt)
    finally:
        pool.close()


@pytest.mark.torch
def test_torch_factored_sample_initial_mask():
    torch = pytest.importorskip("torch")
    from torch_common import FactoredActorCritic, factored_sample, flatten_obs

    pool = rustpool.make("BinPack-v0", num_envs=B, batch_size=B, **KW)
    try:
        spec = pool.action_factorization()
        widths, K = spec["factor_widths"], spec["num_factors"]
        ids = np.arange(B, dtype=np.int32)
        pool.async_reset()
        obs, _, _, info = pool.recv()
        obs = {k: torch.as_tensor(v) for k, v in obs.items()}
        flat_obs = flatten_obs(obs)
        full = sum(int(np.prod(v.shape[1:])) for v in obs.values())
        assert flat_obs.shape == (B, full - int(np.prod(obs["ground_truth_locations"].shape[1:])))
        net = FactoredActorCritic(flat_obs.shape[-1], widths)

        query, calls = _counting_query(pool, ids, torch.as_tensor)
        torch.manual_seed(0)
        ref = factored_sample(net, flat_obs, query, widths)
        assert len(calls) == K
        query, calls = _counting_query(pool, ids, torch.as_tensor)
        torch.manual_seed(0)
        got = factored_sample(net, flat_obs, query, widths,
                              initial_mask=torch.as_tensor(info["action_mask"]))
        assert len(calls) == K - 1
        for a, b in zip(ref, got, strict=True):
            torch.testing.assert_close(a, b)
    finally:
        pool.close()
