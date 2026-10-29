"""The factored-action surface around query_mask: argument validation, the positional
(duplicate-env_id-safe) round-trip, factored BinPack's default stage-0 mask (pool, JaxPool,
TorchEnvPool), and the ValueError every bridge raises on an env with no action factorization.
The factorization specs and chain masks themselves are in test_binpack_multi.py.
"""

import numpy as np
import pytest

import rustpool


def test_query_mask_validates_arguments():
    pool = rustpool.make("Maze-v0", num_envs=4, batch_size=4)
    try:
        pool.async_reset()
        pool.recv()
        with pytest.raises(ValueError):  # rows != env_ids length
            pool.query_mask(np.array([[-1], [-1]], np.int32), np.array([0], np.int32))
        with pytest.raises(IndexError):  # env_id out of bounds (pre-validated)
            pool.query_mask(np.array([[-1]], np.int32), np.array([99], np.int32))
    finally:
        pool.close()


def test_query_mask_duplicate_env_ids_does_not_corrupt_the_buffer():
    # Duplicate env_ids are collected positionally; a subsequent recv must be unaffected.
    pool = rustpool.make("BinPack-v0", num_envs=4, batch_size=4, factored=True)
    try:
        pool.async_reset()
        pool.recv()
        pool.query_mask(np.array([[-1], [-1], [-1]], np.int32), np.array([0, 0, 1], np.int32))
        pool.send(np.zeros(4, np.int32), np.arange(4, dtype=np.int32))
        _, _, _, info = pool.recv()
        assert sorted(np.asarray(info["env_id"]).tolist()) == [0, 1, 2, 3]
    finally:
        pool.close()


# --- flat-mask default: factored envs skip it unless asked (and emit the stage-0 mask) -------

_BP = dict(max_items=10, max_ems=20, seed=4)


@pytest.mark.parametrize(
    "kwargs,width",
    [
        ({}, 200),                                            # flat env: mask on
        ({"factored": True}, 20),                             # factored: stage-0, pad_width
        ({"factored": True, "emit_flat_mask": True}, 200),    # explicit opt-in wins
        ({"emit_flat_mask": False}, 0),                       # explicit opt-out wins
    ],
    ids=["flat", "factored-default", "factored-optin", "flat-optout"],
)
def test_flat_mask_default_follows_factored(kwargs, width):
    pool = rustpool.make("BinPack-v0", num_envs=3, batch_size=3, **_BP, **kwargs)
    try:
        assert pool.action_mask_width() == width
        pool.async_reset()
        _, _, _, info = pool.recv()
        assert np.asarray(info["action_mask"]).shape == (3, width)
    finally:
        pool.close()


def test_factored_without_flat_mask_matches_with_it():
    """Skipping the flat mask changes nothing else: same obs, rewards and done flags (done is
    then decided by an existence scan instead of reducing the mask)."""
    B = 4
    on = rustpool.make("BinPack-v0", num_envs=B, batch_size=B, factored=True,
                       emit_flat_mask=True, **_BP)
    off = rustpool.make("BinPack-v0", num_envs=B, batch_size=B, factored=True, **_BP)
    ids = np.arange(B, dtype=np.int32)
    rng = np.random.default_rng(0)
    try:
        on.async_reset(), off.async_reset()
        (o1, _, d1, i1), (o2, _, d2, _) = on.recv(), off.recv()
        saw_done = False
        for _ in range(80):
            for k in o1:
                assert np.array_equal(np.asarray(o1[k]), np.asarray(o2[k])), k
            assert np.array_equal(np.asarray(d1), np.asarray(d2))
            mask = np.asarray(i1["action_mask"])
            a = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in mask], np.int32)
            on.send(a, ids), off.send(a, ids)
            (o1, r1, d1, i1), (o2, r2, d2, _) = on.recv(), off.recv()
            assert np.array_equal(np.asarray(r1), np.asarray(r2))
            saw_done |= bool(np.asarray(d1).any())
        assert saw_done, "episodes never ended -- the done path was not exercised"
    finally:
        on.close(), off.close()


def test_factored_action_mask_is_stage0_query_mask():
    """Without the flat mask a factored env's ``action_mask`` is the stage-0 (item) mask padded
    to ``pad_width`` -- exactly ``query_mask`` at the empty prefix, at every step and across
    auto-resets -- so a factored sampler needs only the K-1 later stages."""
    for extra in ({}, {"num_containers": 3}):
        B = 6
        pool = rustpool.make("BinPack-v0", num_envs=B, batch_size=B, factored=True, **_BP,
                             **extra)
        spec = pool.action_factorization()
        ids = np.arange(B, dtype=np.int32)
        rng = np.random.default_rng(1)
        empty = np.full((B, spec["num_factors"] - 1), -1, np.int32)
        try:
            assert pool.action_mask_width() == spec["pad_width"]
            pool.async_reset()
            _, _, done, info = pool.recv()
            saw_done = False
            for _ in range(60):
                mask = np.asarray(info["action_mask"])
                assert mask.shape == (B, spec["pad_width"])
                order = np.argsort(info["env_id"])
                np.testing.assert_array_equal(mask[order], pool.query_mask(empty, ids))
                # Random completion through the later stages.
                prefix = empty.copy()
                choices = []
                for k in range(spec["num_factors"]):
                    m = mask[order] if k == 0 else pool.query_mask(prefix, ids)
                    c = np.array([rng.choice(np.flatnonzero(r)) if r.any() else 0 for r in m])
                    choices.append(c)
                    if k < spec["num_factors"] - 1:
                        prefix[:, k] = c
                flat = np.zeros(B, np.int64)
                for c, w in zip(choices, spec["factor_widths"], strict=True):
                    flat = flat * w + c
                pool.send(flat.astype(np.int32), ids)
                _, _, done, info = pool.recv()
                saw_done |= bool(np.any(done))
            assert saw_done
        finally:
            pool.close()


@pytest.mark.jax
def test_jaxpool_factored_default_has_stage0_mask():
    jax = pytest.importorskip("jax")
    from rustpool.jax_api import JaxPool

    pool = JaxPool("BinPack-v0", 3, factored=True, max_items=10, max_ems=20)
    try:
        ids = np.arange(3, dtype=np.int32)
        out = pool.reset(jax.random.key(0))
        assert out.action_mask.shape == (3, 20)
        stage0 = pool.query_mask(np.full((3, 1), -1, np.int32), ids)
        np.testing.assert_array_equal(out.action_mask, stage0)
        out = pool.step(np.zeros(3, np.int32))
        assert out.action_mask.shape == (3, 20)
        mask = pool.query_mask(np.full((3, 1), -1, np.int32), ids)
        assert mask.shape == (3, 20)
        np.testing.assert_array_equal(out.action_mask, mask)
    finally:
        pool.close()


FLAT = [("Maze-v0", {}), ("CartPole-v1", {}), ("BinPack-v0", {"factored": False})]


def _ids(n):
    return np.arange(n, dtype=np.int32)


@pytest.mark.parametrize("task, kwargs", FLAT)
def test_query_mask_on_a_flat_env_raises(task, kwargs):
    with rustpool.make(task, num_envs=2, batch_size=2, **kwargs) as pool:
        pool.async_reset()
        pool.recv()
        with pytest.raises(ValueError, match="no action factorization"):
            pool.query_mask(np.full((2, 1), -1, np.int32), _ids(2))
        sids = np.asarray(pool.snapshot(_ids(2)), np.int32)
        for state_ids in (sids, np.full(2, -1, np.int32)):  # -1 rows are not evidence of a width
            with pytest.raises(ValueError, match="width 0"):
                pool.query_state_mask(state_ids, np.full((2, 1), -1, np.int32))


def test_factored_queries_on_zero_rows_and_dead_rows_keep_pad_width():
    with rustpool.make("BinPack-v0", num_envs=2, batch_size=2, factored=True) as pool:
        pad = pool.action_factorization()["pad_width"]
        pool.async_reset()
        pool.recv()
        assert pool.query_mask(np.zeros((0, 1), np.int32), _ids(0)).shape == (0, pad)
        dead = pool.query_state_mask(np.full(3, -1, np.int32), np.full((3, 1), -1, np.int32))
        assert dead.shape == (3, pad) and not dead.any()


@pytest.mark.torch
def test_torch_query_mask_on_a_flat_env_raises():
    pytest.importorskip("torch")
    from rustpool.torch_api import TorchEnvPool

    pool = TorchEnvPool("Maze-v0", num_envs=2)
    try:
        pool.reset()
        with pytest.raises(ValueError, match="no action factorization"):
            pool.query_mask(np.full((2, 1), -1, np.int32))
    finally:
        pool.close()


@pytest.mark.jax
def test_jax_query_mask_on_a_flat_env_raises_at_trace_time():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from rustpool.jax_api import JaxPool

    pool = JaxPool("Maze-v0", num_envs=2)
    try:
        prefixes, ids = jnp.full((2, 1), -1, jnp.int32), jnp.arange(2, dtype=jnp.int32)
        with pytest.raises(ValueError, match="no action factorization"):
            jax.jit(pool.query_mask)(prefixes, ids)
        with pytest.raises(ValueError, match="no action factorization"):
            jax.jit(pool.query_state_mask)(ids, prefixes)
    finally:
        pool.close()
