"""Coverage for ``RlPool.query_state_mask``: conditional (factored) masks of STORED registry
states (``snapshot`` / ``simulate_batch`` handles) instead of the live envs.

Pins the contract a batched factored tree search relies on: rows in input order (ids may
repeat), ``state_id < 0`` -> all-false row without a lookup, unknown / released ids -> a clean
``ValueError``, a row of the wrong width -> ``ValueError`` (never silently empty), ``None`` for an
env with no conditional masks, and agreement with the live ``query_mask``.
"""

import numpy as np
import pytest

import rustpool

ITEMS, EMS = 10, 20  # BinPack(factored=True): pad_width = max(ITEMS, EMS)


def _factored_pool(num_envs=4, seed=1):
    pool = rustpool.make(
        "BinPack-v0",
        num_envs=num_envs,
        batch_size=num_envs,
        max_items=ITEMS,
        max_ems=EMS,
        factored=True,
        emit_flat_mask=True,  # the stored-vs-live test steps with a valid flat action
        seed=seed,
    )
    pool.async_reset()
    _, _, _, info = pool.recv()
    order = np.argsort(np.asarray(info["env_id"]))
    env_ids = np.asarray(info["env_id"], dtype=np.int32)[order]
    flat = np.asarray(info["action_mask"])[order]
    return pool, env_ids, flat


def _no_prefix(n):
    return np.full((n, 1), -1, np.int32)


def test_query_state_mask_matches_live_query_mask():
    pool, env_ids, _ = _factored_pool()
    try:
        sids = np.asarray(pool.snapshot(env_ids), np.int32)
        stored = pool.query_state_mask(sids, _no_prefix(4))
        live = pool.query_mask(_no_prefix(4), env_ids)
        assert stored.dtype == np.bool_ and stored.shape == (4, max(ITEMS, EMS))
        np.testing.assert_array_equal(stored, live)
        # Factor 1 (EMS given an item) on stored states == on the live envs.
        for b in range(4):
            for item in np.flatnonzero(stored[b, :ITEMS])[:3]:
                pre = np.array([[item]], np.int32)
                np.testing.assert_array_equal(
                    pool.query_state_mask(sids[b : b + 1], pre),
                    pool.query_mask(pre, env_ids[b : b + 1]),
                )
    finally:
        pool.close()


def test_query_state_mask_reads_the_stored_state_not_the_live_env():
    pool, env_ids, flat = _factored_pool()
    try:
        sids = np.asarray(pool.snapshot(env_ids), np.int32)
        before = pool.query_state_mask(sids, _no_prefix(4))
        actions = np.argmax(flat, axis=1).astype(np.int32)  # a valid flat action per env
        assert flat[np.arange(4), actions].all()

        # simulate_batch children == the live envs stepped with the same actions.
        _, _, sim_done, sim_info = pool.simulate_batch(sids, actions)
        children = np.asarray(sim_info["state_id"], np.int32)
        pool.send(actions, env_ids)
        _, _, done, info = pool.recv()
        order = np.argsort(np.asarray(info["env_id"]))
        live_done = np.asarray(done)[order]
        child_masks = pool.query_state_mask(children, _no_prefix(4))
        live_masks = pool.query_mask(_no_prefix(4), env_ids)
        keep = ~(np.asarray(sim_done) | live_done)  # terminal live envs auto-reset
        np.testing.assert_array_equal(child_masks[keep], live_masks[keep])

        # The parents are untouched by the live step, the fork and the queries.
        np.testing.assert_array_equal(pool.query_state_mask(sids, _no_prefix(4)), before)
    finally:
        pool.close()


def test_query_state_mask_dead_rows_duplicates_and_empty_batch():
    pool, env_ids, _ = _factored_pool()
    try:
        sids = np.asarray(pool.snapshot(env_ids), np.int32)
        ref = pool.query_state_mask(sids, _no_prefix(4))
        ids = np.array([-1, sids[2], -7, sids[2], sids[0]], np.int32)
        out = pool.query_state_mask(ids, _no_prefix(5))
        assert out.shape == (5, max(ITEMS, EMS))
        assert not out[0].any() and not out[2].any()
        np.testing.assert_array_equal(out[1], ref[2])
        np.testing.assert_array_equal(out[3], ref[2])
        np.testing.assert_array_equal(out[4], ref[0])

        # All rows dead: never looked up (not even after a full release), width from the spec.
        pool.release_batch(sids)
        dead = pool.query_state_mask(np.array([-1, -1, -3], np.int32), _no_prefix(3))
        assert dead.shape == (3, max(ITEMS, EMS)) and not dead.any()

        empty = pool.query_state_mask(np.zeros(0, np.int32), np.zeros((0, 1), np.int32))
        assert empty.shape == (0, max(ITEMS, EMS))
    finally:
        pool.close()


def test_query_state_mask_unknown_or_released_ids_raise():
    pool, env_ids, _ = _factored_pool()
    try:
        sids = np.asarray(pool.snapshot(env_ids), np.int32)
        with pytest.raises(ValueError, match="invalid state_id 99999"):
            pool.query_state_mask(np.array([sids[0], 99999], np.int32), _no_prefix(2))
        pool.release_batch(sids[:1])
        with pytest.raises(ValueError, match=f"invalid state_id {sids[0]}"):
            pool.query_state_mask(sids, _no_prefix(4))
        # The pool keeps working after the error.
        assert pool.query_state_mask(sids[1:], _no_prefix(3)).shape == (3, max(ITEMS, EMS))
    finally:
        pool.close()


def test_query_state_mask_argument_and_width_validation():
    pool, env_ids, _ = _factored_pool()
    try:
        sids = np.asarray(pool.snapshot(env_ids), np.int32)
        with pytest.raises(ValueError, match="must match"):  # rows != state_ids length
            pool.query_state_mask(sids, _no_prefix(3))
        width = max(ITEMS, EMS)
        assert pool.query_state_mask(sids, _no_prefix(4), width).shape == (4, width)
        assert pool.query_state_mask(sids, _no_prefix(4), width=width).shape == (4, width)
        with pytest.raises(ValueError, match="width"):
            pool.query_state_mask(sids, _no_prefix(4), width=width + 3)
        with pytest.raises(ValueError, match="width"):
            pool.query_state_mask(sids, _no_prefix(4), width=0)
    finally:
        pool.close()


@pytest.mark.parametrize(
    "task, kwargs", [("Maze-v0", {}), ("BinPack-v0", {"max_items": ITEMS, "max_ems": EMS})]
)
def test_query_state_mask_without_conditional_masks(task, kwargs):
    """Flat action spaces raise ValueError, with or without a requested width (their conditional
    masks are EMPTY, which must never pass as a [n, W] batch)."""
    pool = rustpool.make(task, num_envs=2, batch_size=2, **kwargs)
    try:
        pool.async_reset()
        _, _, _, info = pool.recv()
        sids = np.asarray(pool.snapshot(np.asarray(info["env_id"], np.int32)), np.int32)
        assert pool.action_factorization() is None
        with pytest.raises(ValueError, match="width 0"):
            pool.query_state_mask(sids, _no_prefix(2))
        with pytest.raises(ValueError, match="width 0"):
            pool.query_state_mask(sids, _no_prefix(2), width=4)
    finally:
        pool.close()


@pytest.mark.jax
def test_jax_pool_query_state_mask_matches_query_mask():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from rustpool.jax_api import JaxPool

    jp = JaxPool("BinPack-v0", 4, max_items=ITEMS, max_ems=EMS, factored=True, seed=3)
    try:
        out = jp.reset(jax.random.PRNGKey(0))
        sids = jp.snapshot(out.env_id)
        pre = jnp.full((4, 1), -1, jnp.int32)
        stored = jax.jit(jp.query_state_mask)(sids, pre)
        live = jp.query_mask(pre, out.env_id)
        np.testing.assert_array_equal(np.asarray(stored), np.asarray(live))
        jp.release_batch(sids)
    finally:
        jp.close()
