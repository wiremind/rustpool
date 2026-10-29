"""Dispatch modes (``per_env``, ``per_worker``, ``auto``) are a transport choice only: identical
trajectories, identical ``query_mask`` answers, and each env back once per ``send`` in async mode.
A rejected ``send`` dispatches nothing."""

import numpy as np
import pytest

import rustpool

MODES = ("per_env", "per_worker", "auto")
ENVS = {
    "CartPole-v1": {},
    "Maze-v0": {},
    "BinPack-v0": {"max_items": 6, "max_ems": 8, "num_containers": 2},
    "BinPack-v0/factored": {"max_items": 6, "max_ems": 8, "num_containers": 2, "factored": True},
}


def _make(name, dispatch, n=16, b=None, threads=3, seed=5):
    return rustpool.make(name.split("/")[0], num_envs=n, batch_size=b or n, seed=seed,
                         num_threads=threads, dispatch=dispatch, **ENVS[name])


def _valid_actions(pool, info, rng, ids):
    """Random valid actions: from the flat mask, down the factored chain, or uniform."""
    spec = pool.action_factorization()
    n = len(ids)
    if spec is not None and spec.get("conditional") and info["action_mask"].shape[1] == \
            spec["pad_width"] and pool.action_spec()[0] != spec["pad_width"]:
        k, widths = spec["num_factors"], spec["factor_widths"]
        prefix = np.full((n, k - 1), -1, np.int32)
        flat = np.zeros(n, np.int64)
        for f in range(k):
            m = pool.query_mask(prefix, ids)[:, : widths[f]]
            ch = np.where(m, rng.random(m.shape), -1.0).argmax(1)
            flat = flat * widths[f] + ch
            if f < k - 1:
                prefix[:, f] = ch
        return flat.astype(np.int32)
    mask = info.get("action_mask")
    if mask is None or mask.shape[1] == 0:
        return rng.integers(0, pool.action_spec()[0], n).astype(np.int32)
    return np.where(mask, rng.random(mask.shape), -1.0).argmax(1).astype(np.int32)


@pytest.mark.parametrize("name", list(ENVS))
@pytest.mark.parametrize("threads", [1, 3, 16])
def test_dispatch_modes_give_identical_trajectories(name, threads):
    pools = [_make(name, d, threads=threads) for d in MODES]
    assert [p.dispatch for p in pools] == list(MODES)
    ids = np.arange(16, dtype=np.int32)
    try:
        outs = []
        for p in pools:
            p.async_reset()
            outs.append(p.recv())
        rng = np.random.default_rng(0)
        for _ in range(150):
            a = _valid_actions(pools[0], outs[0][3], rng, ids)
            outs = []
            for p in pools:
                p.send(a, ids)
                outs.append(p.recv())
            o0, r0, d0, i0 = outs[0]
            for o, r, d, i in outs[1:]:
                assert o.keys() == o0.keys()
                assert all(np.array_equal(o[k], o0[k]) for k in o0)
                assert np.array_equal(r, r0) and np.array_equal(d, d0)
                assert i.keys() == i0.keys()
                assert all(np.array_equal(i[k], i0[k]) for k in i0)
    finally:
        for p in pools:
            p.close()


@pytest.mark.parametrize("threads", [1, 3, 16])
def test_batched_query_mask_equals_per_row(threads):
    """Factored BinPack: per-worker (batched) query_mask answers row for row like the per-env
    one -- duplicate env_ids, every stage, interleaved with send/recv."""
    pools = [_make("BinPack-v0/factored", d, n=12, threads=threads, seed=2) for d in MODES]
    spec = pools[0].action_factorization()
    k, widths = spec["num_factors"], spec["factor_widths"]
    rng = np.random.default_rng(1)
    try:
        infos = []
        for p in pools:
            p.async_reset()
            infos.append(p.recv()[3])
        for _ in range(60):
            assert all(np.array_equal(i["action_mask"], infos[0]["action_mask"]) for i in infos)
            ids = np.concatenate([infos[0]["env_id"], infos[0]["env_id"][:5]]).astype(np.int32)
            prefix = np.full((len(ids), k - 1), -1, np.int32)
            for f in range(k):
                got = [p.query_mask(prefix, ids) for p in pools]
                assert got[0].shape == (len(ids), spec["pad_width"])
                assert all(np.array_equal(g, got[0]) for g in got[1:])
                m = got[0][:, : widths[f]]
                ch = np.where(m, rng.random(m.shape), -1.0).argmax(1)
                if f < k - 1:
                    prefix[:, f] = ch
            act = _valid_actions(pools[0], infos[0], rng, ids[:12])
            infos = []
            for p in pools:
                p.send(act, ids[:12])
                infos.append(p.recv()[3])
    finally:
        for p in pools:
            p.close()


@pytest.mark.parametrize("dispatch", MODES)
def test_async_returns_each_env_once_per_send(dispatch):
    n, b = 24, 6
    pool = _make("CartPole-v1", dispatch, n=n, b=b, threads=4)
    pool.async_reset()
    outstanding = set(range(n))  # sent (or reset) and not yet returned
    held = []  # returned last round and deliberately sent one round late
    try:
        for _ in range(500):
            _, _, _, info = pool.recv()
            got = info["env_id"].tolist()
            assert len(got) == b and len(set(got)) == b
            assert set(got) <= outstanding, "an env came back without being sent"
            outstanding -= set(got)
            to_send = held + got[: b // 2]
            held = got[b // 2:]
            pool.send(np.zeros(len(to_send), np.int32), np.asarray(to_send, np.int32))
            outstanding |= set(to_send)
    finally:
        pool.close()


def test_dispatch_rejects_unknown_mode():
    with pytest.raises(ValueError, match="dispatch"):
        rustpool.make("CartPole-v1", num_envs=2, dispatch="per_core")


@pytest.mark.parametrize("dispatch", MODES)
@pytest.mark.parametrize("bad_id", [4, -1])
def test_rejected_send_dispatches_nothing(dispatch, bad_id):
    """``send`` used to check each env_id while sending, so an invalid id in the middle of a
    batch raised only after the actions before it had reached their workers: those envs
    stepped once more than the caller knew, and their results surfaced in a later ``recv``."""
    ids = np.arange(4, dtype=np.int32)
    actions = np.array([1, 0, 1, 0], dtype=np.int32)
    poisoned, clean = (_make("CartPole-v1", dispatch, n=4, threads=2, seed=11) for _ in "ab")
    try:
        for pool in (poisoned, clean):
            pool.async_reset()
            pool.recv()
        bad_ids = ids.copy()
        bad_ids[2] = bad_id
        with pytest.raises(IndexError, match="out of bounds"):
            poisoned.send(actions, bad_ids)
        # The poisoned pool must now behave exactly like one that never saw the bad call.
        for _ in range(3):
            outs = []
            for pool in (poisoned, clean):
                pool.send(actions, ids)
                outs.append(pool.recv())
            (o1, r1, d1, i1), (o2, r2, d2, i2) = outs
            assert np.array_equal(i1["env_id"], i2["env_id"])
            assert all(np.array_equal(o1[k], o2[k]) for k in o1)
            assert np.array_equal(r1, r2) and np.array_equal(d1, d2)
    finally:
        poisoned.close()
        clean.close()
