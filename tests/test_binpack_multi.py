"""BinPack-v0 with several containers (``num_containers``) and item orientations
(``orientations``): kwargs, specs, observation layout, and the factored chain
item -> [orientation] -> [container] -> ems against the flat mask through the Python transport.
(C = O = 1 bit-identity is pinned by ``test_binpack_anchor.py``.)
"""

import numpy as np
import pytest

import rustpool

MI, ME = 6, 8


def make(n=4, **kw):
    return rustpool.make("BinPack-v0", num_envs=n, batch_size=n, seed=2, max_items=MI, max_ems=ME,
                         **kw)


def widths(c, o):
    return [MI * c] + ([o] if o > 1 else []) + ([c] if c > 1 else []) + [ME]


@pytest.mark.parametrize("kw", [{"orientations": 3}, {"orientations": 0}, {"num_containers": 0}])
def test_invalid_kwargs_raise(kw):
    with pytest.raises(ValueError):
        make(**kw)


@pytest.mark.parametrize("c,o,names", [
    (1, 1, ["item", "ems"]),
    (2, 1, ["item", "container", "ems"]),
    (1, 6, ["item", "orientation", "ems"]),
    (3, 2, ["item", "orientation", "container", "ems"]),
])
def test_specs(c, o, names):
    pool = make(factored=True, num_containers=c, orientations=o)
    try:
        spec = pool.action_factorization()
        w = widths(c, o)
        assert spec["factor_names"] == names and spec["name"] == "binpack_" + "_".join(names)
        assert spec["factor_widths"] == w and spec["num_factors"] == len(w)
        assert spec["conditional"] is True
        assert spec["pad_width"] == max(w)
        assert spec["flat_action_size"] == np.prod(w) == MI * c * o * c * ME
        assert pool.action_spec()[0] == np.prod(w)

        obs_spec = pool.observation_spec()
        assert obs_spec["items"] == ([MI * c, 3], "float32")
        assert obs_spec["ems"] == ([c * ME, 6], "float32")
        assert obs_spec["ems_mask"] == ([c * ME], "bool")
        assert ("items_container" in obs_spec) == (c > 1)
        assert ("items_orientation" in obs_spec) == (o > 1)
    finally:
        pool.close()


def _prefix(choices, num_factors):
    return np.array(list(choices) + [-1] * (num_factors - 1 - len(choices)), np.int32)


@pytest.mark.parametrize("c,o", [(1, 1), (2, 1), (1, 6), (3, 2)])
def test_chain_masks_match_flat_mask(c, o):
    """Each level's query_mask is the existence projection of the flat mask, at several states;
    query_state_mask on a snapshot returns the same rows."""
    B = 3
    pool = make(B, factored=True, emit_flat_mask=True, num_containers=c, orientations=o)
    w = widths(c, o)
    nf = len(w)
    ids = np.arange(B, dtype=np.int32)
    rng = np.random.default_rng(0)
    try:
        pool.async_reset()
        obs, _, _, info = pool.recv()
        for _ in range(4):
            flat = np.asarray(info["action_mask"]).reshape(B, *w)
            sids = np.asarray(pool.snapshot(ids), np.int32)
            for b in range(B):
                # every prefix along a few random valid paths, plus the empty prefix
                for _path in range(3):
                    chosen = []
                    for k in range(nf):
                        pre = _prefix(chosen, nf)[None]
                        got = np.asarray(pool.query_mask(pre, ids[b:b + 1]))[0, : w[k]]
                        stored = np.asarray(pool.query_state_mask(sids[b:b + 1], pre))[0, : w[k]]
                        sub = flat[b][tuple(chosen)]
                        want = sub.reshape(w[k], -1).any(axis=1)
                        np.testing.assert_array_equal(got, want, err_msg=f"level {k} {chosen}")
                        np.testing.assert_array_equal(stored, got)
                        if not got.any():
                            break
                        chosen.append(int(rng.choice(np.flatnonzero(got))))
            pool.release_batch(sids)
            m = np.asarray(info["action_mask"])
            a = np.array([rng.choice(np.flatnonzero(r)) if r.any() else 0 for r in m], np.int32)
            pool.send(a, ids)
            obs, _, _, info = pool.recv()
    finally:
        pool.close()


def test_placement_observations():
    """items_container / items_orientation / placed_mask reflect the decoded flat action."""
    c, o = 3, 6
    pool = make(1, num_containers=c, orientations=o)
    w = [MI * c, o, c, ME]
    try:
        pool.async_reset()
        obs, _, _, info = pool.recv()
        assert (obs["items_container"] == -1).all() and (obs["items_orientation"] == -1).all()
        rng = np.random.default_rng(1)
        for _ in range(5):
            m = np.asarray(info["action_mask"])[0]
            if not m.any():
                break
            a = int(rng.choice(np.flatnonzero(m)))
            item, oo, cc, _ = np.unravel_index(a, w)
            pool.send(np.array([a], np.int32), np.array([0], np.int32))
            obs, rew, done, info = pool.recv()
            if done[0]:
                break
            assert obs["placed_mask"][0, item]
            assert obs["items_container"][0, item] == cc
            assert obs["items_orientation"][0, item] == oo
            assert rew[0] > 0
    finally:
        pool.close()


def test_containers_double_the_capacity():
    """Items are generated per container, so C containers carry ~C times the item volume and
    the dense reward is normalised by the total capacity: a full episode sums to <= 1."""
    pool = make(8, num_containers=2)
    try:
        pool.async_reset()
        obs, _, _, info = pool.recv()
        assert obs["items_mask"].sum(axis=1).max() > MI  # more items than one container holds
        totals, ret = [], np.zeros(8)
        rng = np.random.default_rng(3)
        for _ in range(200):
            m = np.asarray(info["action_mask"])
            a = np.array([rng.choice(np.flatnonzero(r)) if r.any() else 0 for r in m], np.int32)
            pool.send(a, np.arange(8, dtype=np.int32))
            obs, rew, done, info = pool.recv()
            ret += rew
            totals += list(ret[done])
            ret[done] = 0.0
        assert totals and max(totals) <= 1.0 + 1e-5 and min(totals) > 0.0
    finally:
        pool.close()



# --- shuffle_items -------------------------------------------------------------------------------

def _first_obs(**kw):
    pool = make(n=3, num_containers=3, **kw)
    try:
        pool.async_reset(np.array([11, 12, 13], np.int64))
        obs, _, _, info = pool.recv()
        return obs, info
    finally:
        pool.close()


def test_shuffle_items_default_off():
    a, _ = _first_obs()
    b, _ = _first_obs(shuffle_items=False)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_shuffle_items_permutes_items_with_their_locations():
    plain, _ = _first_obs()
    shuf, info = _first_obs(shuffle_items=True)
    again, _ = _first_obs(shuffle_items=True)
    np.testing.assert_array_equal(shuf["items"], again["items"])  # deterministic per seed
    assert not np.array_equal(plain["items"], shuf["items"])
    def pairs(o, e, n):
        rows = np.concatenate([o["items"][e, :n], o["ground_truth_locations"][e, :n]], axis=1)
        return sorted(map(tuple, rows))

    for e in range(3):
        n = int(plain["items_mask"][e].sum())
        assert int(shuf["items_mask"][e].sum()) == n
        assert pairs(plain, e, n) == pairs(shuf, e, n)
    # Shuffling reorders the items, never the action layout.
    assert info["action_mask"].shape[1] == MI * 3 * 3 * ME
