"""The ``--factored`` AlphaZero candidate sampler (examples/alphazero_jax.py).

Without the full mask, each tree node's children are ``K`` full actions sampled without
replacement from the factored policy, with legality read per factor on the stored state. Check
that the candidates are legal and distinct, that ``query_state_mask`` agrees with the live
``query_mask``, and that the sampling distribution is exactly the policy's (K=1) and exactly
sampling without replacement (K=2), for the 2-factor chain (item -> ems) and the 4-factor one
(item -> orientation -> container -> ems, ``num_containers``/``orientations`` > 1). Also checks
the N-factor flat-prior composition (``networks.factored_flat_logits``).

Not collected on its own (no ``test_`` prefix): ``test_alphazero_candidates.py`` runs this file in
a separate pytest process, so the JAX/flax/mctx state these tests build does not stay in the main
suite's process. Run it directly with ``pytest tests/alphazero_candidates_checks.py``.
"""

import ctypes
import gc
import itertools
import math
import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.jax
jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("mctx")
pytest.importorskip("optax")
jnp = jax.numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
from alphazero_jax import make_candidate_sampler  # noqa: E402
from networks import FactoredActorCritic, factored_flat_logits  # noqa: E402

from rustpool.jax_api import JaxPool  # noqa: E402

MI, ME = 10, 12
# emit_flat_mask=True only so the test can check legality against the full mask; the sampler
# itself never reads it.
KW = dict(max_items=MI, max_ems=ME, factored=True, emit_flat_mask=True)
# Distribution tests draw N samples in chunks: the CI runner is memory-limited, and one
# N-row batch (plus the reference sampler's N x |A| temporaries) peaked near 1 GB.
CHUNK = 2_000
_open_pools = []


def _malloc_trim():
    """Hand freed heap pages back to the OS (glibc keeps them otherwise); no-op elsewhere."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _release_memory():
    """Close every pool a test opened (each holds worker threads, envs and stored states) and
    drop JAX's compiled executables, which otherwise outlive the test in the shared process."""
    yield
    while _open_pools:
        _open_pools.pop().close()
    jax.clear_caches()
    gc.collect()
    _malloc_trim()


def _setup(pool_cls, num_envs, seed=0):
    pool = pool_cls("BinPack-v0", num_envs, seed=seed, **KW)
    _open_pools.append(pool)
    net = FactoredActorCritic(factor_widths=(MI, ME), hidden_sizes=(32,))
    out = pool.reset(jax.random.key(seed))
    params = net.init(jax.random.key(1), out.obs)
    # Sharpen the (near-uniform at init) policy so the distribution tests are informative.
    params = jax.tree.map(lambda x: x * 40.0, params)
    return pool, net, params, out


def test_candidates_are_legal_and_distinct():
    B, K = 8, 7
    pool, net, params, out = _setup(JaxPool, B)
    expand, _ = make_candidate_sampler(pool, net, (MI, ME), K)
    rng = np.random.default_rng(0)
    for t in range(12):
        ids = pool.snapshot(jnp.arange(B, dtype=jnp.int32))
        # query_state_mask on the snapshot == query_mask on the live env.
        pre = jnp.full((B, 1), -1, jnp.int32)
        live = pool.query_mask(pre, jnp.arange(B))
        np.testing.assert_array_equal(np.asarray(pool.query_state_mask(ids, pre)), np.asarray(live))

        logits, _, cand = expand(params, ids, out.obs, jax.random.key(t))
        mask = np.asarray(out.action_mask)  # [B, MI*ME] full mask (test-only)
        act, valid = np.asarray(cand.action), np.asarray(cand.valid)
        for b in range(B):
            assert mask[b, act[b, valid[b]]].all(), "an illegal action was marked valid"
            assert len(set(act[b, valid[b]])) == valid[b].sum(), "duplicate candidates"
            assert valid[b].sum() == min(K, mask[b].sum())
        assert (np.asarray(logits)[~valid] <= -1e8).all()
        pool.release_batch(ids)
        a = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in mask], np.int32)
        out = pool.step(jnp.asarray(a))


def _sample_counts(pool, expand, params, obs0, n, size):
    """Counts of the actions ``expand`` keeps over ``n`` expansions of ONE state, in chunks of
    ``CHUNK`` rows: one snapshot, its handle repeated (the sampler only reads stored states)."""
    sid = pool.snapshot(jnp.zeros(1, jnp.int32))
    ids = jnp.broadcast_to(sid, (CHUNK,))
    obs_n = jax.tree.map(lambda x: jnp.broadcast_to(x, (CHUNK, *x.shape[1:])), obs0)
    expand = jax.jit(expand)
    counts = np.zeros(size, np.int64)
    for i in range(n // CHUNK):
        _, _, cand = expand(params, ids, obs_n, jax.random.fold_in(jax.random.key(0), i))
        act, valid = np.asarray(cand.action), np.asarray(cand.valid)
        assert valid.all()
        counts += np.bincount(act.reshape(-1), minlength=size)
    pool.release_batch(sid)
    return counts


def _exact_joint(net, params, obs0, flat_mask0):
    """The factored policy's exact joint over full actions at one state, under its masks."""
    feats = net.apply(params, obs0, method=net.features)
    m = flat_mask0.reshape(MI, ME)
    il = net.apply(params, feats, jnp.zeros((1, 0)), 0, method=net.factor_logits)[0]
    p_item = jax.nn.softmax(jnp.where(m.any(1), il, -1e9))
    p = []
    for i in range(MI):
        el = net.apply(params, feats, jax.nn.one_hot(jnp.array([i]), MI), 1,
                       method=net.factor_logits)[0]
        p.append(p_item[i] * jax.nn.softmax(jnp.where(m[i], el, -1e9)))
    return np.asarray(jnp.stack(p).reshape(-1))


@pytest.mark.parametrize("k", [1, 2])
def test_candidate_sampling_matches_policy(k):
    """N samples at ONE state (one snapshot of env 0, expanded N times): K=1 frequencies == the joint policy;
    K=2 inclusion frequencies == sampling 2 without replacement from it.

    The state is one placement in, so items have several legal ems each: at reset every item
    has exactly one (the empty container), which would leave the ems stage untested."""
    N = 20_000
    pool, net, params, out = _setup(JaxPool, 1, seed=1)
    out = pool.step(jnp.asarray([np.flatnonzero(np.asarray(out.action_mask[0]))[-1]], jnp.int32))
    obs0, mask0 = out.obs, np.asarray(out.action_mask[0])
    assert (mask0.reshape(MI, ME).sum(1) >= 2).sum() >= 4, "want several multi-ems items"
    p = _exact_joint(net, params, obs0, jnp.asarray(mask0))
    assert p.max() < 0.5 and (p > 0.02).sum() >= 8, "policy too peaked/flat to be informative"

    expand, _ = make_candidate_sampler(pool, net, (MI, ME), k)
    freq = _sample_counts(pool, expand, params, obs0, N, MI * ME) / N

    if k == 1:
        expected = p
    else:  # P(a in sample) = p_a + sum_{b != a} p_b * p_a / (1 - p_b)
        expected = p + p * (np.sum(p / (1 - p)) - p / (1 - p))
    np.testing.assert_allclose(freq, expected, atol=0.01)


# --- 4 factors: item -> orientation -> container -> ems -------------------------------------
MULTI = dict(max_items=8, max_ems=8, factored=True, emit_flat_mask=True, num_containers=2,
             orientations=2)
W4 = (16, 2, 2, 8)


def _setup_multi(num_envs, seed):
    pool = JaxPool("BinPack-v0", num_envs, seed=seed, **MULTI)
    _open_pools.append(pool)
    assert tuple(pool.factorization["factor_widths"]) == W4
    net = FactoredActorCritic(factor_widths=W4, hidden_sizes=(32,))
    out = pool.reset(jax.random.key(seed))
    params = jax.tree.map(lambda x: x * 40.0, net.init(jax.random.key(1), out.obs))
    return pool, net, params, out


def _exact_joint_n(net, params, obs0, flat_mask0, widths):
    """Exact joint of the autoregressive policy at one state, each factor's softmax masked by
    the existence projection of the flat mask (what query_mask returns)."""
    feats = net.apply(params, obs0, method=net.features)
    m = np.asarray(flat_mask0).reshape(widths)
    p = np.zeros(widths)
    cache = {}

    def probs(prefix):
        if prefix not in cache:
            k = len(prefix)
            prev = [jax.nn.one_hot(jnp.array([v]), widths[j]) for j, v in enumerate(prefix)]
            prev = jnp.concatenate(prev, -1) if prev else jnp.zeros((1, 0))
            lk = net.apply(params, feats, prev, k, method=net.factor_logits)[0]
            mk = m[prefix].reshape(widths[k], -1).any(1)
            cache[prefix] = np.asarray(jax.nn.softmax(jnp.where(mk, lk, -1e9)))
        return cache[prefix]

    for a in map(tuple, np.argwhere(m)):
        p[a] = math.prod(probs(tuple(int(v) for v in a[:k]))[a[k]] for k in range(len(widths)))
    return p.reshape(-1)


def _inclusion_without_replacement(p, k, draws=400_000, seed=0):
    """Reference inclusion frequencies of exact sampling without replacement (Gumbel-top-k)."""
    rng = np.random.default_rng(seed)
    logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf)
    cnt = np.zeros(p.size)
    for _ in range(draws // CHUNK):
        g = logp + rng.gumbel(size=(CHUNK, p.size))
        np.add.at(cnt, np.argpartition(-g, k - 1, axis=1)[:, :k].reshape(-1), 1)
    return cnt / draws


def test_multi_factor_candidates_are_legal_and_distinct():
    B, K = 6, 9
    pool, net, params, out = _setup_multi(B, seed=0)
    expand, _ = make_candidate_sampler(pool, net, W4, K)
    rng = np.random.default_rng(0)
    for t in range(10):
        ids = pool.snapshot(jnp.arange(B, dtype=jnp.int32))
        _, _, cand = expand(params, ids, out.obs, jax.random.key(t))
        mask = np.asarray(out.action_mask)
        act, valid = np.asarray(cand.action), np.asarray(cand.valid)
        ch = np.asarray(cand.choices)
        np.testing.assert_array_equal(np.ravel_multi_index(np.moveaxis(ch, -1, 0), W4), act)
        for b in range(B):
            assert mask[b, act[b, valid[b]]].all(), "an illegal action was marked valid"
            assert len(set(act[b, valid[b]])) == valid[b].sum(), "duplicate candidates"
            assert valid[b].sum() == min(K, mask[b].sum())
        pool.release_batch(ids)
        a = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in mask], np.int32)
        out = pool.step(jnp.asarray(a))


@pytest.mark.parametrize("k", [1, 3])
def test_multi_factor_sampling_matches_policy(k):
    """4-factor chain, one state with several options at every level: K=1 frequencies == the
    exact joint; K=3 inclusion frequencies == exact sampling without replacement."""
    N = 20_000
    pool, net, params, out = _setup_multi(1, seed=1)
    out = pool.step(jnp.asarray([np.flatnonzero(np.asarray(out.action_mask[0]))[-1]], jnp.int32))
    obs0, mask0 = out.obs, np.asarray(out.action_mask[0])
    m = mask0.reshape(W4)
    assert m.any(axis=(0, 2, 3)).all() and m.any(axis=(0, 1, 3)).all(), "want >1 orient/container"
    assert (m.reshape(-1, W4[3]).sum(1) >= 2).any(), "want several legal ems somewhere"
    p = _exact_joint_n(net, params, obs0, mask0, W4)
    assert abs(p.sum() - 1) < 1e-5
    assert p.max() < 0.5 and (p > 0.02).sum() >= 6, "policy too peaked/flat to be informative"

    expand, _ = make_candidate_sampler(pool, net, W4, k)
    freq = _sample_counts(pool, expand, params, obs0, N, p.size) / N
    expected = p if k == 1 else _inclusion_without_replacement(p, k)
    np.testing.assert_allclose(freq, expected, atol=0.01)


def test_factored_flat_logits_matches_direct_composition():
    """The affine-head shortcut == summing each factor's log-softmax head at every prefix."""
    widths = (4, 2, 3, 5)
    net = FactoredActorCritic(factor_widths=widths, hidden_sizes=(16,))
    obs = {"x": jax.random.normal(jax.random.key(0), (3, 7))}
    params = jax.tree.map(lambda x: x * 10.0, net.init(jax.random.key(1), obs))
    flat, value = factored_flat_logits(net, params, obs, widths)
    assert flat.shape == (3, math.prod(widths)) and value.shape == (3,)
    feats = net.apply(params, obs, method=net.features)
    for idx, a in enumerate(itertools.product(*map(range, widths))):
        want = 0.0
        for k in range(len(widths)):
            prev = [jax.nn.one_hot(jnp.full((3,), a[j]), widths[j]) for j in range(k)]
            prev = jnp.concatenate(prev, -1) if prev else jnp.zeros((3, 0))
            lk = net.apply(params, feats, prev, k, method=net.factor_logits)
            want = want + jax.nn.log_softmax(lk)[:, a[k]]
        np.testing.assert_allclose(np.asarray(flat[:, idx]), np.asarray(want), atol=2e-4)
