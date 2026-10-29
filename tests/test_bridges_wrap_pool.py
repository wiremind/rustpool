"""JaxPool.from_pools / TorchEnvPool.from_pool wrap pools built outside rustpool.make.

An external env crate's pool is a different Python class from ``rustpool.RlPool``, so the
bridges are duck-typed. ``ForeignPool`` stands in for it: it is not a ``RlPool`` and only
forwards the pool API. ``_rustpool._pool_from_factory`` builds the pool underneath, the way an
external crate does.
"""

import numpy as np
import pytest

import rustpool
from rustpool import _rustpool


class ForeignPool:
    """Not a rustpool.RlPool: forwards the pool API to a pool built like an external crate's."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def foreign(task_id, n, **kw):
    pool = ForeignPool(_rustpool._pool_from_factory(task_id, n, n, **kw))
    assert not isinstance(pool, rustpool.RlPool)
    return pool


@pytest.mark.jax
def test_jaxpool_from_foreign_pool_matches_jaxpool():
    jax = pytest.importorskip("jax")
    from rustpool.jax_api import JaxPool

    wrapped = JaxPool.from_pools(foreign("Maze-v0", 4, seed=3))
    native = JaxPool("Maze-v0", 4, seed=3)
    assert (wrapped.num_envs, wrapped.num_devices, wrapped.num_actions) == (4, 1, 4)
    key = jax.random.key(0)
    outs = []
    for pool in (wrapped, native):
        out = jax.jit(pool.reset)(key)
        acts = jax.numpy.zeros(4, jax.numpy.int32)
        for _ in range(5):
            out = jax.jit(pool.step)(acts)
        outs.append(out)
        pool.close()
    for a, b in zip(jax.tree_util.tree_leaves(outs[0]), jax.tree_util.tree_leaves(outs[1]),
                    strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.jax
def test_jaxpool_from_foreign_pool_supports_search_and_factored_masks():
    jax = pytest.importorskip("jax")
    from rustpool.jax_api import JaxPool

    pool = JaxPool.from_pools([foreign("BinPack-v0", 3, factored=True, max_items=10,
                                       max_ems=20)])
    out = pool.reset(jax.random.key(0))
    assert pool.factorization is not None
    sids = pool.snapshot(jax.numpy.arange(3, dtype=jax.numpy.int32))
    masks = pool.query_state_mask(sids, jax.numpy.full((3, 1), -1, jax.numpy.int32))
    assert masks.shape == (3, pool.factorization["pad_width"])
    np.testing.assert_array_equal(np.asarray(masks), np.asarray(out.action_mask))
    pool.release_batch(sids)
    pool.close()


@pytest.mark.jax
def test_jaxpool_from_pools_validates():
    pytest.importorskip("jax")
    from rustpool.jax_api import JaxPool

    with pytest.raises(ValueError, match="batch_size == num_envs"):
        JaxPool.from_pools(_rustpool._pool_from_factory("Maze-v0", 4, 2))
    with pytest.raises(ValueError, match="same num_envs"):
        JaxPool.from_pools([foreign("Maze-v0", 4), foreign("Maze-v0", 2)])
    with pytest.raises(TypeError, match="rustpool pool"):
        JaxPool.from_pools(object())


@pytest.mark.torch
def test_torchenvpool_from_foreign_pool_matches_torchenvpool():
    torch = pytest.importorskip("torch")
    from rustpool.torch_api import TorchEnvPool

    wrapped = TorchEnvPool.from_pool(foreign("Maze-v0", 4, seed=2, autoreset="next_step"))
    native = TorchEnvPool("Maze-v0", 4, seed=2, autoreset="next_step")
    for pool in (wrapped, native):
        pool.reset()
    for _ in range(20):
        a = torch.randint(0, 4, (4,), generator=torch.Generator().manual_seed(0))
        (o1, r1, d1, _), (o2, r2, d2, _) = wrapped.step(a), native.step(a)
        assert all(torch.equal(o1[k], o2[k]) for k in o1)
        assert torch.equal(r1, r2) and torch.equal(d1, d2)
    wrapped.close()
    native.close()


@pytest.mark.torch
def test_torchenvpool_from_pool_validates():
    pytest.importorskip("torch")
    from rustpool.torch_api import TorchEnvPool

    with pytest.raises(ValueError, match="batch_size == num_envs"):
        TorchEnvPool.from_pool(_rustpool._pool_from_factory("Maze-v0", 4, 2))
    with pytest.raises(TypeError, match="rustpool pool"):
        TorchEnvPool.from_pool(object())


def test_bridges_check_imports_neither_jax_nor_torch():
    import subprocess
    import sys

    code = ("import sys, rustpool._pool_check; "
            "assert 'jax' not in sys.modules and 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
