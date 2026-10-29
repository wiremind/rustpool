"""Public-surface smoke tests: the package exports, aliases, and registry helpers."""

import subprocess
import sys

import pytest

import rustpool
from rustpool import _rustpool


def test_pool_class_is_rlpool():
    assert rustpool.RlPool.__name__ == "RlPool"
    assert rustpool.RlPool.__module__ == "rustpool._rustpool"


def test_myenvpool_is_a_deprecated_alias():
    with pytest.warns(DeprecationWarning, match="RlPool"):
        assert rustpool.MyEnvPool is rustpool.RlPool
    assert _rustpool.MyEnvPool is _rustpool.RlPool


def test_list_envs_matches_registry_and_is_sorted():
    envs = rustpool.list_envs()
    assert set(envs) == {"BinPack-v0", "CartPole-v1", "Maze-v0"}
    assert envs == sorted(envs)
    assert envs == list(_rustpool.registered_tasks())


def test_version_is_a_string():
    assert isinstance(rustpool.__version__, str) and rustpool.__version__


def test_make_builds_a_pool():
    pool = rustpool.make("Maze-v0", num_envs=2, batch_size=2)
    try:
        assert isinstance(pool, rustpool.RlPool)
        assert pool.num_envs == 2 and pool.batch_size == 2
    finally:
        pool.close()


def test_import_does_not_pull_jax_or_torch():
    # Isolated subprocess: `import rustpool` must stay lightweight (optional backends are lazy).
    code = (
        "import sys, rustpool; "
        "assert 'jax' not in sys.modules, 'rustpool eagerly imported jax'; "
        "assert 'torch' not in sys.modules, 'rustpool eagerly imported torch'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pool_is_a_context_manager():
    with rustpool.make("Maze-v0", num_envs=2, batch_size=2) as pool:
        pool.async_reset()
        pool.recv()
    try:
        pool.recv()
    except RuntimeError as err:
        assert "closed" in str(err)
    else:
        raise AssertionError("the pool should be closed after the with block")

