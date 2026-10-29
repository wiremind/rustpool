"""JaxPool on two (forced) host devices: each pmap replica drives its own pool.

Not collected on its own (no ``test_`` prefix): ``test_jax_multidevice.py`` runs it in a child
process, because the forced device count must be set before JAX is imported. Run it directly
with ``XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest tests/jax_multidevice_checks.py``.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import rustpool  # noqa: E402
from rustpool.jax_api import JaxPool  # noqa: E402

D, N = 2, 4


def _reference(task, key):
    """A plain pool reset the way JaxPool's host callback seeds device ``key``."""
    pool = rustpool.make(task, num_envs=N, batch_size=N)
    base = int(np.asarray(key).reshape(-1)[-1]) & 0x7FFFFFFF
    pool.async_reset((base + np.arange(N)).astype(np.int64))
    return pool, pool.recv()


def test_two_devices_are_available():
    assert jax.local_device_count() == D


def test_each_replica_steps_its_own_pool():
    jpool = JaxPool("Maze-v0", N, num_devices=D)
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(0), D))
    actions = np.stack([np.full(N, d, np.int32) for d in range(D)])  # different per device
    out = jax.pmap(jpool.reset, axis_name="device")(keys)
    refs = [_reference("Maze-v0", keys[d]) for d in range(D)]
    for d, (_, (obs, *_)) in enumerate(refs):
        np.testing.assert_array_equal(np.asarray(out.obs["agent_pos"][d]), obs["agent_pos"])
    step = jax.pmap(jpool.step, axis_name="device")
    for _ in range(3):
        out = step(actions)
        for d, (ref, _) in enumerate(refs):
            ref.send(actions[d], np.arange(N, dtype=np.int32))
            obs, reward, done, _ = ref.recv()
            np.testing.assert_array_equal(np.asarray(out.obs["agent_pos"][d]), obs["agent_pos"])
            np.testing.assert_array_equal(np.asarray(out.reward[d]), reward)
            np.testing.assert_array_equal(np.asarray(out.done[d]), done)
    for ref, _ in refs:
        ref.close()
    jpool.close()


def test_search_calls_route_per_device():
    jpool = JaxPool("Maze-v0", N, num_devices=D)
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(1), D))
    out = jax.pmap(jpool.reset, axis_name="device")(keys)
    # The two pools hold different states, so a replica routed to the wrong pool shows up.
    assert not np.array_equal(np.asarray(out.obs["agent_pos"][0]), np.asarray(out.obs["agent_pos"][1]))
    ids = np.tile(np.arange(N, dtype=np.int32), (D, 1))
    roots = jax.pmap(jpool.snapshot, axis_name="device")(ids)
    actions = np.zeros((D, N), np.int32)
    child, child_ids = jax.pmap(jpool.simulate_batch, axis_name="device")(roots, actions)
    for d in range(D):  # replica d's children are pool d's roots stepped once
        obs, _, _, info = jpool._pools[d].simulate_batch(np.asarray(roots[d]), actions[d])
        np.testing.assert_array_equal(np.asarray(child.obs["agent_pos"][d]), obs["agent_pos"])
        jpool._pools[d].release_batch(np.asarray(info["state_id"], np.int32))
    jax.pmap(jpool.release_batch, axis_name="device")(child_ids)
    jax.pmap(jpool.release_batch, axis_name="device")(roots)
    jpool.close()


def test_pmap_over_another_axis_name_raises():
    jpool = JaxPool("Maze-v0", N, num_devices=D)
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(0), D))
    with pytest.raises(ValueError, match="axis_name='device'"):
        jax.pmap(jpool.reset, axis_name="batch")(keys)
    jpool.close()


def test_jit_on_a_multi_device_pool_uses_pool_zero():
    jpool = JaxPool("Maze-v0", N, num_devices=D)
    key = jax.random.PRNGKey(3)
    out = jax.jit(jpool.reset)(key)
    _, (obs, *_) = _reference("Maze-v0", key)
    np.testing.assert_array_equal(np.asarray(out.obs["agent_pos"]), obs["agent_pos"])
    jpool.close()
