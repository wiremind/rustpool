"""Test explicit seeding functionality for deterministic environment resets."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from rustpool.envpool_api.gymnasium import RustPoolGymnasiumWrapper  # noqa: E402


@pytest.mark.parametrize("task_id", ["BinPack-v0", "Maze-v0"])
def test_explicit_seeding_determinism(task_id):
    """Test that explicit seeding produces identical observations across resets."""
    num_envs = 4
    seed = 123

    # Two pools built with different seeds: only the explicit reset seed can make them agree.
    env1 = RustPoolGymnasiumWrapper(num_envs=num_envs, task_id=task_id, seed=0)
    env2 = RustPoolGymnasiumWrapper(num_envs=num_envs, task_id=task_id, seed=999)

    # Reset both with the same explicit seed
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)

    # Observations should be identical
    for key in obs1:
        assert np.allclose(obs1[key], obs2[key], rtol=1e-5, atol=1e-8), \
            f"Mismatch in observation key '{key}' for {task_id}"

    env1.close()
    env2.close()


@pytest.mark.parametrize("task_id", ["BinPack-v0", "Maze-v0"])
def test_explicit_seeding_reproducibility(task_id):
    """Test that resetting with the same seed twice produces identical results."""
    num_envs = 4
    seed = 456

    env = RustPoolGymnasiumWrapper(num_envs=num_envs, task_id=task_id, seed=0)

    # Reset twice with the same seed
    obs1, _ = env.reset(seed=seed)
    obs2, _ = env.reset(seed=seed)

    # Observations should be identical
    for key in obs1:
        assert np.allclose(obs1[key], obs2[key], rtol=1e-5, atol=1e-8), \
            f"Mismatch in observation key '{key}' for {task_id} on repeated reset"

    env.close()


@pytest.mark.parametrize("task_id", ["BinPack-v0", "Maze-v0"])
def test_different_seeds_produce_different_observations(task_id):
    """Test that different seeds produce different observations."""
    num_envs = 4

    env = RustPoolGymnasiumWrapper(num_envs=num_envs, task_id=task_id, seed=0)

    # Reset with different seeds
    obs1, _ = env.reset(seed=100)
    obs2, _ = env.reset(seed=200)

    # At least one observation key should differ
    any_different = False
    for key in obs1:
        if not np.allclose(obs1[key], obs2[key], rtol=1e-5, atol=1e-8):
            any_different = True
            break

    assert any_different, f"Different seeds produced identical observations for {task_id}"

    env.close()


@pytest.mark.parametrize("task_id", ["BinPack-v0", "Maze-v0"])
def test_unseeded_reset_still_works(task_id):
    """Test that reset without explicit seed still works (None case)."""
    num_envs = 4

    env = RustPoolGymnasiumWrapper(num_envs=num_envs, task_id=task_id, seed=42)

    # Reset without explicit seed (None)
    obs1, _ = env.reset()
    obs2, _ = env.reset()

    # Observations should differ (continuous RNG state)
    any_different = False
    for key in obs1:
        if not np.allclose(obs1[key], obs2[key], rtol=1e-5, atol=1e-8):
            any_different = True
            break

    assert any_different, f"Unseeded resets produced identical observations for {task_id}"

    env.close()


@pytest.mark.parametrize("task_id", ["BinPack-v0", "Maze-v0"])
def test_seed_distribution_per_environment(task_id):
    """reset(seed=s) gives env i the seed s + i: env i's first observation equals env 0's
    after reset(seed=s + i)."""
    num_envs = 4
    base_seed = 777
    env = RustPoolGymnasiumWrapper(num_envs=num_envs, task_id=task_id, seed=0)
    obs, _ = env.reset(seed=base_seed)
    for i in range(num_envs):
        shifted, _ = env.reset(seed=base_seed + i)
        for key in obs:
            np.testing.assert_array_equal(obs[key][i], shifted[key][0], err_msg=f"{key}, env {i}")
    assert any(not np.array_equal(obs[k][0], obs[k][1]) for k in obs), "envs 0 and 1 got one seed"
    env.close()


def test_sequence_and_numpy_integer_seeds():
    env = RustPoolGymnasiumWrapper("Maze-v0", num_envs=2)
    obs, info = env.reset(seed=[5, 9])
    np.testing.assert_array_equal(obs["agent_pos"][0], env.reset(seed=5)[0]["agent_pos"][0])
    np.testing.assert_array_equal(obs["agent_pos"][1], env.reset(seed=8)[0]["agent_pos"][1])
    for key, value in env.reset(seed=np.int64(5))[0].items():
        np.testing.assert_array_equal(value, env.reset(seed=5)[0][key])
    np.testing.assert_array_equal(info["env_id"], np.arange(2))
    _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert not truncated.any()
    env.close()
