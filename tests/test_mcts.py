import numpy as np
import pytest

import rustpool


def test_stateless_simulation_basic():
    """Verify that we can snapshot an environment and simulate from that snapshot."""
    num_envs = 4
    env_pool = rustpool.RlPool(num_envs=num_envs, batch_size=num_envs, task_id="Maze-v0")

    try:
        seeds = np.array([42, 43, 44, 45], dtype=np.int64)
        env_pool.async_reset(seeds=seeds)
        obs, reward, done, info = env_pool.recv()
        env_ids = info["env_id"]

        # 1. Take snapshot of all environments
        state_ids = env_pool.snapshot(env_ids)
        assert len(state_ids) == num_envs
        assert all(isinstance(sid, int) for sid in state_ids)

        # 2. Simulate one step for each state
        # In the branching model, simulate_batch returns NEW state_ids.
        actions = np.zeros(num_envs, dtype=np.int32)  # action 0 is Up in Maze
        obs_sim, reward_sim, done_sim, info_sim = env_pool.simulate_batch(
            np.array(state_ids, dtype=np.int32),
            actions
        )

        assert len(reward_sim) == num_envs
        assert "state_id" in info_sim
        new_state_ids = info_sim["state_id"]
        assert set(new_state_ids.tolist()).isdisjoint(state_ids)  # fresh ids

        # 3. Verify that the simulation didn't affect the parent state IDs (Branching)
        # Re-simulating from the same parent state_ids should yield the exact same results.
        obs_sim2, reward_sim2, done_sim2, info_sim2 = env_pool.simulate_batch(
            np.array(state_ids, dtype=np.int32),
            actions
        )

        for k in obs_sim.keys():
            assert np.array_equal(obs_sim[k], obs_sim2[k])

        assert np.array_equal(reward_sim, reward_sim2)
        assert np.array_equal(done_sim, done_sim2)

        # 4. Release all states
        env_pool.release_batch(np.array(state_ids, dtype=np.int32))
        env_pool.release_batch(np.array(new_state_ids, dtype=np.int32))
        env_pool.release_batch(np.array(info_sim2["state_id"], dtype=np.int32))

        # 5. Verify invalid state_id raises error
        with pytest.raises(ValueError, match="invalid state_id"):
            env_pool.simulate_batch(np.array([99999], dtype=np.int32), np.array([0], dtype=np.int32))

    finally:
        env_pool.close()

def test_stateless_simulation_branching_mcts():
    """Verify that multiple simulations from the same parent state don't interfere."""
    env_pool = rustpool.RlPool(num_envs=1, batch_size=1, task_id="Maze-v0")

    try:
        env_pool.async_reset(seeds=np.array([42], dtype=np.int64))
        _, _, _, info = env_pool.recv()
        env_id = info["env_id"][0]

        # Take snapshot
        parent_state_id = env_pool.snapshot(np.array([env_id], dtype=np.int32))[0]

        # Simulation 1 (Branch A): Move in one direction
        obs1, _, _, info1 = env_pool.simulate_batch(
            np.array([parent_state_id], dtype=np.int32),
            np.array([0], dtype=np.int32)
        )
        child1_id = info1["state_id"][0]

        # Simulation 2 (Branch B): Move in another direction FROM THE SAME PARENT
        obs2, _, _, info2 = env_pool.simulate_batch(
            np.array([parent_state_id], dtype=np.int32),
            np.array([1], dtype=np.int32)
        )
        child2_id = info2["state_id"][0]

        # Verify all IDs are distinct
        assert child1_id != parent_state_id
        assert child2_id != parent_state_id
        assert child1_id != child2_id

        # Up and down never both hit a wall (height > 1), so the two children differ.
        assert not np.array_equal(obs1["agent_pos"], obs2["agent_pos"])

    finally:
        env_pool.close()


def test_snapshot_repeated_env_id_gives_independent_handles():
    with rustpool.make("Maze-v0", num_envs=2, batch_size=2, seed=0) as pool:
        pool.async_reset()
        pool.recv()
        ids = pool.snapshot(np.array([0, 1, 0], dtype=np.int32))
        assert len(set(ids)) == 3
        obs, _, _, child = pool.simulate_batch(np.asarray(ids, np.int32), np.zeros(3, np.int32))
        np.testing.assert_array_equal(obs["agent_pos"][0], obs["agent_pos"][2])
        pool.release_batch(np.asarray(child["state_id"], np.int32))
        pool.release_batch(np.asarray(ids, np.int32))


SIM_CASES = {
    "CartPole": ("CartPole-v1", {}),
    "Maze": ("Maze-v0", {"width": 4, "height": 4}),
    "BinPack-2x2": ("BinPack-v0", {"max_items": 6, "max_ems": 8, "num_containers": 2,
                                   "orientations": 2, "emit_flat_mask": True}),
    "BinPack-factored": ("BinPack-v0", {"max_items": 6, "max_ems": 8, "factored": True,
                                        "emit_flat_mask": True}),
}


@pytest.mark.parametrize("case", sorted(SIM_CASES))
def test_simulate_batch_equals_the_live_step(case):
    """Forking a snapshot with action a gives exactly what the live env returns for a, terminal
    steps included (next_step auto-reset, so the live terminal step shows the terminal obs)."""
    task, kwargs = SIM_CASES[case]
    n = 8
    rng = np.random.default_rng(0)
    ids = np.arange(n, dtype=np.int32)
    with rustpool.make(task, num_envs=n, batch_size=n, seed=3, autoreset="next_step",
                       **kwargs) as pool:
        width = pool.action_spec()[0]
        pool.async_reset()
        _, _, done, info = pool.recv()
        compared_terminal = 0
        for _ in range(40):
            mask = info.get("action_mask")
            actions = rng.integers(0, width, n).astype(np.int32)
            if mask is not None and mask.shape[1] == width:  # mostly valid actions, some not
                for e in range(n):
                    valid = np.flatnonzero(mask[e])
                    if valid.size and rng.random() < 0.9:
                        actions[e] = rng.choice(valid)
            pending = done.copy()  # rows whose next step is next_step's reset: not a real step
            roots = np.asarray(pool.snapshot(ids), np.int32)
            s_obs, s_rew, s_done, s_info = pool.simulate_batch(roots, actions)
            pool.send(actions, ids)
            obs, reward, done, info = pool.recv()
            live = ~pending
            np.testing.assert_array_equal(s_rew[live], reward[live])
            np.testing.assert_array_equal(s_done[live], done[live])
            for k in obs:
                np.testing.assert_array_equal(s_obs[k][live], obs[k][live])
            compared_terminal += int((done & live).sum())
            pool.release_batch(np.asarray(s_info["state_id"], np.int32))
            pool.release_batch(roots)
        assert compared_terminal > 0, "no terminal step was compared"


def test_release_batch_ignores_the_sentinel_and_unknown_ids():
    with rustpool.make("Maze-v0", num_envs=2, batch_size=2) as pool:
        pool.async_reset()
        pool.recv()
        roots = np.asarray(pool.snapshot(np.arange(2, dtype=np.int32)), np.int32)
        pool.release_batch(np.array([-1, 10**6], np.int32))
        pool.simulate_batch(roots, np.zeros(2, np.int32))  # still live
        with pytest.raises(ValueError, match="same length"):
            pool.simulate_batch(roots, np.zeros(1, np.int32))


def test_next_step_snapshot_between_terminal_and_reset_holds_the_terminal_state():
    """Under next_step, a snapshot taken after an episode's terminal step (before its reset)
    holds the terminal state; simulate_batch steps it as is (it never resets), while the live
    env's next step is the reset."""
    with rustpool.make("CartPole-v1", num_envs=1, batch_size=1, autoreset="next_step") as pool:
        ids = np.zeros(1, np.int32)
        pool.async_reset()
        pool.recv()
        for _ in range(500):
            pool.send(np.zeros(1, np.int32), ids)
            obs, _, done, _ = pool.recv()
            if done[0]:
                break
        assert done[0]
        np.testing.assert_allclose(pool.query_state(0, "state"), obs["obs"][0], rtol=1e-6)
        root = np.asarray(pool.snapshot(ids), np.int32)
        _, _, child_done, _ = pool.simulate_batch(root, np.zeros(1, np.int32))
        assert child_done[0]  # past the end: the pole stays down
        pool.send(np.zeros(1, np.int32), ids)
        _, reward, live_done, _ = pool.recv()
        assert reward[0] == 0.0 and not live_done[0]  # the live env reset instead
