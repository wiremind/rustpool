"""Maze-v0 dynamics: moves, masks, the goal, the step limit and invalid actions.

Actions are probed on snapshots of one frozen state, so each check sees the same start cell.
A reset may place the agent on the goal (the next step then pays 1 whatever the action), so the
checks below use start cells away from it.
"""

import numpy as np
import pytest

import rustpool

UP, DOWN, LEFT, RIGHT = range(4)
MOVES = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}


def _pool(**kwargs):
    pool = rustpool.make("Maze-v0", num_envs=1, batch_size=1, autoreset="next_step", **kwargs)
    pool.async_reset()
    obs, _, _, info = pool.recv()
    return pool, obs["agent_pos"][0].astype(int), info["action_mask"][0]


def _start(seed, **kwargs):
    pool, pos, _ = _pool(seed=seed, **kwargs)
    pool.close()
    return pos


def _fork(pool, action):
    root = np.asarray(pool.snapshot(np.zeros(1, np.int32)), np.int32)
    obs, reward, done, info = pool.simulate_batch(root, np.array([action], np.int32))
    pool.release_batch(np.concatenate([root, np.asarray(info["state_id"], np.int32)]))
    return obs["agent_pos"][0].astype(int), float(reward[0]), bool(done[0]), info


def test_valid_moves_shift_the_agent_and_the_mask_matches_the_walls():
    for seed in range(20):
        pool, pos, mask = _pool(width=5, height=5, seed=seed)
        x, y = pos
        np.testing.assert_array_equal(mask, [y > 0, y < 4, x > 0, x < 4])
        for action, (dx, dy) in MOVES.items():
            if mask[action]:
                new, reward, done, _ = _fork(pool, action)
                assert tuple(new) == (x + dx, y + dy)
                assert 0 <= new[0] < 5 and 0 <= new[1] < 5
                assert done == (reward == 1.0)  # only the goal ends a valid step this early
        pool.close()


def test_an_invalid_move_ends_the_episode_without_reward_and_without_moving():
    for seed in range(20):
        pool, pos, mask = _pool(width=5, height=5, seed=seed)
        blocked = np.flatnonzero(~mask)
        if blocked.size and tuple(pos) != (4, 4):
            new, reward, done, _ = _fork(pool, int(blocked[0]))
            assert done and reward == 0.0 and tuple(new) == tuple(pos)
            pool.close()
            return
        pool.close()
    pytest.fail("no start cell on a wall in 20 seeds")


def test_reaching_the_goal_pays_one_and_ends_the_episode():
    # A 4x1 corridor: the goal is the right end (3, 0); walk right until it is reached.
    seed = next(s for s in range(50) if _start(s, width=4, height=1)[0] < 3)
    pool, pos, _ = _pool(width=4, height=1, seed=seed, max_episode_steps=50)
    rewards = []
    done = False
    while not done:
        pool.send(np.array([RIGHT], np.int32), np.zeros(1, np.int32))
        obs, reward, d, _ = pool.recv()
        rewards.append(float(reward[0]))
        done = bool(d[0])
    assert tuple(obs["agent_pos"][0].astype(int)) == (3, 0)
    assert rewards[-1] == 1.0 and all(r == 0.0 for r in rewards[:-1])
    assert len(rewards) == 3 - pos[0]
    pool.close()


def test_the_step_limit_ends_the_episode_on_exactly_that_step():
    pool, _, mask = _pool(width=7, height=7, seed=1, max_episode_steps=3)
    dones = []
    for t in range(3):
        # Bounce between two cells away from the goal so only the limit can end it.
        action = UP if t % 2 == 0 else DOWN
        if not mask[UP]:
            action = DOWN if t % 2 == 0 else UP
        pool.send(np.array([action], np.int32), np.zeros(1, np.int32))
        _, reward, done, info = pool.recv()
        mask = info["action_mask"][0]
        dones.append(bool(done[0]))
        assert reward[0] == 0.0
    assert dones == [False, False, True]
    pool.close()
