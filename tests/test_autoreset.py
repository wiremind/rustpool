"""autoreset="same_step" (default) vs "next_step" (EnvPool's and gymnasium's convention)."""

import numpy as np
import pytest

import rustpool


def _drive(pool, steps, rng):
    """Step a 1-env pool with random actions; return (obs, reward, done) per call."""
    pool.async_reset()
    obs, _, _, info = pool.recv()
    out = [(obs["obs"].copy(), None, None)]
    for _ in range(steps):
        pool.send(rng.integers(0, 2, size=1).astype(np.int32), info["env_id"])
        obs, rew, done, info = pool.recv()
        out.append((obs["obs"].copy(), float(rew[0]), bool(done[0])))
    return out


def test_same_step_is_the_default():
    pool = rustpool.make("CartPole-v1", num_envs=2)
    assert pool.autoreset == "same_step"
    pool.close()


@pytest.mark.parametrize("dispatch", ["per_env", "per_worker"])
def test_next_step_spends_one_call_per_episode(dispatch):
    pool = rustpool.make("CartPole-v1", num_envs=1, batch_size=1, seed=3, num_threads=1,
                         autoreset="next_step", dispatch=dispatch)
    assert pool.autoreset == "next_step"
    calls = _drive(pool, 3000, np.random.default_rng(0))
    pool.close()
    resets = 0
    for (prev_obs, _, prev_done), (obs, rew, done) in zip(calls[:-1], calls[1:], strict=True):
        if prev_done:
            # The call after a terminal step is the reset: reward 0, not done, a fresh state.
            assert rew == 0.0 and not done
            assert np.all(np.abs(obs) <= 0.05) and not np.array_equal(obs, prev_obs)
            resets += 1
        else:
            assert rew == 1.0
    assert resets > 0


def test_next_step_returns_the_terminal_observation():
    """Same seed and actions: next_step's stream is same_step's with one reset call inserted
    after each episode, and its terminal observations are states same_step never returns."""
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    same = rustpool.make("CartPole-v1", num_envs=1, batch_size=1, seed=9, num_threads=1)
    nxt = rustpool.make("CartPole-v1", num_envs=1, batch_size=1, seed=9, num_threads=1,
                        autoreset="next_step")
    a = _drive(same, 600, rng_a)
    # next_step needs one extra call per episode; its reset call's action is ignored, so feed
    # it the same action stream with a dummy action at each reset call.
    nxt.async_reset()
    obs, _, _, info = nxt.recv()
    b, done, terminal = [obs["obs"].copy()], False, []
    while len(b) < len(a):
        if done:
            terminal.append(b[-1])
            nxt.send(np.zeros(1, np.int32), info["env_id"])  # the reset call
            obs, _, done, info = nxt.recv()
            b[-1] = obs["obs"].copy()  # same_step returned this obs on the terminal call
            continue
        nxt.send(rng_b.integers(0, 2, size=1).astype(np.int32), info["env_id"])
        obs, _, d, info = nxt.recv()
        done = bool(d[0])
        b.append(obs["obs"].copy())
    same.close()
    nxt.close()
    assert any(c[2] for c in a), "no episode ended: the test proves nothing"
    for t, (sa, sb) in enumerate(zip([c[0] for c in a], b, strict=True)):
        np.testing.assert_array_equal(sa, sb, err_msg=f"call {t}")
    # Random-action episodes end by termination well before the 500-step limit: every terminal
    # obs is out of bounds (|x| > 2.4 or |theta| > 12 degrees), which same_step never returns.
    assert terminal
    for o in terminal:
        assert abs(o[0, 0]) > 2.4 or abs(o[0, 2]) > 12 * 2 * np.pi / 360
    assert all(abs(c[0][0, 0]) <= 2.4 and abs(c[0][0, 2]) <= 12 * 2 * np.pi / 360 for c in a)


def test_async_reset_clears_a_pending_reset():
    pool = rustpool.make("CartPole-v1", num_envs=1, batch_size=1, seed=0, num_threads=1,
                         autoreset="next_step")
    pool.async_reset()
    _, _, _, info = pool.recv()
    done = np.zeros(1, bool)
    while not done[0]:
        pool.send(np.zeros(1, np.int32), info["env_id"])
        _, _, done, info = pool.recv()
    pool.async_reset()
    pool.recv()
    pool.send(np.zeros(1, np.int32), info["env_id"])
    _, rew, _, _ = pool.recv()
    assert rew[0] == 1.0  # a real step, not the stale pending reset
    pool.close()


def test_next_step_with_masked_env_resets_mask():
    """Maze emits a mask: the reset call returns the new episode's mask."""
    pool = rustpool.make("Maze-v0", num_envs=4, batch_size=4, seed=1, num_threads=2,
                         autoreset="next_step")
    pool.async_reset()
    _, _, _, info = pool.recv()
    pending = np.zeros(4, bool)
    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(500):
        pool.send(rng.integers(0, 4, size=4).astype(np.int32), info["env_id"])
        _, rew, done, info = pool.recv()
        assert info["action_mask"].shape == (4, 4) and info["action_mask"].any(axis=1).all()
        if pending.any():
            assert (rew[pending] == 0.0).all() and not done[pending].any()
            seen += int(pending.sum())
        pending = done.copy()
    pool.close()
    assert seen > 0


def test_autoreset_rejects_unknown_mode():
    with pytest.raises(ValueError, match="autoreset"):
        rustpool.make("CartPole-v1", num_envs=2, autoreset="later")
