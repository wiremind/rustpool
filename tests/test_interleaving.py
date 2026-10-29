"""snapshot / query_state / query_mask called between the recv calls of an async round must not
disturb it: their replies are buffered apart from the step results, which come back exactly as
in a pool that made no extra calls."""

import itertools

import numpy as np
import pytest

import rustpool

N, B, THREADS, ROUNDS = 12, 3, 3, 20
TASKS = {
    "CartPole-v1": ({}, "state"),
    "BinPack-v0": ({"factored": True, "max_items": 6, "max_ems": 8}, "factored_dims"),
}


def _rollout(task, dispatch, autoreset, interleave):
    kwargs, key = TASKS[task]
    pool = rustpool.make(task, num_envs=N, batch_size=B, num_threads=THREADS, seed=7,
                         dispatch=dispatch, autoreset=autoreset, **kwargs)
    width = pool.action_spec()[0]
    per_env = {e: [] for e in range(N)}

    def collect():
        obs, reward, done, info = pool.recv()
        ids = info["env_id"]
        assert len(ids) == B and np.all(np.diff(ids) > 0)  # one batch, sorted by env id
        for row, e in enumerate(ids):
            per_env[int(e)].append((
                {k: v[row].copy() for k, v in obs.items()}, float(reward[row]), bool(done[row])))
        return info

    pool.async_reset()
    for rnd in range(ROUNDS + 1):
        for batch in range(N // B):
            info = collect()
            if interleave and batch == 0:
                roots = pool.snapshot(np.array([0, 5, 11], np.int32))
                pool.query_state(int(info["env_id"][0]), key)
                if task == "BinPack-v0":
                    pool.query_mask(np.full((2, 1), -1, np.int32), np.array([1, 1], np.int32))
                pool.release_batch(np.asarray(roots, np.int32))
        if rnd < ROUNDS:
            # Fixed per-env actions that change every 4 rounds: CartPole falls over and BinPack
            # hits invalid actions, so both auto-reset during the run.
            actions = np.array([(e * 7 + rnd // 4) % width for e in range(N)], np.int32)
            pool.send(actions, np.arange(N, dtype=np.int32))
    pool.close()
    return per_env


@pytest.mark.parametrize("task", sorted(TASKS))
@pytest.mark.parametrize("dispatch, autoreset",
                         list(itertools.product(["per_env", "per_worker"], ["same_step", "next_step"])))
def test_queries_between_recvs_do_not_disturb_the_round(task, dispatch, autoreset):
    plain = _rollout(task, dispatch, autoreset, interleave=False)
    busy = _rollout(task, dispatch, autoreset, interleave=True)
    for e in range(N):
        assert len(plain[e]) == len(busy[e]) == ROUNDS + 1
        for (o1, r1, d1), (o2, r2, d2) in zip(plain[e], busy[e], strict=True):
            assert (r1, d1) == (r2, d2)
            for k in o1:
                np.testing.assert_array_equal(o1[k], o2[k])
