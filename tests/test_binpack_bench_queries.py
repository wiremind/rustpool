"""BinPack-v0 bench-only queries: ``query_state(env_id, "bench:<kind>[:reps]")`` time the flat
mask, the factored chain and a placement on a clone of the frozen state, without advancing it.
An internal, unstable benchmarking hook (see src/envs/binpack/bench.rs)."""

import numpy as np

import rustpool


def test_bench_queries_do_not_advance_the_state():
    pool = rustpool.make("BinPack-v0", num_envs=1, batch_size=1, seed=4, num_containers=2,
                         max_items=6, max_ems=8, factored=True, emit_flat_mask=True)
    try:
        pool.async_reset()
        obs, _, _, info = pool.recv()
        flat = pool.query_state(0, "bench:flat:2")
        chain = pool.query_state(0, "bench:chain:2")
        place = pool.query_state(0, "bench:place:2")
        assert pool.query_state(0, "bench:unknown") is None
        for key in ("bench:flat:0", "bench:flat:1001", "bench:flat:2:extra"):  # 1..=1000 reps
            assert pool.query_state(0, key) is None
        assert flat["reps"] == 2 and len(flat["ns_all"]) == 2 and flat["calls"] > 0
        assert flat["valid"] == int(info["action_mask"].sum())
        assert chain["factors"] == ["item", "container", "ems"] and chain["consistent"]
        assert chain["calls"] == sum(chain["stage_calls"])
        assert place["placed"] and info["action_mask"][0, place["action"]]
        # Same state afterwards: the first valid action is still valid and the obs unchanged.
        pool.send(np.array([place["action"]], np.int32), np.zeros(1, np.int32))
        obs2, rew, _, _ = pool.recv()
        assert rew[0] > 0
        assert obs2["placed_mask"].sum() == obs["placed_mask"].sum() + 1
    finally:
        pool.close()


def test_bench_query_before_the_first_reset_answers_none():
    with rustpool.make("BinPack-v0", num_envs=1, batch_size=1, max_items=4, max_ems=8,
                       factored=True, emit_flat_mask=True) as pool:
        assert pool.query_state(0, "bench:flat:1") is None  # it used to kill the worker
        pool.async_reset()
        pool.recv()
        assert pool.query_state(0, "bench:flat:1")["reps"] == 1
