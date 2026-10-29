"""rustpool throughput microbenchmarks.

Two metrics are reported **separately** (never conflated); see benchmarks/README.md:

1. **Engine overhead** — a near-trivial env (``Maze-v0``) stepped with random actions, isolating
   the cost of ``send``/``recv`` + worker threading + NumPy batching from env compute. Reported
   for the synchronous (``batch_size == num_envs``) and asynchronous (``batch_size < num_envs``)
   regimes.
2. **End-to-end** — steps/sec on each registered env with random valid actions.

Throughput is ``(num_envs * steps) / wall_time``, summarized as mean ± std over ``--repeats``
runs (default 50) with fixed seeds. Pass ``--json out.json`` to record reference numbers.

Run:
    python benchmarks/bench_engine.py
    python benchmarks/bench_engine.py --repeats 50 --json benchmarks/reference.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

import rustpool


def _random_valid_actions(info, num_actions, rng):
    """Sample a valid action per env from info['action_mask'] (uniform fallback)."""
    mask = info.get("action_mask")
    if mask is None:
        return rng.integers(0, num_actions, size=len(info["env_id"]), dtype=np.int32)
    actions = np.empty(mask.shape[0], dtype=np.int32)
    for i, row in enumerate(mask):
        valid = np.flatnonzero(row)
        actions[i] = rng.choice(valid) if valid.size else 0
    return actions


def _measure(task_id, num_envs, batch_size, steps, seed, **kwargs):
    """One timed run; returns steps/sec for `steps` recv cycles."""
    pool = rustpool.make(task_id, num_envs=num_envs, batch_size=batch_size, seed=seed, **kwargs)
    num_actions = int(pool.action_spec()[0])
    rng = np.random.default_rng(seed)
    try:
        pool.async_reset()
        # Drain the initial batch(es) and prime env_id bookkeeping.
        obs, _, _, info = pool.recv()
        t0 = time.perf_counter()
        for _ in range(steps):
            actions = _random_valid_actions(info, num_actions, rng)
            pool.send(actions, np.asarray(info["env_id"], dtype=np.int32))
            obs, _, _, info = pool.recv()
        dt = time.perf_counter() - t0
    finally:
        pool.close()
    # Each cycle advances `batch_size` envs.
    return (steps * batch_size) / dt


def bench(task_id, num_envs, batch_size, steps, repeats, **kwargs):
    samples = [
        _measure(task_id, num_envs, batch_size, steps, seed=s, **kwargs)
        for s in range(repeats)
    ]
    mean = statistics.mean(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    p95 = sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]
    return {"mean": mean, "std": std, "p95": p95, "n": repeats}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--json", default=None)
    cfg = p.parse_args(argv)

    import os

    hw = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "rustpool_version": rustpool.__version__,
        "num_envs": cfg.num_envs,
        "steps_per_run": cfg.steps,
        "repeats": cfg.repeats,
    }
    print(f"rustpool {hw['rustpool_version']} | {hw['platform']} | {hw['cpu_count']} CPUs")
    print(f"num_envs={cfg.num_envs} steps/run={cfg.steps} repeats={cfg.repeats}\n")

    results = {}
    rows = [
        ("engine sync  (Maze, bs=num_envs)", "Maze-v0", cfg.num_envs, {}),
        ("engine async (Maze, bs=num_envs/4)", "Maze-v0", cfg.num_envs // 4, {}),
        ("end-to-end  (BinPack-v0)", "BinPack-v0", cfg.num_envs, {}),
    ]
    print(f"{'metric':36s} {'steps/s (mean)':>16s} {'± std':>12s} {'p95':>12s}")
    print("-" * 78)
    for label, task, bs, kw in rows:
        r = bench(task, cfg.num_envs, bs, cfg.steps, cfg.repeats, **kw)
        results[label] = r
        print(f"{label:36s} {r['mean']:>16,.0f} {r['std']:>12,.0f} {r['p95']:>12,.0f}")

    if cfg.json:
        with open(cfg.json, "w") as f:
            json.dump({"hardware": hw, "results": results}, f, indent=2)
        print(f"\nwrote {cfg.json}")


if __name__ == "__main__":
    main()
