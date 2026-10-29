# rustpool benchmarks

```bash
python benchmarks/bench_engine.py --repeats 50 --json benchmarks/reference.json
```

## Methodology

Two metrics are reported **separately and never conflated**, because a heavy environment masks
engine parallelism while a trivial one exaggerates it:

1. **Engine overhead** — `Maze-v0` (a near-trivial step) with random valid actions, isolating
   `send`/`recv` + worker threading + NumPy batching from env compute. Reported for the
   **synchronous** (`batch_size == num_envs`) and **asynchronous** (`batch_size < num_envs`)
   regimes; the async advantage is a function of per-env step-time variance, so the two regimes
   are reported as distinct conditions, not averaged.
2. **End-to-end** — steps/sec on a real env (`BinPack-v0`).

Throughput is `(num_envs × steps) / wall_time`. Each figure is **mean ± std and p95 over
`--repeats` runs** (default 50) with fixed seeds; hardware (CPU model, core count), `num_envs`,
`batch_size`, and the rustpool version are recorded alongside (`--json`).

The harness's action sampling does a small per-row Python loop over the action mask; that cost is
part of the *caller*, not the engine, and weighs more on wide-mask envs (BinPack) than on Maze —
so read the Maze rows as the engine-overhead figure.

## A fair comparison with EnvPool

`bench_engine.py` measures rustpool only. To compare against EnvPool **honestly**:

- Compare **engine overhead on an identical, trivial env**, not env-compute FPS on EnvPool's
  bundled C++ suite (that measures env compute — identical to prior work — not rustpool's
  contribution).
- Hold the regime equal (sync vs async) or report both.
- Report rustpool's *unique* capabilities (`snapshot`/`simulate_batch` search, factored masks)
  against the honest baseline (a Python `SubprocVecEnv` + `deepcopy`), **not** EnvPool — which
  cannot do them at all.

`reference.json` is one example output, from one development machine and an earlier version
(0.1.6); regenerate it on yours to compare.
