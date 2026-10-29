# rustpool examples

Reference implementations of **PPO** and **AlphaZero** driving the rustpool engine, in JAX and
in PyTorch. They import only rustpool, the relevant framework and the shared helper module next
to them (`networks.py` / `torch_common.py`).

```bash
pip install "rustpool[examples]"   # jax, jaxlib, flax, optax, mctx, torch
```

| File | What it shows |
| --- | --- |
| [`ppo_jax.py`](ppo_jax.py) | PPO in pure JAX: jitted/pmapped rollout (`lax.scan`) over `JaxPool`, GAE, clipped loss, value clipping, entropy, action masking. Single- or multi-GPU. |
| [`alphazero_jax.py`](alphazero_jax.py) | AlphaZero via `mctx` with the **real Rust simulator** as the `recurrent_fn` — `snapshot`/`simulate_batch`/`release_batch`. rustpool's headline capability. |
| [`ppo_torch.py`](ppo_torch.py) | PPO in PyTorch (CleanRL-style) with `CategoricalMasked` invalid-action masking, over `TorchEnvPool`. |
| [`alphazero_torch.py`](alphazero_torch.py) | AlphaZero in PyTorch: a hand-written **array-batched MCTS** driving `snapshot`/`simulate_batch`/`release_batch` (no mctx in torch). |
| [`networks.py`](networks.py) | Shared Flax Linen actor-critic + masked-categorical helpers (no distrax) for the **JAX** examples. |
| [`torch_common.py`](torch_common.py) | Shared encoder, masked-categorical helpers, init, and GAE for the **PyTorch** examples (the torch counterpart of `networks.py`). |
| [`factored_ppo_jax.py`](factored_ppo_jax.py) / [`factored_ppo_torch.py`](factored_ppo_torch.py) | **Factored** (autoregressive) PPO on BinPack's `item → ems` action space — samples a factor at a time under per-factor conditional masks (`query_mask`). |
| [`rust/external_env.rs`](rust/external_env.rs) | An environment in **its own crate**: the `RlEnv` impl, a `make()` with rustpool's pool options (`PoolOptions::parse` + `RlPool::new_with_options`) and the `#[pymodule]`. Built by `cargo build --example external_env`; the docs' "Bring your own crate" guide walks through it. |

### Factored action spaces

`BinPack-v0` with `factored=True` exposes a conditional action space (`action_factorization()` +
`query_mask`): `item → ems`, growing to `item → orientation → container → ems` with
`num_containers` / `orientations` > 1.

```bash
python examples/factored_ppo_jax.py            # PPO, JAX  (autoregressive sampling)
python examples/factored_ppo_torch.py          # PPO, PyTorch
python examples/alphazero_jax.py   --factored  # AlphaZero, JAX  (no full mask, sampled actions per node)
python examples/alphazero_jax.py   --task BinPack-v0  # AlphaZero, full mask (factored head -> flat prior)
python examples/alphazero_torch.py --task BinPack-v0  # same, PyTorch
python benchmarks/bench_factored.py            # flat vs factored policy-step (wins for large |A|)
```

PPO samples autoregressively — `sum(widths)` candidate logits/step instead of the flat
`prod(widths)` masked categorical. That is a net wall-clock win only once the flat head dominates
(large combinatorial action spaces); for small spaces flat is cheaper and factored adds
`query_mask` round-trips. AlphaZero `--task BinPack-v0` searches the flat masked action space with
a parameter-efficient factored head; `--factored` never builds the full mask and expands each node
into `--num-candidates` full actions sampled from the factored policy. See [Factored action masks](../docs/concepts/factored-actions.md) for the measured
table and the honest trade-offs.

## How rustpool plugs into JAX

The training loops are one XLA program; only env transitions hop to the host via
`jax.experimental.io_callback`, wrapped by [`rustpool.jax_api.JaxPool`](../rustpool/jax_api.py):

```python
import jax, rustpool.jax_api as rj
pool = rj.JaxPool("Maze-v0", num_envs=256)          # single device
out  = jax.jit(pool.reset)(jax.random.key(0))
out  = jax.jit(pool.step)(actions)                  # int32[num_envs]
```

For AlphaZero the same object exposes the search substrate, used as an mctx `recurrent_fn`:

```python
state_ids                 = pool.snapshot(env_ids)            # freeze live envs
env_out, child_state_ids  = pool.simulate_batch(state_ids, a) # fork-and-step (no mutation)
pool.release_batch(child_state_ids)                            # free tree nodes
```

## Single- and multi-GPU

`JaxPool` holds one independent pool **per device** and routes each `io_callback` with
`jax.lax.axis_index("device")`, so the examples run
unchanged under `jax.pmap`:

```bash
python ppo_jax.py --num-devices 1          # force a single device
python ppo_jax.py --num-devices 0          # 0 = all local GPUs (Anakin: pmean'd grads)
python alphazero_jax.py --num-devices 0
```

`--num-envs` is **per device**; the global batch is `num_devices × num_envs × num_steps`.
You can rehearse the multi-device code path on CPU:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python ppo_jax.py --num-devices 2 --num-envs 16 --num-steps 8 --total-timesteps 50000
```

## PyTorch performance notes

The torch examples apply the standard modern speedups by default: host-sync elimination
(on-device metric accumulation, on-device minibatch permutation — no `.item()` in the hot loop),
`torch.optim.Adam(fused=True)`, and TF32 matmuls (`set_float32_matmul_precision("high")`). A
`--compile` flag additionally `torch.compile`s the MLPs with CUDA graphs (`reduce-overhead`).

The sync/fused/TF32 changes give PPO a modest speedup; **`torch.compile` does *not* help these
examples** — they are host-round-trip bound (the Rust env
is stepped once per timestep) on a tiny MLP, so there is little compute to fuse and the CUDA-graph
warmup makes end-to-end *slower*. `--compile` is kept as an opt-in because it pays off in the
compute-bound regime (bigger networks, or env-on-device); it does not here. AlphaZero is even more
host-bound (`simulate_batch` per simulation), so neither compile nor sync-removal moves it.

## Reproducibility

Every script takes `--seed` (default 0) and seeds the env pool, network init, and rollout RNG.
A quick correctness check (a few seconds each, CPU):

```bash
python ppo_jax.py --smoke
python alphazero_jax.py --smoke
python ppo_torch.py --smoke
python alphazero_torch.py --smoke
```

### Expected behaviour (`Maze-v0`, the bundled navigation task)

The agent starts at a random cell and must reach the fixed corner goal `(W-1, H-1)`; reward is
a sparse `+1` at the goal, and invalid moves are masked out. A healthy run shows the **mean
episodic return rise from ~0 toward ~1.0** and policy entropy fall as the policy commits to
walking toward the goal. Example (10×10 maze, single RTX-class GPU, seed 0):

```bash
python ppo_jax.py --task Maze-v0 --maze-size 10 --total-timesteps 2_000_000
# return climbs to ~1.0 within a few hundred updates
```

Wall-clock and exact returns depend on hardware, device count, and JAX/driver versions — pin
those (and the rustpool version, `python -c "import rustpool; print(rustpool.__version__)"`)
when reporting numbers.

> **Note on GPU memory.** The examples set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so they grow
> memory on demand and coexist with other processes. Set `CUDA_VISIBLE_DEVICES` to pick GPUs.
