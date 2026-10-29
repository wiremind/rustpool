# Training with JAX and PyTorch

The repository's `examples/` folder holds reference implementations of PPO and AlphaZero that
drive rustpool. Each script imports only rustpool, its framework and the shared helper module
next to it (`networks.py` for JAX, `torch_common.py` for PyTorch): copy the script and its helper
into your project to start from there.

```bash
pip install "rustpool[examples]"   # jax, flax, optax, mctx and torch
```

| File | What it shows |
| --- | --- |
| `ppo_jax.py` | PPO in one jitted (or pmapped) JAX program: a `lax.scan` rollout over `JaxPool`, GAE, clipped loss, action masking. One or several GPUs. |
| `alphazero_jax.py` | AlphaZero with `mctx`, whose `recurrent_fn` is `simulate_batch` on the real environment: the search's embedding is a registry handle. `--factored` searches BinPack without the full mask. |
| `ppo_torch.py` | PPO in PyTorch (CleanRL style) with masked categoricals, over `TorchEnvPool`. |
| `alphazero_torch.py` | AlphaZero in PyTorch with a hand-written, array-batched MCTS over `snapshot` / `simulate_batch` / `release_batch`. |
| `factored_ppo_jax.py`, `factored_ppo_torch.py` | Autoregressive PPO on BinPack's factored action space: one factor at a time under conditional masks. |
| `networks.py`, `torch_common.py` | The shared networks and helpers: actor-critics, masked categoricals, the factored sampler (`factored_sample`), GAE. |

`--task` takes `Maze-v0` or `BinPack-v0`: the examples learn under an action mask, and
`CartPole-v1` emits none.

```bash
python examples/ppo_jax.py --task Maze-v0                 # all local GPUs
python examples/ppo_jax.py --task BinPack-v0              # dense-reward 3D packing
python examples/alphazero_jax.py --num-simulations 32
python examples/alphazero_jax.py --factored               # BinPack, no full mask
python examples/factored_ppo_jax.py
python examples/ppo_torch.py
python examples/alphazero_torch.py --task BinPack-v0
# quick checks on CPU, in seconds:
python examples/ppo_jax.py --smoke
python examples/alphazero_jax.py --smoke
```

## How the PPO examples use the pool

The rollout is a `lax.scan` over `pool.step` inside the jitted update, so the whole iteration
(rollout, GAE, epochs of minibatch updates) is one XLA program, with one host trip per
environment step. Under `pmap`, each device steps its own pool (`num_devices` pools,
`--num-envs` environments each) and the gradients are averaged with `pmean`.

`--num-devices 0` uses every local device and `--num-devices 1` forces one. You can rehearse the
multi-device path on CPU:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python examples/ppo_jax.py --num-devices 2 --num-envs 16 --num-steps 8 --total-timesteps 50000
```

## How AlphaZero uses the registry

At each real step the search snapshots the live environments (the roots), then `mctx` expands
nodes by calling `simulate_batch` on their handles; every expansion runs the real environment
code. After the search picks an action, the whole tree is released and the pool takes the real
step. See [State forks for search](../concepts/mcts.md) for the registry's rules, and
[the JAX bridge](../concepts/jax-bridge.md#ordered-and-unordered-calls) for why `simulate_batch`
can run inside `mctx`'s loop.

## Reproducibility

Every script takes `--seed`. On `Maze-v0` the mean episodic return rises toward 1 as the policy
learns to walk to the goal; on `BinPack-v0` the dense return is the fraction of the container
filled. When you report numbers, record the hardware, the device count, the JAX or PyTorch
version and `rustpool.__version__`.
