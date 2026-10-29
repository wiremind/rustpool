# rustpool

rustpool runs many reinforcement-learning environments on native Rust worker threads and drives
them from Python through an asynchronous, [EnvPool](https://github.com/sail-sg/envpool)-style
`send` / `recv` API. The environments are written in Rust; the policy, the training loop and the
search live in Python (NumPy, JAX or PyTorch).

```python
import numpy as np
import rustpool

pool = rustpool.make("CartPole-v1", num_envs=8, batch_size=8)
pool.async_reset()
obs, reward, done, info = pool.recv()          # obs["obs"]: float32[8, 4]
pool.send(np.ones(8, np.int32), info["env_id"])
obs, reward, done, info = pool.recv()
pool.close()
```

## Three capabilities

1. **A parallel, asynchronous pool.** Each environment lives on a Rust worker thread, so stepping
   needs no GIL. `recv` can return as soon as part of the pool is ready, so policy inference
   overlaps with simulation. See [The pool, threads and dispatch](concepts/pool.md).
2. **State forks for search.** `snapshot`, `simulate_batch` and `release_batch` clone any
   environment state and step the copies in parallel without touching the original: the
   operation a batched tree search (MCTS, AlphaZero) needs, for a simulator that is not a JAX
   program. See [State forks for search](concepts/mcts.md).
3. **Factored action masks.** An environment with a large combinatorial action space can expose
   its mask one factor at a time (`query_mask`, `query_state_mask`), so a policy samples
   `item → position → …` without ever building the full mask. See
   [Factored action masks](concepts/factored-actions.md).

Around them: a [JAX bridge](concepts/jax-bridge.md) usable under `jit` and `pmap`, a
[PyTorch view](concepts/pytorch.md), two [auto-reset conventions](concepts/autoreset.md), and two
ways to add an environment: [inside rustpool](guides/env-in-rustpool.md) or
[in your own crate](guides/own-crate.md), with the same pool options either way.

## When to use it

| You have… | Consider |
| --- | --- |
| Standard benchmark environments (Atari, MuJoCo, classic control) and want the fastest stepping | **EnvPool**: a large bundled suite, and `env.xla()` steps inside XLA with no host round trip. |
| An environment written in JAX | **A pure-JAX environment** (Brax, Pgx, Jumanji): it runs on the accelerator, and `mctx` can search it directly. |
| Your own simulator, not a JAX program, possibly with a large structured action space, possibly searched by MCTS | **rustpool**: write the environment in Rust once and get the parallel pool, state forks and factored masks, from NumPy, JAX or PyTorch. |

What rustpool does not do: it ships three environments, not a suite, and its JAX bridge crosses
to the host once per call (`io_callback`), which costs more than EnvPool's `env.xla()` for a
cheap environment.

|  | EnvPool | rustpool |
| --- | --- | --- |
| Asynchronous batched `recv` | yes | yes |
| Auto-reset in the worker | yes (next step) | yes (same step by default, or next step) |
| Clone and fork states for search | no | yes |
| Factored, conditional action masks | no | yes |
| JAX | pure `env.xla()` | `io_callback` bridge |
| Bundled environments | many | three; you write yours in Rust |

Start with the [Quickstart](quickstart.md).
