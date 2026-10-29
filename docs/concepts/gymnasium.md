# gymnasium

`rustpool.envpool_api.gymnasium.RustPoolGymnasiumWrapper` presents a pool as a
[gymnasium](https://gymnasium.farama.org/) `VectorEnv`, for code written against that API.
It needs the `gymnasium` extra: `pip install "rustpool[gymnasium]"`.

```python
import numpy as np
from rustpool.envpool_api.gymnasium import RustPoolGymnasiumWrapper

env = RustPoolGymnasiumWrapper("Maze-v0", num_envs=4, width=5, height=5, seed=0)
obs, info = env.reset(seed=42)                      # env i is seeded with 42 + i
print(env.single_action_space, env.action_space)    # Discrete(4) MultiDiscrete([4 4 4 4])

for _ in range(10):
    actions = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(actions)
    assert env.observation_space.contains(obs)
env.close()
```

The keyword arguments go to `rustpool.make`, so environment options (`width` above) and pool
options (`seed`, `num_threads`, `dispatch`) both work.

## Behaviour

- **Synchronous.** The pool runs with `batch_size == num_envs`; every `step` steps every
  environment.
- **Auto-reset on the next step.** The wrapper uses the pool's `autoreset="next_step"`, which
  is gymnasium's `AutoresetMode.NEXT_STEP` (declared in `env.metadata`): the step that ends an
  episode returns its terminal observation, and the following step resets that environment,
  ignores its action and returns reward `0`. See [Episodes and auto-reset](autoreset.md).
- **`truncated` is always `False`.** Environments report a single `done` flag, so an episode
  cut by its step limit comes back as `terminated=True`.
- **Spaces.** Observations are a `Dict` of `Box` spaces built from the pool's
  `observation_spec()`: unbounded for floats, `[0, 1]` for boolean masks. `info` carries the
  pool's `env_id` and `action_mask` arrays.
