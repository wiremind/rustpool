# Factored action masks

A flat action of size `|A|` can be assembled one component at a time (for example
`item → container → position`), each component masked **conditionally** ("given the components
chosen so far, does ≥1 fully-valid completion exist?"). This is **not a new action space**: the
committed action is the same flat index, decoded and executed by the normal `step`. It is a new
way to *expose the mask* so a policy can sample autoregressively without materialising the full
flat mask (for large factored spaces this can be orders of magnitude fewer candidate evaluations).

The capability is **generic** — rustpool transports the spec and the masks without knowing what
the factors mean — and **opt-in** (defaulted on the `RlEnv` trait, so envs that don't use it are
unaffected).

## The API

- **`pool.action_factorization() -> dict | None`** — the env's chosen decomposition, or `None`
  for a flat action space. Carries `{name, factor_names, factor_widths, num_factors, pad_width,
  flat_action_size, conditional}`. `conditional=True` ⇒ masks depend on the prefix of
  already-chosen factors (autoregressive); `False` ⇒ independent per-factor masks.
- **`pool.query_mask(prefixes, env_ids) -> bool[B, pad_width]`** — the batched,
  **non-advancing** transport for the conditional masks. `prefixes` is `int32[B, K-1]` (the last
  factor has no successor; `-1` = unset). Mirrors `send`/`recv` internally (fans out to the worker
  shards) but steps nothing. Each `True` is a genuine ∃-witness, so sampling any `True` at every
  stage is guaranteed to reach a valid full action — no dead ends. On an env with no action
  factorization it raises `ValueError`.
- **`pool.query_state_mask(state_ids, prefixes, width=None) -> bool[n, W]`** — the same
  conditional masks for **stored states** (`snapshot` / `simulate_batch` handles) instead of the
  live envs, so a tree search can read per-factor masks at any node. Rows come back in input order
  (ids may repeat); `state_id < 0` gives an all-false row without a registry lookup; an unknown or
  released id raises `ValueError`. `W` is `width`, else the spec's `pad_width`; a row of any other
  width (an env outside its factored mode answers empty masks) raises `ValueError`, and so does an
  env with no conditional masks at all. It reads the registry directly (no clone, no worker
  round-trip) in parallel with the GIL released. Also on `JaxPool.query_state_mask` (jittable).
  Caveat under `jit`: `JaxPool.query_state_mask`, `simulate_batch` and `release_batch` are all
  unordered callbacks, so inside one jitted function XLA may run a `release_batch` of the same ids
  *before* a query that has no data dependency on it (the query then raises `invalid state_id`).
  Make the release's inputs depend on the query's output, or release in a later call.
- **Commit** — the policy composes the chosen factor indices back to the flat index and commits
  via the normal `send` / `step` (the env owns the encoding; the policy reproduces it on-device).

## Notes

A flat space can be decomposed many ways (one flat Categorical, *independent* factors, or an
*autoregressive* chain); the env's spec names which one is in force. A factorization decomposes the
*entire* action — a setup where the agent picks only part of the action and a heuristic decides the
rest is a *smaller* action space, not a factorization.

A policy reads the spec, builds one head per factor, and loops over the factors: sample factor
0 from its mask, ask for the mask of factor 1 given that choice, and so on. The examples'
`factored_sample` helpers (`examples/networks.py`, `examples/torch_common.py`) implement that
loop for JAX and PyTorch.

## Worked example: factored BinPack (`item → ems`)

`BinPack-v0` takes the opt-in kwarg `factored=True`. Its flat action is `item * max_ems + ems`;
factored, it advertises a 2-factor conditional spec — factor 0 (`item`) is masked to items with
≥1 fitting EMS, factor 1 (`ems` | item) to the EMS that fit the chosen item. The flat `step`
encoding is unchanged, so the **same env** serves both a flat policy/search (via the flat
`action_mask` in `recv`) and a factored autoregressive policy (via `query_mask`).

**The flat mask is off by default when `factored=True`.** A factored policy never reads the
`max_items × max_ems` table — it asks `query_mask` for one factor at a time — so building and
shipping it every step is pure cost (at 256 × 512 it is 131,072 entries per env per step).
`info["action_mask"]` then carries the **stage-0 (item) mask** instead, padded to `pad_width`
(`[B, pad_width]`, `pool.action_mask_width() == pad_width`) — exactly `query_mask` at the empty
prefix. A factored sampler takes factor 0 from `recv` and queries only the `K-1` later stages
(`factored_sample(..., initial_mask=info["action_mask"])` in the examples), so a step costs `K-1`
`query_mask` round-trips, not `K`. `done` is decided by the same stage-0 scan ("is any item
placeable"), which equals "the flat mask has a legal entry". Pass `emit_flat_mask=True` when you
need the flat table too (tree search over the flat space), or `emit_flat_mask=False` to drop it
on a flat env (`info["action_mask"]` is then `[B, 0]`; derive the mask from the observation).

**Containers and orientations.** With `num_containers=C` and `orientations=O` the chain grows to
`item -> orientation -> container -> ems` (stages with a single option are dropped, so the
defaults keep `item -> ems`), and the flat space to `|A| = 25C · O · C · 50 = 1250 · O · C²` at
the default sizes. Every stage's mask is an existence witness (a value is valid iff it leaves a
valid completion), so the product of the chain's masks is exactly the flat mask, and an
autoregressive sampler never dead-ends. `query_mask` / `query_state_mask` take
`num_factors - 1` prefix columns (`-1` = unset).

```python
import numpy as np
import rustpool

pool = rustpool.make("BinPack-v0", num_envs=8, batch_size=8, max_items=25, max_ems=50,
                     factored=True)
spec = pool.action_factorization()
print(spec["factor_names"], spec["factor_widths"], spec["pad_width"])   # ['item', 'ems'] [25, 50] 50
```

One autoregressive step by hand, sampling uniformly among the valid choices at each stage:

```python
rng = np.random.default_rng(0)

def pick(mask):
    """A uniformly random True index per row."""
    return (rng.random(mask.shape) * mask).argmax(axis=1).astype(np.int32)

pool.async_reset()
obs, reward, done, info = pool.recv()
for _ in range(5):
    ids = info["env_id"]
    item = pick(info["action_mask"][:, :25])                 # stage 0 comes with recv
    ems_mask = pool.query_mask(item[:, None], ids)            # stage 1 given the item
    ems = pick(ems_mask[:, :50])
    pool.send(item * 50 + ems, ids)                           # flat action = item * max_ems + ems
    obs, reward, done, info = pool.recv()
    assert (reward > 0).all()                                 # every sampled action was valid
pool.close()
```

Runnable showcases:

| | PPO | AlphaZero |
| --- | --- | --- |
| JAX | `examples/factored_ppo_jax.py` | `examples/alphazero_jax.py --factored` |
| PyTorch | `examples/factored_ppo_torch.py` | — |

- **PPO** samples autoregressively: query factor-0 mask → sample item → `query_mask([item])` →
  sample ems → compose the flat action → `step`. The loss recomputes the summed factored
  log-prob/entropy under the *stored* masks (bit-identical to sampling). The policy evaluates
  `sum(factor_widths)` logits per step, never the flat `prod(factor_widths)` masked categorical.
- **AlphaZero** (`--factored`) keeps one tree edge per full action but never builds the flat
  mask: a node cannot enumerate its legal full actions, so its children are `--num-candidates`
  full actions sampled without replacement from the factored policy (Gumbel-top-k over items,
  then ems | item for the kept items only), with legality read per factor on the node's
  *stored* state via `query_state_mask(state_ids, prefixes)`. Each node costs `1 + K` mask rows
  and `K` ems-head calls instead of the `max_items × max_ems` table.
- **Flat-mask comparison**: `alphazero_{jax,torch}.py --task BinPack-v0` runs plain BinPack (full
  mask, every full action a child) with the same item/ems head composed into the full prior.

## The speedup

The implementation-independent metric is **candidate action-logits per step**:
`sum(factor_widths)` (factored) vs `prod(factor_widths)` (flat). `benchmarks/bench_factored.py`
in the repository measures the policy step (head + masked-categorical sample + log-prob), flat vs factored.
Representative numbers (`python benchmarks/bench_factored.py`; default batch=256, 200 iters, one
RTX 5090, TF32 — exact ms vary with hardware/torch version, so regenerate to reproduce):

| factors | flat \|A\| | candidate logits | flat ms | factored ms | speedup |
| --- | --- | --- | --- | --- | --- |
| 25×50 | 1,250 | 75 (16×) | 0.24 | 0.72 | 0.3× |
| 64×128 | 8,192 | 192 (42×) | 0.26 | 0.70 | 0.4× |
| 128×256 | 32,768 | 384 (85×) | 0.39 | 0.68 | 0.6× |
| 128×512 | 65,536 | 640 (102×) | 0.76 | 0.68 | 1.1× |
| 256×512 | 131,072 | 768 (170×) | 2.05 | 0.76 | 2.7× |

**Honest reading:** factored evaluates 16–170× fewer logits, but it costs `K-1` extra `query_mask`
host round-trips per step and the autoregressive head has a fixed overhead. So for *small* action
spaces the flat masked categorical is simply cheaper end-to-end; factored wins once the flat
`prod(widths)` head/categorical dominates — i.e. for genuinely large **combinatorial** action
spaces (here the crossover is ~65k flat actions, reaching 2.7× at 131k). The deeper the
factorization, the larger the gap between `sum(factor_widths)` and `prod(factor_widths)`.
