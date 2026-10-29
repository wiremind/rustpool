"""Factored (autoregressive) PPO in pure JAX on rustpool's factored BinPack action space.

With ``factored=True`` BinPack advertises a 2-factor (``item → ems``) conditional action space
(``action_factorization()`` + ``query_mask``). This PPO samples the flat action one factor at a
time under per-factor ∃-witness masks queried on-device via ``JaxPool.query_mask`` (an
``io_callback``), so the policy evaluates ``sum(factor_widths)`` logits per step instead of the
flat ``prod(factor_widths)`` masked categorical. The whole training step is one ``jax.jit``
program (single device here for clarity; multi-GPU works exactly as in ``ppo_jax.py``).

See ``ppo_jax.py`` for the flat baseline and ``factored_ppo_torch.py`` for the PyTorch version.
``benchmarks/bench_factored.py`` measures the flat-vs-factored policy step: factored evaluates far
fewer candidate logits, but it wins on wall-clock only for *large* combinatorial action spaces
(small ones are slower — see ``docs/concepts/factored-actions.md``).

Run
---
    pip install "rustpool[examples]"
    python examples/factored_ppo_jax.py --max-items 25 --max-ems 50
    python examples/factored_ppo_jax.py --smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from networks import FactoredActorCritic, factored_log_prob_entropy, factored_sample

import rustpool.jax_api as rj


class Transition(NamedTuple):
    obs: dict
    choices: jax.Array  # [B, K]
    masks: jax.Array  # [B, K, pad]
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    def step(carry, x):
        gae, next_value = carry
        reward, value, done = x
        nd = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * nd - value
        gae = delta + gamma * gae_lambda * nd * gae
        return (gae, value), gae

    _, adv = jax.lax.scan(
        step, (jnp.zeros_like(last_value), last_value), (rewards, values, dones), reverse=True
    )
    return adv, adv + values


def make_update_step(pool, net, widths, cfg):
    num_envs = pool.num_envs
    env_ids = jnp.arange(num_envs, dtype=jnp.int32)

    def query_mask_fn(prefix):
        return pool.query_mask(prefix, env_ids)

    def update_step(train_state, env_out, ep_return, key):
        def rollout_body(carry, _):
            ts, out, ep_ret, key = carry
            key, sk = jax.random.split(key)
            # Factor 0's mask came back with the obs (stage-0 action_mask): K-1 queries.
            flat, logp, choices, masks, value = factored_sample(
                net, ts.params, out.obs, query_mask_fn, widths, sk, initial_mask=out.action_mask
            )
            trans = Transition(out.obs, choices, masks, logp, value, None, None)
            nxt = pool.step(flat)
            ep_ret_new = ep_ret + nxt.reward
            completed = jnp.where(nxt.done, ep_ret_new, 0.0)
            n_done = nxt.done.astype(jnp.float32)
            ep_ret_next = jnp.where(nxt.done, 0.0, ep_ret_new)
            trans = trans._replace(reward=nxt.reward, done=nxt.done)
            return (ts, nxt, ep_ret_next, key), (trans, completed, n_done)

        (train_state, env_out, ep_return, key), (traj, completed, n_done) = jax.lax.scan(
            rollout_body, (train_state, env_out, ep_return, key), None, length=cfg.num_steps
        )

        last_feats = net.apply(train_state.params, env_out.obs, method=net.features)
        last_value = net.apply(train_state.params, last_feats, method=net.value)
        advantages, returns = compute_gae(
            traj.reward, traj.value, traj.done, last_value, cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = (traj, advantages, returns)
        batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
        total = cfg.num_steps * num_envs
        num_minibatches = total // cfg.minibatch_size

        def epoch(carry, _):
            ts, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, total)
            shuffled = jax.tree.map(lambda x: x[perm], batch)
            mbs = jax.tree.map(
                lambda x: x.reshape((num_minibatches, cfg.minibatch_size) + x.shape[1:]), shuffled
            )

            def minibatch(ts, mb):
                traj_mb, adv_mb, ret_mb = mb

                def loss_fn(params):
                    new_lp, ent, value = factored_log_prob_entropy(
                        net, params, traj_mb.obs, traj_mb.choices, traj_mb.masks, widths
                    )
                    ratio = jnp.exp(new_lp - traj_mb.log_prob)
                    pg = -jnp.minimum(
                        ratio * adv_mb,
                        jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb,
                    ).mean()
                    v_loss = 0.5 * ((value - ret_mb) ** 2).mean()
                    loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent.mean()
                    return loss, (pg, v_loss, ent.mean())

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                return ts.apply_gradients(grads=grads), (loss, *aux)

            ts, metrics = jax.lax.scan(minibatch, ts, mbs)
            return (ts, key), jax.tree.map(jnp.mean, metrics)

        (train_state, key), epoch_metrics = jax.lax.scan(
            epoch, (train_state, key), None, length=cfg.update_epochs
        )
        loss, pg, v_loss, ent = jax.tree.map(jnp.mean, epoch_metrics)
        ep_count = jnp.maximum(n_done.sum(), 1.0)
        metrics = {"episodic_return": completed.sum() / ep_count, "loss": loss, "entropy": ent}
        return (train_state, env_out, ep_return, key), metrics

    return update_step


def train(cfg):
    key = jax.random.key(cfg.seed)
    key, net_key = jax.random.split(key)
    pool = rj.JaxPool("BinPack-v0", num_envs=cfg.num_envs, num_devices=1, seed=cfg.seed,
                      max_items=cfg.max_items, max_ems=cfg.max_ems, factored=True)
    spec = pool.action_factorization()
    assert spec is not None
    widths = tuple(spec["factor_widths"])

    net = FactoredActorCritic(factor_widths=widths, hidden_sizes=tuple(cfg.hidden_sizes))
    probe = jax.jit(pool.reset)(jax.random.key(0))
    params = net.init(net_key, probe.obs)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)

    env_out = pool.reset(jax.random.key(cfg.seed + 1))
    ep_return = jnp.zeros(cfg.num_envs)
    update_step = jax.jit(make_update_step(pool, net, widths, cfg))

    steps_per_update = cfg.num_steps * cfg.num_envs
    print(
        f"Factored PPO (jax) on BinPack-v0 | {jax.default_backend()} | num_envs={cfg.num_envs} "
        f"num_steps={cfg.num_steps} | factors={list(widths)} -> flat {spec['flat_action_size']} "
        f"(policy evaluates {sum(widths)} logits/step)"
    )
    t0 = time.time()
    for update in range(cfg.num_updates):
        (train_state, env_out, ep_return, key), metrics = update_step(
            train_state, env_out, ep_return, key
        )
        if update % cfg.log_every == 0 or update == cfg.num_updates - 1:
            m = jax.tree.map(float, metrics)
            sps = steps_per_update * (update + 1) / max(time.time() - t0, 1e-6)
            print(
                f"  update {update:4d} | return {m['episodic_return']:+.3f} "
                f"| loss {m['loss']:+.3f} | entropy {m['entropy']:.3f} | {sps:,.0f} steps/s"
            )
    pool.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-items", type=int, default=25)
    p.add_argument("--max-ems", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    cfg = p.parse_args(argv)
    if cfg.smoke:
        cfg.max_items, cfg.max_ems, cfg.num_envs = 8, 12, 16
        cfg.num_steps, cfg.minibatch_size, cfg.total_timesteps = 6, 32, 16 * 6 * 3
        cfg.update_epochs, cfg.log_every = 2, 1
    cfg.num_updates = max(cfg.total_timesteps // (cfg.num_steps * cfg.num_envs), 1)
    return cfg


if __name__ == "__main__":
    train(parse_args())
