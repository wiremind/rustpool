"""PPO in pure JAX, training on a rustpool environment — single or multi-GPU.

A compact PPO: a ``jax.pmap`` (Anakin-style) training step whose rollout is a
``lax.scan`` over a device-sharded :class:`rustpool.jax_api.JaxPool`, with GAE and a clipped
PPO update (value clipping, entropy bonus, action masking). The *only* host hop is the env
transition (via ``io_callback``); everything else is one XLA program. Gradients are averaged
across devices with ``lax.pmean``.

* **Single GPU** (or CPU): runs as ``pmap`` over 1 device.
* **Multi-GPU**: ``--num-devices 0`` (auto) shards ``num_devices`` independent rustpool pools,
  one per device. Global batch =
  ``num_devices * num_envs * num_steps``.

Run
---
    pip install "rustpool[examples]"
    python examples/ppo_jax.py --task Maze-v0 --total-timesteps 2_000_000      # all GPUs
    python examples/ppo_jax.py --num-devices 1                                  # one device
    python examples/ppo_jax.py --smoke                                          # CI / quick

On ``Maze-v0`` the mean episodic return should climb from ~0 toward ~1.0 as the policy learns
to walk to the corner goal (invalid moves are masked out).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import pathlib
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from networks import ActorCritic, entropy, log_prob, sample_action  # noqa: E402

import rustpool.jax_api as rj  # noqa: E402


class Transition(NamedTuple):
    obs: dict
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    mask: jax.Array


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Backward GAE recursion: ``A_t = δ_t + γλ(1-done_t) A_{t+1}``."""

    def step(carry, x):
        gae, next_value = carry
        reward, value, done = x
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    return advantages, advantages + values


def make_update_step(pool, net, cfg):
    num_envs = pool.num_envs

    def update_step(train_state, env_out, ep_return, key):
        # ---- rollout (lax.scan over time) -----------------------------------------
        def rollout_body(carry, _):
            ts, out, ep_ret, key = carry
            key, sk = jax.random.split(key)
            logits, value = net.apply(ts.params, out.obs, out.action_mask)
            action = sample_action(sk, logits)
            lp = log_prob(logits, action)
            transition = Transition(out.obs, action, lp, value, None, None, out.action_mask)
            next_out = pool.step(action)
            ep_ret_new = ep_ret + next_out.reward
            completed = jnp.where(next_out.done, ep_ret_new, 0.0)
            n_done = next_out.done.astype(jnp.float32)
            ep_ret_next = jnp.where(next_out.done, 0.0, ep_ret_new)
            transition = transition._replace(reward=next_out.reward, done=next_out.done)
            return (ts, next_out, ep_ret_next, key), (transition, completed, n_done)

        (train_state, env_out, ep_return, key), (traj, completed, n_done) = jax.lax.scan(
            rollout_body, (train_state, env_out, ep_return, key), None, length=cfg.num_steps
        )

        _, last_value = net.apply(train_state.params, env_out.obs, env_out.action_mask)
        advantages, returns = compute_gae(
            traj.reward, traj.value, traj.done, last_value, cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- flatten [T, B, ...] -> [T*B, ...] ------------------------------------
        batch = (traj, advantages, returns)
        batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
        total = cfg.num_steps * num_envs
        num_minibatches = total // cfg.minibatch_size

        def epoch(carry, _):
            ts, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, total)
            shuffled = jax.tree.map(lambda x: x[perm], batch)
            minibatches = jax.tree.map(
                lambda x: x.reshape((num_minibatches, cfg.minibatch_size) + x.shape[1:]),
                shuffled,
            )

            def minibatch(ts, mb):
                traj_mb, adv_mb, ret_mb = mb

                def loss_fn(params):
                    logits, value = net.apply(params, traj_mb.obs, traj_mb.mask)
                    new_lp = log_prob(logits, traj_mb.action)
                    ratio = jnp.exp(new_lp - traj_mb.log_prob)
                    pg1 = ratio * adv_mb
                    pg2 = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
                    pg_loss = -jnp.minimum(pg1, pg2).mean()
                    v_clipped = traj_mb.value + jnp.clip(
                        value - traj_mb.value, -cfg.clip_eps, cfg.clip_eps
                    )
                    v_loss = 0.5 * jnp.maximum(
                        (value - ret_mb) ** 2, (v_clipped - ret_mb) ** 2
                    ).mean()
                    ent = entropy(logits).mean()
                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                    return loss, (pg_loss, v_loss, ent)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                # Average gradients across devices (Anakin) — no-op with a single device.
                grads = jax.lax.pmean(grads, axis_name="device")
                ts = ts.apply_gradients(grads=grads)
                return ts, (loss, *aux)

            ts, metrics = jax.lax.scan(minibatch, ts, minibatches)
            return (ts, key), jax.tree.map(jnp.mean, metrics)

        (train_state, key), epoch_metrics = jax.lax.scan(
            epoch, (train_state, key), None, length=cfg.update_epochs
        )

        loss, pg_loss, v_loss, ent = jax.tree.map(jnp.mean, epoch_metrics)
        ep_count = jnp.maximum(n_done.sum(), 1.0)
        metrics = {
            "episodic_return": completed.sum() / ep_count,
            "episodes": n_done.sum(),
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": ent,
        }
        metrics = jax.lax.pmean(metrics, axis_name="device")
        return (train_state, env_out, ep_return, key), metrics

    return update_step


def train(cfg):
    n_devices = cfg.num_devices or jax.local_device_count()
    cfg.num_updates = max(
        cfg.total_timesteps // (cfg.num_steps * cfg.num_envs * n_devices), 1
    )
    key = jax.random.key(cfg.seed)
    key, net_key = jax.random.split(key)

    env_kwargs = {}
    if cfg.task == "Maze-v0":
        env_kwargs = {"width": cfg.maze_size, "height": cfg.maze_size}
    pool = rj.JaxPool(cfg.task, num_envs=cfg.num_envs, num_devices=n_devices,
                      seed=cfg.seed, **env_kwargs)

    net = ActorCritic(num_actions=pool.num_actions, hidden_sizes=cfg.hidden_sizes)

    # Init params once on host using a device-0 reset for shapes, then replicate.
    probe = jax.jit(pool.reset)(jax.random.key(0))
    params = net.init(net_key, probe.obs, probe.action_mask)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5)
    )
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)
    # Replicate the train state across devices for pmap (leading axis = n_devices).
    def _replicate(x):
        x = jnp.asarray(x)
        return jnp.broadcast_to(x, (n_devices, *x.shape))

    train_state = jax.tree.map(_replicate, train_state)

    # Per-device env reset + per-device keys.
    reset_keys = jax.random.split(jax.random.key(cfg.seed + 1), n_devices)
    env_out = jax.pmap(pool.reset, axis_name="device")(reset_keys)
    ep_return = jnp.zeros((n_devices, cfg.num_envs))
    run_keys = jax.random.split(key, n_devices)

    update_step = jax.pmap(make_update_step(pool, net, cfg), axis_name="device")

    steps_per_update = cfg.num_steps * cfg.num_envs * n_devices
    print(
        f"PPO on {cfg.task} | devices={n_devices}x {jax.default_backend()} "
        f"| num_envs/device={cfg.num_envs} num_steps={cfg.num_steps} "
        f"updates={cfg.num_updates} | global batch {steps_per_update:,}/update"
    )
    t0 = time.time()
    for update in range(cfg.num_updates):
        (train_state, env_out, ep_return, run_keys), metrics = update_step(
            train_state, env_out, ep_return, run_keys
        )
        if update % cfg.log_every == 0 or update == cfg.num_updates - 1:
            m = jax.tree.map(lambda x: float(x[0]), metrics)
            elapsed = time.time() - t0
            sps = steps_per_update * (update + 1) / max(elapsed, 1e-6)
            print(
                f"  update {update:4d} | return {m['episodic_return']:+.3f} "
                f"| loss {m['loss']:+.3f} | entropy {m['entropy']:.3f} "
                f"| {sps:,.0f} steps/s"
            )
    pool.close()
    return jax.tree.map(lambda x: float(x[0]), metrics)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Maze-v0", choices=("Maze-v0", "BinPack-v0"),
                   help="an env with an action mask (CartPole-v1 has none)")
    p.add_argument("--maze-size", type=int, default=10)
    p.add_argument("--num-devices", type=int, default=0, help="0 = all local devices")
    p.add_argument("--num-envs", type=int, default=256, help="environments per device")
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
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
    p.add_argument("--smoke", action="store_true", help="tiny run for CI / a quick check")
    cfg = p.parse_args(argv)
    if cfg.smoke:
        cfg.maze_size, cfg.num_envs, cfg.num_steps = 5, 16, 8
        cfg.minibatch_size, cfg.total_timesteps, cfg.update_epochs = 64, 8 * 16 * 3, 2
        cfg.log_every, cfg.num_devices = 1, 1
    # num_updates depends on the resolved device count; computed in train().
    return cfg


if __name__ == "__main__":
    train(parse_args())
