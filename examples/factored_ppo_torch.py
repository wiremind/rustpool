"""Factored (autoregressive) PPO in PyTorch on rustpool's factored BinPack action space.

BinPack's action is a flat ``item × ems`` index. With ``factored=True`` the env advertises a
2-factor (``item → ems``) **conditional** action space (``action_factorization()`` +
``query_mask``). This PPO samples the action one factor at a time under per-factor ∃-witness
masks queried from the env, so the policy evaluates ``sum(factor_widths)`` logits per step instead
of materialising the flat ``prod(factor_widths)`` masked categorical. See ``ppo_torch.py`` for the
flat baseline and ``factored_ppo_jax.py`` for the JAX version. ``benchmarks/bench_factored.py``
measures the flat-vs-factored policy step: factored evaluates far fewer candidate logits but wins on
wall-clock only for *large* combinatorial action spaces (small ones are slower — see
``docs/concepts/factored-actions.md``).

Run
---
    pip install "rustpool[torch]"
    python examples/factored_ppo_torch.py --max-items 25 --max-ems 50
    python examples/factored_ppo_torch.py --smoke
"""

from __future__ import annotations

import argparse
import time
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch_common import (
    FactoredActorCritic,
    compute_gae,
    factored_log_prob_entropy,
    factored_sample,
    flatten_obs,
)

from rustpool.torch_api import TorchEnvPool


class Transition(NamedTuple):
    obs: torch.Tensor  # flattened [B, feat]
    choices: torch.Tensor  # [B, K] factor indices
    masks: torch.Tensor  # [B, K, pad] per-factor masks (for teacher-forced loss)
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


def train(cfg):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    pool = TorchEnvPool(
        "BinPack-v0", num_envs=cfg.num_envs, device=device, seed=cfg.seed,
        max_items=cfg.max_items, max_ems=cfg.max_ems, factored=True,
    )
    spec = pool.action_factorization()
    assert spec is not None, "env is not factored; pass factored=True"
    widths = list(spec["factor_widths"])

    obs_raw, _, _, info = pool.reset()
    next_obs = flatten_obs(obs_raw)
    next_stage0 = info["action_mask"]  # stage-0 (item) mask: factor 0 needs no query
    obs_dim = next_obs.shape[-1]

    net = FactoredActorCritic(obs_dim, widths, cfg.hidden_sizes).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5, fused=(device.type == "cuda"))

    def query_mask_fn(prefix):
        return pool.query_mask(prefix)

    T, B, K = cfg.num_steps, cfg.num_envs, len(widths)
    batch_size = T * B
    minibatch_size = batch_size // cfg.num_minibatches
    ep_return = torch.zeros(B, device=device)

    print(
        f"Factored PPO (torch) on BinPack-v0 | device={device} | num_envs={B} num_steps={T} "
        f"| factors={widths} -> flat {spec['flat_action_size']} (policy evaluates "
        f"{sum(widths)} logits/step, not {spec['flat_action_size']})"
    )
    t0, global_step = time.time(), 0
    for iteration in range(1, cfg.num_iterations + 1):
        traj = []
        completed = torch.zeros((), device=device)
        n_done = torch.zeros((), device=device)
        for _ in range(T):
            global_step += B
            with torch.no_grad():
                flat, logp, _, choices, masks, value = factored_sample(
                    net, next_obs, query_mask_fn, widths, initial_mask=next_stage0
                )
            obs_raw, reward, done, info = pool.step(flat)
            next_stage0 = info["action_mask"]
            done_f = done.float()
            traj.append(Transition(next_obs, choices, masks, logp, value, reward, done_f))
            ep_return = ep_return + reward
            completed = completed + (ep_return * done_f).sum()
            n_done = n_done + done_f.sum()
            ep_return = ep_return * (1.0 - done_f)
            next_obs = flatten_obs(obs_raw)

        with torch.no_grad():
            last_value = net.value(net.features(next_obs))
        rewards = torch.stack([t.reward for t in traj])
        values = torch.stack([t.value for t in traj])
        dones = torch.stack([t.done for t in traj])
        advantages, returns = compute_gae(rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda)

        b_obs = torch.stack([t.obs for t in traj]).reshape(batch_size, obs_dim)
        b_choices = torch.stack([t.choices for t in traj]).reshape(batch_size, K)
        b_masks = torch.stack([t.masks for t in traj]).reshape(batch_size, K, -1)
        b_logp = torch.stack([t.log_prob for t in traj]).reshape(batch_size)
        b_adv = advantages.reshape(batch_size)
        b_ret = returns.reshape(batch_size)

        for _ in range(cfg.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = perm[start : start + minibatch_size]
                new_logp, entropy, new_value = factored_log_prob_entropy(
                    net, b_obs[mb], b_choices[mb], b_masks[mb], widths
                )
                ratio = (new_logp - b_logp[mb]).exp()
                adv = b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = torch.max(-adv * ratio,
                               -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_loss = 0.5 * ((new_value - b_ret[mb]) ** 2).mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if iteration % cfg.log_every == 0 or iteration == cfg.num_iterations:
            ret = (completed / n_done.clamp_min(1.0)).item()
            sps = global_step / max(time.time() - t0, 1e-6)
            print(
                f"  iter {iteration:4d} | return {ret:+.3f} | loss {loss.item():+.3f} "
                f"| entropy {entropy.mean().item():.3f} | {sps:,.0f} steps/s"
            )
    pool.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-items", type=int, default=25)
    p.add_argument("--max-ems", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    cfg = p.parse_args(argv)
    if cfg.smoke:
        cfg.device, cfg.max_items, cfg.max_ems, cfg.num_envs = "cpu", 8, 12, 16
        cfg.num_steps, cfg.total_timesteps, cfg.num_minibatches = 6, 16 * 6 * 3, 2
        cfg.update_epochs, cfg.log_every = 2, 1
    cfg.num_iterations = max(cfg.total_timesteps // (cfg.num_steps * cfg.num_envs), 1)
    return cfg


if __name__ == "__main__":
    train(parse_args())
