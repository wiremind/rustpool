"""PPO in PyTorch, training on a rustpool environment.

A compact, single-file PPO that follows the canonical CleanRL ``ppo.py`` recipe
(https://github.com/vwxyzjn/cleanrl, MIT; see the "37 implementation details of PPO",
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) with invalid-action
masking via ``CategoricalMasked`` (Huang & Ontañón 2020, "A Closer Look at Invalid Action
Masking in Policy Gradient Algorithms"). It is wired to :class:`rustpool.torch_api.TorchEnvPool`
(dict observations, ``info["action_mask"]``, int32 actions, in-worker auto-reset).

For the JAX version, see ``ppo_jax.py``.

Run
---
    pip install "rustpool[torch]"
    python examples/ppo_torch.py --task Maze-v0 --total-timesteps 1_000_000
    python examples/ppo_torch.py --task BinPack-v0 --clip-coef 0.1
    python examples/ppo_torch.py --smoke          # tiny, CPU, for CI / a quick check
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch_common import CategoricalMasked, compute_gae, flatten_obs, layer_init

from rustpool.torch_api import TorchEnvPool


class Agent(nn.Module):
    """Separate-torso actor-critic MLP (CleanRL discrete style)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, num_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).flatten()

    def get_action_and_value(self, x, mask, action=None):
        dist = CategoricalMasked(self.actor(x), mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x).flatten()


def train(cfg):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # TF32 tensor cores (free on Ampere+/Blackwell)

    env_kwargs = {"width": cfg.maze_size, "height": cfg.maze_size} if cfg.task == "Maze-v0" else {}
    pool = TorchEnvPool(cfg.task, num_envs=cfg.num_envs, device=device, seed=cfg.seed, **env_kwargs)
    A = pool.num_actions

    next_obs_raw, _, _, info = pool.reset()
    next_obs = flatten_obs(next_obs_raw)
    next_mask = info["action_mask"]
    obs_dim = next_obs.shape[-1]

    agent = Agent(obs_dim, A, cfg.hidden).to(device)
    optimizer = optim.Adam(
        agent.parameters(), lr=cfg.lr, eps=1e-5, fused=(device.type == "cuda")
    )
    if cfg.compile:
        # Compile the MLPs with CUDA graphs (reduce-overhead): small-model RL is
        # kernel-launch-overhead bound, so this is the big lever. The distribution/sampling
        # math stays eager to avoid graph breaks.
        agent.actor = torch.compile(agent.actor, mode="reduce-overhead")
        agent.critic = torch.compile(agent.critic, mode="reduce-overhead")

    # Rollout storage [T, B, ...]
    T, B = cfg.num_steps, cfg.num_envs
    obs_buf = torch.zeros((T, B, obs_dim), device=device)
    mask_buf = torch.zeros((T, B, A), dtype=torch.bool, device=device)
    actions_buf = torch.zeros((T, B), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((T, B), device=device)
    rewards_buf = torch.zeros((T, B), device=device)
    dones_buf = torch.zeros((T, B), device=device)
    values_buf = torch.zeros((T, B), device=device)

    batch_size = T * B
    minibatch_size = batch_size // cfg.num_minibatches
    ep_return = torch.zeros(B, device=device)

    print(
        f"PPO (torch) on {cfg.task} | device={device} | num_envs={B} num_steps={T} "
        f"iterations={cfg.num_iterations} | batch {batch_size}"
    )
    t0 = time.time()
    global_step = 0
    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.compile:
            # New step: lets CUDA-graph buffers be safely reused this iteration.
            torch.compiler.cudagraph_mark_step_begin()
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.lr

        # Metric accumulators stay on-device (no per-step host sync); read once at log time.
        completed = torch.zeros((), device=device)
        n_done = torch.zeros((), device=device)
        for t in range(T):
            global_step += B
            obs_buf[t] = next_obs
            mask_buf[t] = next_mask
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_mask)
            values_buf[t] = value
            actions_buf[t] = action
            logprobs_buf[t] = logprob

            next_obs_raw, reward, done, info = pool.step(action)
            done_f = done.float()
            rewards_buf[t] = reward
            dones_buf[t] = done_f  # episode ended at step t (rustpool auto-resets)
            ep_return = ep_return + reward
            completed = completed + (ep_return * done_f).sum()
            n_done = n_done + done_f.sum()
            ep_return = ep_return * (1.0 - done_f)
            next_obs = flatten_obs(next_obs_raw)
            next_mask = info["action_mask"]

        # GAE (bootstrap from the value of the final observation)
        with torch.no_grad():
            last_value = agent.get_value(next_obs)
        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf, last_value, cfg.gamma, cfg.gae_lambda
        )

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_masks = mask_buf.reshape(-1, A)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        clipfracs = []
        for _ in range(cfg.update_epochs):
            b_inds = torch.randperm(batch_size, device=device)  # on-device: no host sync
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_masks[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean())

                mb_adv = b_advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if iteration % cfg.log_every == 0 or iteration == cfg.num_iterations:
            ret = (completed / n_done.clamp_min(1.0)).item()
            clipfrac = torch.stack(clipfracs).mean().item()
            sps = global_step / max(time.time() - t0, 1e-6)
            print(
                f"  iter {iteration:4d} | return {ret:+.3f} | loss {loss.item():+.3f} "
                f"| entropy {entropy_loss.item():.3f} | clipfrac {clipfrac:.3f} "
                f"| {sps:,.0f} steps/s"
            )
    pool.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Maze-v0", choices=("Maze-v0", "BinPack-v0"),
                   help="an env with an action mask (CartPole-v1 has none)")
    p.add_argument("--maze-size", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the MLPs with CUDA graphs (reduce-overhead)")
    p.add_argument("--smoke", action="store_true", help="tiny run for CI / a quick check")
    cfg = p.parse_args(argv)
    if cfg.smoke:
        cfg.device, cfg.maze_size, cfg.num_envs = "cpu", 5, 16
        cfg.num_steps, cfg.total_timesteps, cfg.num_minibatches = 8, 8 * 16 * 3, 2
        cfg.update_epochs, cfg.log_every = 2, 1
    cfg.num_iterations = max(cfg.total_timesteps // (cfg.num_steps * cfg.num_envs), 1)
    return cfg


if __name__ == "__main__":
    train(parse_args())
