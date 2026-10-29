"""AlphaZero in PyTorch, searching through the *real* rustpool simulator.

PyTorch has no equivalent of JAX's ``mctx``, so the MCTS is hand-written here — and, crucially,
**array-batched**: B independent searches run in lockstep and the search calls
``pool.simulate_batch`` exactly **once per expansion** (a ``[B]`` batch), which is what makes
rustpool's external-simulator search substrate worthwhile. The tree-as-arrays layout follows
DeepMind's ``mctx`` (Apache-2.0); the single-agent PUCT / target conventions follow
``alpha-zero-general`` (MIT) with all two-player sign-flips removed; the rustpool lifecycle
(snapshot root → simulate_batch as the dynamics → release the whole tree) mirrors
``alphazero_jax.py``.

For the JAX version (using mctx), see ``alphazero_jax.py``.

Run
---
    pip install "rustpool[torch]"
    python examples/alphazero_torch.py --task Maze-v0 --num-simulations 32
    python examples/alphazero_torch.py --task BinPack-v0   # full mask, item/ems policy head
    python examples/alphazero_torch.py --smoke      # tiny, CPU, for CI / a quick check
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch_common import (
    NEG_INF,
    FactoredActorCritic,
    compute_gae,
    factored_flat_logits,
    flatten_obs,
    layer_init,
    mask_logits,
    masked_entropy,
)

from rustpool.torch_api import TorchEnvPool

ROOT = 0


class AZNet(nn.Module):
    """Shared-torso policy+value net. ``forward`` returns ``(masked_logits, value[B])``."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.torso = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.policy_head = layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, obs_flat, mask):
        h = self.torso(obs_flat)
        logits = mask_logits(self.policy_head(h), mask)
        return logits, self.value_head(h).squeeze(-1)


class BatchedMCTS:
    """Array-batched MCTS driving ``pool.simulate_batch`` (one call per expansion)."""

    def __init__(self, pool, pv_fn, B, A, num_simulations, gamma, device,
                 c_init=1.25, c_base=19652.0, dirichlet_eps=0.25, dirichlet_alpha=0.3):
        # pv_fn(obs, mask) -> (masked_flat_logits, value): a flat AZNet, or (factored) a 2-factor
        # head composed into the flat prior. The search is always over the flat masked actions.
        self.pool, self.pv_fn = pool, pv_fn
        self.B, self.A, self.N = B, A, num_simulations
        self.gamma, self.device = gamma, device
        self.c_init, self.c_base = c_init, c_base
        self.eps, self.alpha = dirichlet_eps, dirichlet_alpha
        self.bidx = torch.arange(B, device=device)

    def _alloc(self):
        B, A, M, dev = self.B, self.A, self.N + 1, self.device
        self.node_visits = torch.zeros((B, M), dtype=torch.int32, device=dev)
        self.node_values = torch.zeros((B, M), device=dev)
        self.node_state_ids = torch.zeros((B, M), dtype=torch.int32, device=dev)
        self.parents = torch.full((B, M), -1, dtype=torch.long, device=dev)
        self.action_from_parent = torch.full((B, M), -1, dtype=torch.long, device=dev)
        self.child_index = torch.full((B, M, A), -1, dtype=torch.long, device=dev)
        self.child_visits = torch.zeros((B, M, A), device=dev)
        self.child_rewards = torch.zeros((B, M, A), device=dev)
        self.child_discounts = torch.zeros((B, M, A), device=dev)
        self.child_prior_logits = torch.full((B, M, A), NEG_INF, device=dev)

    def _select_action(self, cur):
        b = self.bidx
        pl = self.child_prior_logits[b, cur]            # [B,A] masked logits
        prior = torch.softmax(pl, dim=-1)
        valid = pl > (NEG_INF / 2)
        nchild = self.child_visits[b, cur]              # [B,A]
        cidx = self.child_index[b, cur]                 # [B,A]
        parent_val = self.node_values[b, cur]           # [B]
        child_val = torch.gather(self.node_values, 1, cidx.clamp_min(0))
        q = self.child_rewards[b, cur] + self.child_discounts[b, cur] * child_val
        q = torch.where(cidx >= 0, q, parent_val.unsqueeze(1))   # unvisited -> parent value
        big = torch.full_like(q, 1e30)
        qmax = torch.maximum(parent_val, torch.where(valid, q, -big).max(-1).values)
        qmin = torch.minimum(parent_val, torch.where(valid, q, big).min(-1).values)
        q_norm = (q - qmin.unsqueeze(1)) / (qmax - qmin).clamp_min(1e-8).unsqueeze(1)
        q_norm = torch.where(valid, q_norm, torch.zeros_like(q_norm))
        nparent = self.node_visits[b, cur].float()      # [B]
        pb_c = self.c_init + torch.log((nparent + self.c_base + 1) / self.c_base)
        prior_score = torch.sqrt(nparent).unsqueeze(1) * pb_c.unsqueeze(1) * prior / (1 + nchild)
        score = torch.where(valid, q_norm + prior_score, -big)
        return score.argmax(-1)                         # [B] long

    def _eval(self, obs, mask):
        return self.pv_fn(obs, mask)

    @torch.no_grad()
    def run(self, obs, mask, *, add_noise=True):
        b = self.bidx
        self._alloc()
        # --- root ---
        self.node_state_ids[:, ROOT] = self.pool.snapshot()
        logits, value = self._eval(obs, mask)
        prior = torch.softmax(logits, dim=-1)
        if add_noise:
            valid = mask.bool().float()
            noise = torch.distributions.Dirichlet(
                torch.full((self.A,), self.alpha, device=self.device)
            ).sample((self.B,)) * valid
            noise = noise / noise.sum(-1, keepdim=True).clamp_min(1e-8)
            prior = (1 - self.eps) * prior + self.eps * noise
            prior = prior * valid
            prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
        # Store as masked log-priors: invalid actions must be NEG_INF (not log(1e-12)) so the
        # selection's validity check excludes them and no visit mass lands on invalid actions.
        root_logits = torch.log(prior.clamp_min(1e-12))
        self.child_prior_logits[:, ROOT] = torch.where(
            mask.bool(), root_logits, torch.full_like(root_logits, NEG_INF))
        self.node_values[:, ROOT] = value
        self.node_visits[:, ROOT] = 1

        for sim in range(self.N):
            # --- select: descend until an unvisited child edge ---
            cur = torch.zeros(self.B, dtype=torch.long, device=self.device)
            leaf_parent = torch.zeros(self.B, dtype=torch.long, device=self.device)
            leaf_action = torch.zeros(self.B, dtype=torch.long, device=self.device)
            active = torch.ones(self.B, dtype=torch.bool, device=self.device)
            for _ in range(self.N + 1):
                a = self._select_action(cur)
                child = self.child_index[b, cur, a]
                is_leaf = active & (child < 0)
                leaf_parent = torch.where(is_leaf, cur, leaf_parent)
                leaf_action = torch.where(is_leaf, a, leaf_action)
                active = active & (child >= 0)
                cur = torch.where(active, child, cur)
                # Early-break once every lane has hit its leaf edge. The `any()` is a tiny host
                # sync, but for these shallow trees it saves running the full N+1 descent depths
                # every simulation — measured ~2.5x faster than removing it (the dominant cost is
                # the simulate_batch host round-trip below, not this sync).
                if not active.any():
                    break

            # --- expand: ONE simulate_batch across the whole batch ---
            new_idx = sim + 1
            parent_ids = self.node_state_ids[b, leaf_parent]
            obs_c, r_c, d_c, info_c = self.pool.simulate_batch(parent_ids, leaf_action.int())
            self.node_state_ids[:, new_idx] = info_c["state_id"].to(torch.int32)
            self.parents[:, new_idx] = leaf_parent
            self.action_from_parent[:, new_idx] = leaf_action
            self.child_index[b, leaf_parent, leaf_action] = new_idx
            self.child_rewards[b, leaf_parent, leaf_action] = r_c.float()
            self.child_discounts[b, leaf_parent, leaf_action] = (1.0 - d_c.float()) * self.gamma
            logits, value = self._eval(obs_c, info_c["action_mask"])
            self.child_prior_logits[:, new_idx] = logits
            self.node_values[:, new_idx] = value
            self.node_visits[:, new_idx] = 1

            # --- backup: walk parents -> root, discounted bootstrap ---
            idx = torch.full((self.B,), new_idx, dtype=torch.long, device=self.device)
            val = value.clone()
            active = idx != ROOT
            for _ in range(self.N + 1):
                parent = self.parents[b, idx].clamp_min(0)
                a = self.action_from_parent[b, idx].clamp_min(0)
                new_val = self.child_rewards[b, parent, a] + self.child_discounts[b, parent, a] * val
                pv = self.node_visits[b, parent].float() + 1.0
                cur_val = self.node_values[b, parent]
                upd = cur_val + (new_val - cur_val) / pv
                self.node_visits[b, parent] = torch.where(
                    active, pv.to(torch.int32), self.node_visits[b, parent])
                self.node_values[b, parent] = torch.where(active, upd, self.node_values[b, parent])
                cv = self.child_visits[b, parent, a]
                self.child_visits[b, parent, a] = torch.where(active, cv + 1.0, cv)
                val = torch.where(active, new_val, val)
                idx = torch.where(active, self.parents[b, idx], idx)
                active = active & (idx != ROOT)
                if not active.any():  # cheap; avoids walking the full depth bound every backup
                    break

        visits = self.child_visits[b, ROOT]
        policy = visits / visits.sum(-1, keepdim=True).clamp_min(1.0)
        search_value = self.node_values[b, ROOT].clone()
        # release every snapshot in the tree (visited slots; -1 elsewhere is ignored)
        flat = self.node_state_ids.reshape(-1)
        keep = self.node_visits.reshape(-1) > 0
        self.pool.release_batch(torch.where(keep, flat, torch.full_like(flat, -1)))
        return policy, search_value


def temperature_sample(policy, temperature):
    if temperature <= 1e-6:
        m = policy.max(-1, keepdim=True).values
        is_max = (policy >= m - 1e-8).float()
        tie = torch.rand_like(policy) * is_max
        return tie.argmax(-1).int()
    p = policy.clamp_min(0) ** (1.0 / temperature)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(p, 1).squeeze(-1).int()


def train(cfg):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # TF32 tensor cores (free on Ampere+/Blackwell)

    task = cfg.task
    binpack = task == "BinPack-v0"
    if binpack:
        # Plain BinPack-v0 emits the full mask: the MCTS searches every full action. (The
        # factored, no-full-mask mode with sampled actions per node is JAX-only for now.)
        pool = TorchEnvPool("BinPack-v0", num_envs=cfg.num_envs, device=device, seed=cfg.seed,
                            max_items=cfg.max_items, max_ems=cfg.max_ems)
        widths = [cfg.max_items, cfg.max_ems]
    else:
        env_kwargs = {"width": cfg.maze_size, "height": cfg.maze_size} if cfg.task == "Maze-v0" else {}
        pool = TorchEnvPool(cfg.task, num_envs=cfg.num_envs, device=device, seed=cfg.seed, **env_kwargs)
    A, B = pool.num_actions, cfg.num_envs

    obs_raw, _, _, info = pool.reset()
    obs_flat = flatten_obs(obs_raw)
    mask = info["action_mask"]
    obs_dim = obs_flat.shape[-1]

    if binpack:
        # Item head + ems-given-item head, composed into the full prior the search needs.
        net = FactoredActorCritic(obs_dim, widths, (cfg.hidden, cfg.hidden)).to(device)

        def pv_fn(obs, m):
            flat = flatten_obs(obs) if isinstance(obs, dict) else obs
            logits, value = factored_flat_logits(net, flat, widths)
            return mask_logits(logits, m), value
    else:
        net = AZNet(obs_dim, A, cfg.hidden).to(device)

        def pv_fn(obs, m):
            flat = flatten_obs(obs) if isinstance(obs, dict) else obs
            return net(flat, m)

    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5, fused=(device.type == "cuda"))
    if cfg.compile and not binpack:
        net = torch.compile(net, mode="reduce-overhead")  # pv_fn closes over net (late binding)
    mcts = BatchedMCTS(pool, pv_fn, B, A, cfg.num_simulations, cfg.gamma, device,
                       dirichlet_alpha=cfg.dirichlet_alpha or (10.0 / A))

    T = cfg.num_steps
    print(
        f"AlphaZero (torch) on {task}{' [full mask]' if binpack else ''} | device={device} "
        f"| num_envs={B} num_steps={T} sims={cfg.num_simulations} iterations={cfg.num_iterations}"
    )
    t0, global_step = time.time(), 0
    ep_return = torch.zeros(B, device=device)
    for iteration in range(1, cfg.num_iterations + 1):
        obs_b = torch.zeros((T, B, obs_dim), device=device)
        mask_b = torch.zeros((T, B, A), dtype=torch.bool, device=device)
        pol_b = torch.zeros((T, B, A), device=device)
        val_b = torch.zeros((T, B), device=device)
        rew_b = torch.zeros((T, B), device=device)
        done_b = torch.zeros((T, B), device=device)
        completed = torch.zeros((), device=device)  # on-device: no per-step host sync
        n_done = torch.zeros((), device=device)

        for t in range(T):
            global_step += B
            policy, search_value = mcts.run(obs_raw, mask, add_noise=True)
            obs_b[t], mask_b[t], pol_b[t], val_b[t] = obs_flat, mask, policy, search_value
            temp = cfg.temperature if t < cfg.temp_threshold else 0.0
            action = temperature_sample(policy, temp)
            obs_raw, reward, done, info = pool.step(action)
            rew_b[t], done_b[t] = reward, done.float()
            ep_return = ep_return + reward
            completed = completed + (ep_return * done.float()).sum()
            n_done = n_done + done.float().sum()
            ep_return = ep_return * (1.0 - done.float())
            obs_flat, mask = flatten_obs(obs_raw), info["action_mask"]

        with torch.no_grad():
            _, last_value = pv_fn(obs_flat, mask)
        _, value_target = compute_gae(rew_b, val_b, done_b, last_value, cfg.gamma, cfg.gae_lambda)

        f_obs = obs_b.reshape(T * B, obs_dim)
        f_mask = mask_b.reshape(T * B, A)
        f_pol = pol_b.reshape(T * B, A)
        f_val = value_target.reshape(T * B)
        for _ in range(cfg.learner_steps):
            logits, value = pv_fn(f_obs, f_mask)
            logp = torch.log_softmax(logits, dim=-1)
            policy_loss = -(f_pol * logp).sum(-1).mean()
            value_loss = ((value - f_val) ** 2).mean()
            ent = masked_entropy(logits).mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

        if iteration % cfg.log_every == 0 or iteration == cfg.num_iterations:
            ret = (completed / n_done.clamp_min(1.0)).item()
            sps = global_step / max(time.time() - t0, 1e-6)
            print(
                f"  iter {iteration:4d} | return {ret:+.3f} | loss {loss.item():+.3f} "
                f"| v_loss {value_loss.item():.3f} | {sps:,.0f} steps/s"
            )
    pool.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Maze-v0", choices=("Maze-v0", "BinPack-v0"),
                   help="Maze-v0, or BinPack-v0 (full mask, item/ems policy head)")
    p.add_argument("--maze-size", type=int, default=6)
    p.add_argument("--max-items", type=int, default=12, help="BinPack: max items")
    p.add_argument("--max-ems", type=int, default=16, help="BinPack: max EMS")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=16)
    p.add_argument("--num-simulations", type=int, default=32)
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--learner-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.0, help="0 = 10/num_actions")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp-threshold", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the policy/value net (reduce-overhead)")
    p.add_argument("--smoke", action="store_true", help="tiny run for CI / a quick check")
    cfg = p.parse_args(argv)
    if cfg.smoke:
        cfg.device, cfg.maze_size, cfg.num_envs = "cpu", 5, 8
        cfg.num_steps, cfg.num_simulations, cfg.total_timesteps = 6, 8, 8 * 6 * 3
        cfg.learner_steps, cfg.temp_threshold, cfg.log_every = 2, 3, 1
        cfg.max_items, cfg.max_ems = 6, 8
    cfg.num_iterations = max(cfg.total_timesteps // (cfg.num_steps * cfg.num_envs), 1)
    return cfg


if __name__ == "__main__":
    train(parse_args())
