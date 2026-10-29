"""AlphaZero in pure JAX, searching through the *real* rustpool simulator.

This is the showcase for rustpool's headline capability: a batched MCTS whose dynamics model
is the actual (external, non-differentiable) Rust environment, not a learned net and not a
JAX pytree. We drive Google DeepMind's ``mctx`` search and substitute its ``recurrent_fn``
with rustpool's stateless ``simulate_batch``:

* the opaque mctx *embedding* holds ``int32`` handles into rustpool's ``StateRegistry``;
* ``root_fn`` snapshots the live envs (``pool.snapshot``) to seed the search;
* ``recurrent_fn`` forks each node with ``pool.simulate_batch`` (Rayon-parallel on the host)
  and asks the network only for the prior logits + value;
* after each real step we ``release_batch`` the whole search tree's snapshots so the registry
  does not grow without bound.

Targets follow AlphaZero: the policy is trained by cross-entropy to the MCTS visit distribution
and the value by regression to the bootstrapped (GAE) return.

Three modes, every tree edge being one full env action:

* ``--task Maze-v0`` (default): flat action space, flat policy head.
* ``--task BinPack-v0``: the env emits the full ``max_items * max_ems`` action mask; every node
  has all ``max_items * max_ems`` actions as children. The network has an item head and an
  ems-given-item head, composed into the full prior the search needs.
* ``--factored``: BinPack-v0 with a factored action space and **no full mask**. A node cannot
  list its legal full actions, so its children are ``--num-candidates`` full actions sampled
  without replacement from the policy (Gumbel-top-k over item, then ems | item), with legality
  read per factor on the node's stored state (``pool.query_state_mask``). Same network, so the
  two BinPack modes compare search regimes, not architectures.

Run
---
    pip install "rustpool[examples]"
    python examples/alphazero_jax.py --task Maze-v0 --num-simulations 32
    python examples/alphazero_jax.py --task BinPack-v0          # full mask, all actions
    python examples/alphazero_jax.py --factored                 # no full mask, sampled actions
    python examples/alphazero_jax.py --smoke      # tiny, for CI / a quick check
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import math
import pathlib
import sys
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mctx
import optax
from flax.training.train_state import TrainState

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from networks import (  # noqa: E402
    NEG_INF,
    ActorCritic,
    FactoredActorCritic,
    entropy,
    factored_flat_logits,
    mask_logits,
)

import rustpool.jax_api as rj  # noqa: E402


class Transition(NamedTuple):
    obs: dict
    target: Any  # the search's policy target (a mode-specific pytree, see make_*_search)
    search_value: jax.Array  # [B] root value from search
    reward: jax.Array
    done: jax.Array


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    def step(carry, x):
        gae, next_value = carry
        reward, value, done = x
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, value), gae

    _, adv = jax.lax.scan(
        step, (jnp.zeros_like(last_value), last_value), (rewards, values, dones), reverse=True
    )
    return adv + values


def _release_tree(pool, state_ids, tree):
    """Free every snapshot stored in the search tree (root + all expanded children)."""
    visited = tree.node_visits.reshape(-1) > 0
    return pool.release_batch(jnp.where(visited, state_ids.reshape(-1), -1))


# --- flat search: every legal full action is a child (Maze; BinPack with the full mask) -------
def make_flat_search(pool, pv_apply, cfg):
    """Build the mctx root/recurrent fns wired to rustpool's snapshot/simulate_batch.

    ``pv_apply(params, obs, mask) -> (flat_masked_logits, value)`` is the policy/value head —
    a flat ActorCritic, or (BinPack) the item/ems heads composed into the full prior.
    Returns ``(search, loss, last_value)``.
    """

    def root_fn(params, obs, mask, embedding):
        logits, value = pv_apply(params, obs, mask)
        return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=embedding)

    def recurrent_fn(params, rng_key, action, embedding):
        del rng_key
        # embedding = int32[B] registry handles; fork-and-step them through the real sim.
        env_out, new_embedding = pool.simulate_batch(embedding, action.astype(jnp.int32))
        logits, value = pv_apply(params, env_out.obs, env_out.action_mask)
        discount = jnp.where(env_out.done, 0.0, 1.0).astype(jnp.float32) * cfg.gamma
        out = mctx.RecurrentFnOutput(
            reward=env_out.reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return out, new_embedding

    def search(params, rng_key, out):
        obs, mask = out.obs, out.action_mask
        env_ids = jnp.arange(pool.num_envs, dtype=jnp.int32)
        embedding = pool.snapshot(env_ids)  # snapshot live envs -> root handles
        root = root_fn(params, obs, mask, embedding)
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=cfg.num_simulations,
            max_depth=cfg.max_depth,
            invalid_actions=jnp.logical_not(mask),
        )
        tree = policy_output.search_tree
        _release_tree(pool, tree.embeddings, tree)
        target = (mask, policy_output.action_weights)
        return policy_output.action, target, tree.node_values[:, 0]

    def loss(params, obs, target):
        mask, search_policy = target
        logits, value = pv_apply(params, obs, mask)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(search_policy * logp, axis=-1), value, entropy(logits)

    def last_value(params, out):
        return pv_apply(params, out.obs, out.action_mask)[1]

    return search, loss, last_value


# --- factored search: no full mask; each node's children are sampled full actions ------------
class Candidates(NamedTuple):
    """``K`` full actions sampled at one node per env (leading axis ``B``), over ``F`` factors."""

    action: jax.Array  # int32[B, K] flat action (mixed radix over the factors)
    choices: jax.Array  # int32[B, K, F] per-factor choice
    valid: jax.Array  # bool[B, K]: legal (False pads nodes with < K legal actions)
    logp: jax.Array  # float32[B, K] joint log-prob under the factored policy
    first_mask: jax.Array  # bool[B, widths[0]]: the factor-0 mask at the node
    masks: tuple  # per factor k >= 1: bool[B, K, widths[k]] given the candidate's prefix


def _gumbel_with_maximum(phi, target, key, ok):
    """Perturb children log-probs ``phi[..., j]`` with Gumbels conditioned so their max equals the
    parent's perturbed score ``target``: stochastic beam search (Kool et al., 2019), so a top-k
    over the children of the top-k parents samples the joint without replacement."""
    g = jnp.where(ok, phi + jax.random.gumbel(key, phi.shape), NEG_INF)
    z = g.max(axis=-1, keepdims=True)
    t = target[..., None]
    x = g - z  # <= 0
    log1mexp = jnp.where(x > -0.6931, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))
    v = t - g + log1mexp
    return t - jax.nn.relu(v) - jnp.log1p(jnp.exp(-jnp.abs(v)))


def make_candidate_sampler(pool, net, widths, num_candidates):
    """Sample ``K`` distinct legal full actions per node without the full mask.

    Stochastic beam search down the factor chain (e.g. item -> orientation -> container -> ems):
    keep the top-``K`` factor-0 values by Gumbel-perturbed log-prob (mask from
    ``query_state_mask(state, [-1, ...])``), then at each next factor expand only the kept
    prefixes (mask from ``query_state_mask(state, prefix)``) and keep the top ``K`` -- a sample
    without replacement from the factored policy's joint. Cost per node: one mask call per
    factor (``1 + (F-1) K`` rows at most) and ``K`` head calls per factor, instead of the full
    ``prod(widths)`` table.

    Returns ``(expand, heads)``: ``expand(params, state_ids, obs, key) -> (prior_logits[B, K],
    value[B], Candidates)`` and the ``(features, factor_logits, value)`` head fns.
    """
    widths = tuple(int(w) for w in widths)
    num_factors = len(widths)
    num_cand = min(num_candidates, math.prod(widths))

    def features(params, obs):
        return net.apply(params, obs, method=net.features)

    def factor_logits(params, feats, choices, k):
        """Head ``k`` for every kept prefix: ``choices`` int32[B, n, >=k] -> [B, n, widths[k]]."""
        b, n = choices.shape[:2]
        prev = [jax.nn.one_hot(choices[..., j], widths[j]) for j in range(k)]
        prev = jnp.concatenate(prev, axis=-1) if prev else jnp.zeros((b, n, 0))
        return jax.vmap(
            lambda p: net.apply(params, feats, p, k, method=net.factor_logits),
            in_axes=1, out_axes=1,
        )(prev)

    def value(params, feats):
        return net.apply(params, feats, method=net.value)

    def expand(params, state_ids, obs, key):
        """Sample the node's candidates; returns ``(prior_logits[B, K], value[B], Candidates)``."""
        feats = features(params, obs)
        b = feats.shape[0]
        keys = jax.random.split(key, num_factors)
        no_prefix = jnp.full((b, max(num_factors - 1, 1)), -1, jnp.int32)
        first_mask = pool.query_state_mask(state_ids, no_prefix)[:, : widths[0]]
        lp = jax.nn.log_softmax(mask_logits(factor_logits(params, feats, no_prefix[:, None, :0], 0)[:, 0],
                                            first_mask))
        g = jnp.where(first_mask, lp + jax.random.gumbel(keys[0], lp.shape), NEG_INF)
        score, idx = jax.lax.top_k(g, min(num_cand, widths[0]))  # [B, n]
        choices = idx[..., None].astype(jnp.int32)  # [B, n, 1]
        phi = jnp.take_along_axis(lp, idx, axis=1)
        ok = jnp.take_along_axis(first_mask, idx, axis=1)
        masks = ()
        for k in range(1, num_factors):
            n, w = choices.shape[1], widths[k]
            prefix = jnp.concatenate(
                [choices, jnp.full((b, n, num_factors - 1 - k), -1, jnp.int32)], axis=-1
            )
            mk = pool.query_state_mask(
                jnp.repeat(state_ids, n), prefix.reshape(b * n, num_factors - 1)
            )[:, :w].reshape(b, n, w)
            lpk = jax.nn.log_softmax(mask_logits(factor_logits(params, feats, choices, k), mk))
            phi_c, ok_c = phi[..., None] + lpk, ok[..., None] & mk  # [B, n, w]
            g = jnp.where(ok_c, _gumbel_with_maximum(phi_c, score, keys[k], ok_c), NEG_INF)
            score, flat = jax.lax.top_k(g.reshape(b, n * w), min(num_cand, n * w))
            parent, v = flat // w, (flat % w).astype(jnp.int32)
            choices = jnp.concatenate(
                [jnp.take_along_axis(choices, parent[..., None], axis=1), v[..., None]], axis=-1
            )
            phi = jnp.take_along_axis(phi_c.reshape(b, -1), flat, axis=1)
            ok = jnp.take_along_axis(ok_c.reshape(b, -1), flat, axis=1)
            masks = tuple(jnp.take_along_axis(m, parent[..., None], axis=1) for m in masks)
            masks += (jnp.take_along_axis(mk, parent[..., None], axis=1),)
        action = jnp.zeros(choices.shape[:2], jnp.int32)
        for k, w in enumerate(widths):  # mixed radix, factor 0 most significant
            action = action * w + choices[..., k]
        cand = Candidates(action, choices, ok, phi, first_mask, masks)
        return jnp.where(cand.valid, cand.logp, NEG_INF), value(params, feats), cand

    return expand, (features, factor_logits, value)


def make_factored_search(pool, net, widths, cfg):
    """mctx over sampled full actions, reading masks per factor on stored states.

    The env emits no full mask, so a node cannot enumerate its legal full actions; each expanded
    node instead gets ``--num-candidates`` sampled ones (:func:`make_candidate_sampler`). The
    mctx prior over them is their joint log-prob (renormalized over the sample by mctx's
    softmax). Returns ``(search, loss, last_value)``.
    """
    expand, (features, factor_logits, value) = make_candidate_sampler(
        pool, net, widths, cfg.num_candidates
    )

    def recurrent_fn(params, rng_key, action, embedding):
        # embedding = {"sid": int32[B] registry handle, "action": int32[B, K] its candidates}
        flat = jnp.take_along_axis(embedding["action"], action[:, None], axis=1)[:, 0]
        env_out, child = pool.simulate_batch(embedding["sid"], flat)
        logits, value_, cand = expand(params, child, env_out.obs, rng_key)
        discount = jnp.where(env_out.done, 0.0, 1.0).astype(jnp.float32) * cfg.gamma
        out = mctx.RecurrentFnOutput(
            reward=env_out.reward, discount=discount, prior_logits=logits, value=value_
        )
        return out, {"sid": child, "action": cand.action}

    def search(params, rng_key, out):
        env_ids = jnp.arange(pool.num_envs, dtype=jnp.int32)
        roots = pool.snapshot(env_ids)
        k_root, k_search = jax.random.split(rng_key)
        logits, root_value, cand = expand(params, roots, out.obs, k_root)
        root = mctx.RootFnOutput(
            prior_logits=logits, value=root_value, embedding={"sid": roots, "action": cand.action}
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=k_search,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=cfg.num_simulations,
            max_depth=cfg.max_depth,
            invalid_actions=jnp.logical_not(cand.valid),
        )
        tree = policy_output.search_tree
        _release_tree(pool, tree.embeddings["sid"], tree)
        action = jnp.take_along_axis(cand.action, policy_output.action[:, None], axis=1)[:, 0]
        target = (cand.first_mask, cand.choices, cand.masks, policy_output.action_weights)
        return action, target, tree.node_values[:, 0]

    def loss(params, obs, target):
        """Cross-entropy of the factored policy to the visit distribution over the root's
        candidates, under the masks stored at search time."""
        first_mask, choices, masks, weights = target
        feats = features(params, obs)
        l0 = mask_logits(factor_logits(params, feats, choices[:, :1, :0], 0)[:, 0], first_mask)
        logp = jnp.take_along_axis(jax.nn.log_softmax(l0), choices[..., 0], axis=1)  # [N, K]
        ent = entropy(l0)
        for k, mk in enumerate(masks, start=1):
            lk = mask_logits(factor_logits(params, feats, choices, k), mk)  # [N, K, W_k]
            logp = logp + jnp.take_along_axis(
                jax.nn.log_softmax(lk), choices[..., k : k + 1], axis=-1
            )[..., 0]
            ent = ent + jnp.sum(weights * entropy(lk), axis=-1)
        return -jnp.sum(weights * logp, axis=-1), value(params, feats), ent

    def last_value(params, out):
        return value(params, features(params, out.obs))

    return search, loss, last_value


def binpack_factor_widths(cfg):
    """BinPack's action chain item -> [orientation] -> [container] -> ems (single-option stages
    dropped): the env's ``action_factorization()["factor_widths"]``, also for the plain env."""
    c, o = cfg.num_containers, cfg.orientations
    return (cfg.max_items * c, *((o,) if o > 1 else ()), *((c,) if c > 1 else ()), cfg.max_ems)


def make_update_step(pool, search, policy_loss_fn, last_value_fn, cfg):
    def update_step(train_state, env_out, ep_return, key):
        def rollout_body(carry, _):
            ts, out, ep_ret, key = carry
            key, sk = jax.random.split(key)
            action, target, search_value = search(ts.params, sk, out)
            transition = Transition(out.obs, target, search_value, None, None)
            next_out = pool.step(action.astype(jnp.int32))
            ep_ret_new = ep_ret + next_out.reward
            completed = jnp.where(next_out.done, ep_ret_new, 0.0)
            n_done = next_out.done.astype(jnp.float32)
            ep_ret_next = jnp.where(next_out.done, 0.0, ep_ret_new)
            transition = transition._replace(reward=next_out.reward, done=next_out.done)
            return (ts, next_out, ep_ret_next, key), (transition, completed, n_done)

        (train_state, env_out, ep_return, key), (traj, completed, n_done) = jax.lax.scan(
            rollout_body, (train_state, env_out, ep_return, key), None, length=cfg.num_steps
        )

        last_value = last_value_fn(train_state.params, env_out)
        value_targets = compute_gae(
            traj.reward, traj.search_value, traj.done, last_value, cfg.gamma, cfg.gae_lambda
        )

        def loss_fn(params):
            B, T = cfg.num_envs, cfg.num_steps
            flat = jax.tree.map(lambda x: x.reshape((T * B,) + x.shape[2:]), (traj.obs, traj.target))
            policy_loss, value, ent = policy_loss_fn(params, *flat)
            policy_loss, ent = jnp.mean(policy_loss), jnp.mean(ent)
            value_loss = jnp.mean((value - value_targets.reshape(-1)) ** 2)
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent
            return loss, (policy_loss, value_loss, ent)

        def sgd(ts, _):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
            grads = jax.lax.pmean(grads, axis_name="device")  # Anakin; no-op on 1 device
            return ts.apply_gradients(grads=grads), (loss, *aux)

        train_state, metrics = jax.lax.scan(
            sgd, train_state, None, length=cfg.learner_steps
        )
        loss, policy_loss, value_loss, ent = jax.tree.map(jnp.mean, metrics)
        ep_count = jnp.maximum(n_done.sum(), 1.0)
        out_metrics = {
            "episodic_return": completed.sum() / ep_count,
            "episodes": n_done.sum(),
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": ent,
        }
        out_metrics = jax.lax.pmean(out_metrics, axis_name="device")
        return (train_state, env_out, ep_return, key), out_metrics

    return update_step


def train(cfg):
    n_devices = cfg.num_devices or jax.local_device_count()
    cfg.num_updates = max(
        cfg.total_timesteps // (cfg.num_steps * cfg.num_envs * n_devices), 1
    )
    key = jax.random.key(cfg.seed)
    key, net_key = jax.random.split(key)
    task = "BinPack-v0" if cfg.factored else cfg.task

    if task == "BinPack-v0":
        # Item head + ems-given-item head (O(sum widths) params, not a flat O(prod widths) head),
        # shared by both BinPack modes. Plain BinPack-v0 emits the full mask; factored=True
        # (without emit_flat_mask) does not.
        bp_kwargs = {"factored": True} if cfg.factored else {}
        pool = rj.JaxPool("BinPack-v0", num_envs=cfg.num_envs, num_devices=n_devices,
                        seed=cfg.seed, max_items=cfg.max_items, max_ems=cfg.max_ems,
                        num_containers=cfg.num_containers, orientations=cfg.orientations,
                        **bp_kwargs)
        widths = binpack_factor_widths(cfg)
        assert math.prod(widths) == pool.num_actions, (widths, pool.num_actions)
        net = FactoredActorCritic(factor_widths=widths, hidden_sizes=tuple(cfg.hidden_sizes))
        probe = jax.jit(pool.reset)(jax.random.key(0))
        params = net.init(net_key, probe.obs)
        mode_shape = f"C={cfg.num_containers} O={cfg.orientations}, factors {widths}; "
        if cfg.factored:
            search, policy_loss, last_value = make_factored_search(pool, net, widths, cfg)
            mode = (f"factored: no full mask, {min(cfg.num_candidates, math.prod(widths))} "
                    f"sampled actions/node")
        else:
            def pv_apply(params, obs, mask):
                flat_logits, value = factored_flat_logits(net, params, obs, widths)
                return mask_logits(flat_logits, mask), value

            search, policy_loss, last_value = make_flat_search(pool, pv_apply, cfg)
            mode = f"full mask, all {math.prod(widths)} actions/node"
    else:
        env_kwargs = {"width": cfg.maze_size, "height": cfg.maze_size} if task == "Maze-v0" else {}
        pool = rj.JaxPool(task, num_envs=cfg.num_envs, num_devices=n_devices,
                        seed=cfg.seed, **env_kwargs)
        net = ActorCritic(num_actions=pool.num_actions, hidden_sizes=cfg.hidden_sizes)
        probe = jax.jit(pool.reset)(jax.random.key(0))
        params = net.init(net_key, probe.obs, probe.action_mask)

        def pv_apply(params, obs, mask):
            return net.apply(params, obs, mask)

        search, policy_loss, last_value = make_flat_search(pool, pv_apply, cfg)
        mode_shape, mode = "", None
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5)
    )
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)

    def _replicate(x):
        x = jnp.asarray(x)
        return jnp.broadcast_to(x, (n_devices, *x.shape))

    train_state = jax.tree.map(_replicate, train_state)

    update_step = jax.pmap(
        make_update_step(pool, search, policy_loss, last_value, cfg), axis_name="device"
    )

    reset_keys = jax.random.split(jax.random.key(cfg.seed + 1), n_devices)
    env_out = jax.pmap(pool.reset, axis_name="device")(reset_keys)
    ep_return = jnp.zeros((n_devices, cfg.num_envs))
    run_keys = jax.random.split(key, n_devices)

    steps_per_update = cfg.num_steps * cfg.num_envs * n_devices
    print(
        f"AlphaZero on {task}{f' [{mode_shape}{mode}]' if mode else ''}"
        " | "
        f"devices={n_devices}x {jax.default_backend()} | num_envs/device={cfg.num_envs} "
        f"num_steps={cfg.num_steps} sims={cfg.num_simulations} updates={cfg.num_updates}"
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
                f"| loss {m['loss']:+.3f} | v_loss {m['value_loss']:.3f} "
                f"| {sps:,.0f} steps/s"
            )
    pool.close()
    return jax.tree.map(lambda x: float(x[0]), metrics)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Maze-v0", choices=("Maze-v0", "BinPack-v0"),
                   help="Maze-v0, or BinPack-v0 (full mask, every action a child)")
    p.add_argument("--maze-size", type=int, default=8)
    p.add_argument("--factored", action="store_true",
                   help="BinPack-v0 with a factored action space and no full mask: each node's "
                        "children are --num-candidates sampled full actions")
    p.add_argument("--num-candidates", type=int, default=16,
                   help="--factored: full actions sampled per node")
    p.add_argument("--max-items", type=int, default=12, help="BinPack: max items per container")
    p.add_argument("--max-ems", type=int, default=16, help="BinPack: max EMS per container")
    p.add_argument("--num-containers", type=int, default=1, help="BinPack: containers C")
    p.add_argument("--orientations", type=int, default=1, choices=[1, 2, 6],
                   help="BinPack: orientations per item O")
    p.add_argument("--num-devices", type=int, default=0, help="0 = all local devices")
    p.add_argument("--num-envs", type=int, default=64, help="environments per device")
    p.add_argument("--num-steps", type=int, default=16)
    p.add_argument("--num-simulations", type=int, default=32)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--learner-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--smoke", action="store_true", help="tiny run for CI / a quick check")
    cfg = p.parse_args(argv)
    if cfg.smoke:
        cfg.maze_size, cfg.num_envs, cfg.num_steps = 5, 8, 6
        cfg.num_simulations, cfg.total_timesteps, cfg.learner_steps = 4, 8 * 6 * 3, 2
        cfg.log_every, cfg.num_devices = 1, 1
        cfg.max_items, cfg.max_ems, cfg.num_candidates = 6, 8, 8
    # num_updates depends on the resolved device count; computed in train().
    return cfg


if __name__ == "__main__":
    train(parse_args())
