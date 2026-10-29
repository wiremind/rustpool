"""Shared network + masking helpers for the rustpool PyTorch examples (PPO, AlphaZero).

The PyTorch counterpart of ``networks.py`` (which serves the JAX examples): a generic
dict-observation encoder, masked-categorical utilities, and small init helpers. Masking uses a
finite ``NEG_INF`` (not ``-inf``) so ``softmax`` / ``log_softmax`` and entropy stay NaN-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Logit value for masked-out actions: large-negative but finite (so 0 * log 0 -> 0, not NaN).
NEG_INF = -1e9


# Observation keys a policy must never see: BinPack's ``ground_truth_locations`` is the
# generator's perfect-packing solution (diagnostics only), so feeding it would leak the answer.
EXCLUDED_OBS_KEYS = frozenset({"ground_truth_locations"})


def flatten_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate the observation leaves into a flat ``[batch, features]`` float tensor.

    Generic across the bundled envs (Maze: ``agent_pos``; BinPack: ``items``/``ems``/masks/...).
    Keys are sorted for a deterministic feature order; ``EXCLUDED_OBS_KEYS`` are dropped.
    """
    leaves = [
        obs[k].reshape(obs[k].shape[0], -1).float() for k in sorted(obs)
        if k not in EXCLUDED_OBS_KEYS
    ]
    return torch.cat(leaves, dim=-1)


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of invalid actions to ``NEG_INF``.

    Falls back to an all-valid row when a row has no valid action (can happen on a
    terminal/auto-reset frame) so the distribution stays normalizable.
    """
    mask = mask.bool()
    any_valid = mask.any(dim=-1, keepdim=True)
    safe = torch.where(any_valid, mask, torch.ones_like(mask))
    neg = torch.tensor(NEG_INF, dtype=logits.dtype, device=logits.device)
    return torch.where(safe, logits, neg)


def masked_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy of already-(finite-)masked ``logits`` (no NaN on masked actions)."""
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1)


class CategoricalMasked(torch.distributions.Categorical):
    """``Categorical`` over action logits with invalid actions masked out.

    Follows Huang & Ontañón (2020), "A Closer Look at Invalid Action Masking in Policy Gradient
    Algorithms": mask before the distribution, and compute entropy over the masked logits.
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        super().__init__(logits=mask_logits(logits, mask))

    def entropy(self) -> torch.Tensor:  # self.logits are the (masked) normalized log-probs
        return masked_entropy(self.logits)


def layer_init(layer: nn.Linear, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init + constant bias (the standard PPO/CleanRL initialization)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class FactoredActorCritic(nn.Module):
    """Autoregressive actor-critic for a mixed-radix (factored) action space.

    A flat action of size ``prod(factor_widths)`` is assembled one factor at a time; factor ``k``
    has its own head conditioned on the one-hots of factors ``0..k-1``. The policy therefore
    evaluates ``sum(factor_widths)`` logits per step instead of the flat ``prod(factor_widths)``
    masked categorical — the whole point of a factored action space.
    """

    def __init__(self, obs_dim, factor_widths, hidden_sizes=(128, 128)):
        super().__init__()
        self.factor_widths = list(factor_widths)
        sizes = (obs_dim, *hidden_sizes)
        layers = []
        for d_in, d_out in zip(sizes[:-1], sizes[1:], strict=True):
            layers += [layer_init(nn.Linear(d_in, d_out)), nn.Tanh()]
        self.torso = nn.Sequential(*layers)
        feat = hidden_sizes[-1] if hidden_sizes else obs_dim
        prev = 0
        heads = []
        for w in self.factor_widths:
            heads.append(layer_init(nn.Linear(feat + prev, w), std=0.01))
            prev += w  # next head also sees the one-hots of all earlier factors
        self.heads = nn.ModuleList(heads)
        self.value_head = layer_init(nn.Linear(feat, 1), std=1.0)

    def features(self, obs_flat):
        return self.torso(obs_flat)

    def factor_logits(self, feats, prev_onehots, k):
        return self.heads[k](torch.cat([feats, prev_onehots], dim=-1))

    def value(self, feats):
        return self.value_head(feats).squeeze(-1)


def compose_flat_action(choices, factor_widths):
    """Mixed-radix encode factor choices into the flat action index (matches the env's decoding)."""
    flat = torch.zeros_like(choices[0])
    radix = 1
    for k in reversed(range(len(factor_widths))):
        flat = flat + choices[k] * radix
        radix *= factor_widths[k]
    return flat


def factored_sample(net, obs, query_mask_fn, factor_widths, *, greedy=False, initial_mask=None):
    """Autoregressive sample under per-factor conditional masks (queried from the env).

    Returns ``(flat_action, total_log_prob, total_entropy, choices[B,K], stored_masks[B,K,pad], value)``.
    ``query_mask_fn(prefix[B,K-1]) -> bool[B,pad]`` is the env's conditional-mask transport.
    ``initial_mask`` (``bool[B,pad]``), when given, is the stage-0 mask the env already returned
    with the observation (``info["action_mask"]`` of a factored ``BinPack-v0`` without the flat
    mask): factor 0 is sampled under it and only the ``K-1`` later stages are queried.
    """
    feats = net.features(obs if torch.is_tensor(obs) else flatten_obs(obs))
    B = feats.shape[0]
    K = len(factor_widths)
    device = feats.device
    prefix = torch.full((B, max(K - 1, 1)), -1, dtype=torch.int32, device=device)
    prev = feats.new_zeros((B, 0))
    choices, logps, ents, masks = [], [], [], []
    for k in range(K):
        if k == 0 and initial_mask is not None:
            mask_pad = initial_mask.to(device=device, dtype=torch.bool)  # [B, pad], from recv
        else:
            mask_pad = query_mask_fn(prefix)  # [B, pad]
        mask_k = mask_pad[:, : factor_widths[k]]
        logits = mask_logits(net.factor_logits(feats, prev, k), mask_k)
        dist = torch.distributions.Categorical(logits=logits)
        a_k = logits.argmax(-1) if greedy else dist.sample()
        choices.append(a_k)
        logps.append(dist.log_prob(a_k))
        ents.append(masked_entropy(logits))
        masks.append(mask_pad)
        prev = torch.cat([prev, torch.nn.functional.one_hot(a_k, factor_widths[k]).float()], -1)
        if k < K - 1:
            prefix[:, k] = a_k.to(torch.int32)
    flat = compose_flat_action(choices, factor_widths)
    return (
        flat.to(torch.int32),
        torch.stack(logps).sum(0),
        torch.stack(ents).sum(0),
        torch.stack(choices, dim=1),
        torch.stack(masks, dim=1),
        net.value(feats),
    )


def factored_flat_logits(net, obs, factor_widths):
    """Compose a 2-factor head into a flat ``[B, W0*W1]`` log-prior (+ value) for tree search.

    AlphaZero/MCTS searches the flat action space, so it needs the materialised flat prior; a
    factored *head* still keeps the parameters ``O(sum widths)`` instead of ``O(prod widths)``.
    """
    assert len(factor_widths) == 2, "factored_flat_logits supports 2 factors (AZ needs flat prior)"
    w0, w1 = factor_widths
    feats = net.features(obs if torch.is_tensor(obs) else flatten_obs(obs))
    b = feats.shape[0]
    item_logp = torch.log_softmax(net.factor_logits(feats, feats.new_zeros((b, 0)), 0), dim=-1)
    # ems logits for every item, batched: repeat feats W0x and pair with each item's one-hot.
    feats_rep = feats.repeat_interleave(w0, dim=0)            # [B*W0, H]
    onehots = torch.eye(w0, device=feats.device).repeat(b, 1)  # [B*W0, W0]
    ems_logp = torch.log_softmax(net.factor_logits(feats_rep, onehots, 1), dim=-1).reshape(b, w0, w1)
    flat_logp = (item_logp[:, :, None] + ems_logp).reshape(b, w0 * w1)
    return flat_logp, net.value(feats)


def factored_log_prob_entropy(net, obs, choices, stored_masks, factor_widths):
    """Teacher-forced recompute of summed factored log-prob/entropy under the STORED masks.

    ``choices`` is ``[B,K]`` and ``stored_masks`` is ``[B,K,pad]`` from sampling time, so the
    masked logits are bit-identical to the rollout. Returns ``(log_prob, entropy, value)``.
    """
    feats = net.features(obs if torch.is_tensor(obs) else flatten_obs(obs))
    prev = feats.new_zeros((feats.shape[0], 0))
    total_logp = torch.zeros(feats.shape[0], device=feats.device)
    total_ent = torch.zeros(feats.shape[0], device=feats.device)
    for k in range(len(factor_widths)):
        logits = mask_logits(net.factor_logits(feats, prev, k), stored_masks[:, k, : factor_widths[k]])
        dist = torch.distributions.Categorical(logits=logits)
        total_logp = total_logp + dist.log_prob(choices[:, k])
        total_ent = total_ent + masked_entropy(logits)
        prev = torch.cat([prev, torch.nn.functional.one_hot(choices[:, k], factor_widths[k]).float()], -1)
    return total_logp, total_ent, net.value(feats)


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Backward GAE over ``[T, B]`` buffers; returns ``(advantages, returns)``.

    ``dones[t]`` marks the episode ending at step ``t`` (rustpool auto-resets, so the bootstrap
    is masked by ``1 - dones[t]``). ``last_value`` bootstraps the final step.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(num_steps)):
        next_value = last_value if t == num_steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        advantages[t] = gae = delta + gamma * gae_lambda * not_done * gae
    return advantages, advantages + values
