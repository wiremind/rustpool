"""Shared network + masking helpers for the rustpool JAX showcases (PPO, AlphaZero).

Deliberately tiny and framework-light: Flax Linen (the style used by PureJaxRL / CleanRL-JAX),
a generic dict-observation encoder, an actor-critic MLP with action masking, and masked
categorical utilities written with ``jax.nn`` (no distrax dependency).
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

# Logit value used for masked-out actions: large-negative, finite (so softmax/log_softmax
# stay well-defined and gradients do not become NaN).
NEG_INF = -1e9


# Observation keys a policy must never see: BinPack's ``ground_truth_locations`` is the
# generator's perfect-packing solution (diagnostics only), so feeding it would leak the answer.
EXCLUDED_OBS_KEYS = frozenset({"ground_truth_locations"})


def flatten_obs(obs: dict[str, jax.Array]) -> jax.Array:
    """Concatenate the observation leaves into a flat ``[batch, features]`` float32 array.

    Generic across our envs (Maze: ``agent_pos``; BinPack: ``items``/``ems``/masks/...).
    Keys are sorted for a deterministic feature order; ``EXCLUDED_OBS_KEYS`` are dropped.
    """
    leaves = [
        jnp.asarray(obs[k], dtype=jnp.float32).reshape(obs[k].shape[0], -1)
        for k in sorted(obs)
        if k not in EXCLUDED_OBS_KEYS
    ]
    return jnp.concatenate(leaves, axis=-1)


class ActorCritic(nn.Module):
    """Shared-torso actor-critic MLP. ``__call__`` returns ``(masked_logits, value)``."""

    num_actions: int
    hidden_sizes: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, obs: dict[str, jax.Array], action_mask: jax.Array):
        x = flatten_obs(obs)
        for size in self.hidden_sizes:
            x = nn.tanh(
                nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            )
        logits = nn.Dense(
            self.num_actions, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        logits = mask_logits(logits, action_mask)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return logits, jnp.squeeze(value, axis=-1)


class FactoredActorCritic(nn.Module):
    """Autoregressive actor-critic for a mixed-radix (factored) action space.

    Factor ``k`` has its own head conditioned on the one-hots of factors ``0..k-1``, so the policy
    evaluates ``sum(factor_widths)`` logits per step instead of the flat ``prod(factor_widths)``
    masked categorical. Use via ``apply(params, ..., method=...)``: ``features`` / ``factor_logits``
    / ``value``; ``__call__`` exists only to initialise every head's parameters.
    """

    factor_widths: Sequence[int]
    hidden_sizes: Sequence[int] = (128, 128)

    def setup(self):
        self.torso = [
            nn.Dense(h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))
            for h in self.hidden_sizes
        ]
        self.heads = [
            nn.Dense(w, kernel_init=nn.initializers.orthogonal(0.01)) for w in self.factor_widths
        ]
        self.value_head = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))

    def features(self, obs):
        x = flatten_obs(obs)
        for layer in self.torso:
            x = nn.tanh(layer(x))
        return x

    def factor_logits(self, feats, prev_onehots, k):
        return self.heads[k](jnp.concatenate([feats, prev_onehots], axis=-1))

    def value(self, feats):
        return jnp.squeeze(self.value_head(feats), axis=-1)

    def __call__(self, obs):
        feats = self.features(obs)
        prev = jnp.zeros((feats.shape[0], 0))
        for k, w in enumerate(self.factor_widths):
            self.factor_logits(feats, prev, k)
            prev = jnp.concatenate([prev, jnp.zeros((feats.shape[0], w))], axis=-1)
        return self.value(feats)


class _AttentionBlock(nn.Module):
    """Single-head attention + residual + LayerNorm, then MLP + residual + LayerNorm
    (Jumanji's post-LN ``TransformerBlock`` with one head)."""

    model_size: int
    mlp_units: int

    @nn.compact
    def __call__(self, query, kv, mask):
        h = nn.MultiHeadDotProductAttention(num_heads=1, qkv_features=self.model_size,
                                            out_features=self.model_size)(query, kv, kv, mask=mask)
        h = nn.LayerNorm()(h + query)
        out = nn.Dense(self.model_size)(nn.relu(nn.Dense(self.mlp_units)(h)))
        return nn.LayerNorm()(out + h)


class BinPackAttention(nn.Module):
    """A minimal version of Jumanji's BinPack actor-critic: one layer of single-head attention.

    Items (their dims) and EMSs (their corners, plus a learned container embedding when
    ``C > 1``) are embedded separately; one layer of self-attention within each set and
    cross-attention between them (attention restricted to unplaced items and live EMSs). The
    trunk is shared; the two head types differ in what they score:

    - flat: ``__call__(obs, mask) -> (masked_logits, value)``, like :class:`ActorCritic`: a
      bilinear score for every (item, EMS) pair (Jumanji's outer-product head), ``I * C * E``
      logits in the env's mixed radix ``i * C*E + c*E + e``;
    - factored (item -> [container] -> EMS), like :class:`FactoredActorCritic`
      (``features`` / ``factor_logits`` / ``value``), never materialising the ``|A|`` grid:
      item scores from the item embeddings (``I``); container given the item, the item against
      each container's mean live EMS (``C``); EMS given item and container, the item against
      that container's EMSs (``E``). ``I + C + E`` scores per step instead of ``I * C * E``.
    """

    factor_widths: Sequence[int]
    model_size: int = 32
    mlp_units: int = 64

    def setup(self):
        widths = [int(w) for w in self.factor_widths]
        self.num_items, self.num_ems = widths[0], widths[-1]
        self.num_containers = widths[1] if len(widths) == 3 else 1
        d = self.model_size
        self.item_proj = nn.Dense(d)
        self.ems_proj = nn.Dense(d)
        self.container_emb = nn.Embed(self.num_containers, d)
        self.ems_self = _AttentionBlock(d, self.mlp_units)
        self.items_self = _AttentionBlock(d, self.mlp_units)
        self.ems_cross = _AttentionBlock(d, self.mlp_units)
        self.items_cross = _AttentionBlock(d, self.mlp_units)
        self.policy_items = nn.Dense(d)
        self.policy_ems = nn.Dense(d)
        self.item_head = nn.Dense(1)
        self.value_hidden = nn.Dense(d)
        self.value_out = nn.Dense(1)

    def _embed(self, obs):
        ems_ok = jnp.asarray(obs["ems_mask"], jnp.bool_)
        item_ok = (jnp.asarray(obs["items_mask"], jnp.bool_)
                   & ~jnp.asarray(obs["placed_mask"], jnp.bool_))
        container = jnp.repeat(jnp.arange(self.num_containers), self.num_ems)
        ems = self.ems_proj(jnp.asarray(obs["ems"], jnp.float32)) + self.container_emb(container)
        items = self.item_proj(jnp.asarray(obs["items"], jnp.float32))
        return ems, items, ems_ok, item_ok

    def _encode(self, ems, items, ems_ok, item_ok):
        def pair(q_ok, k_ok):  # [B, 1 head, Tq, Tk]
            return (q_ok[:, :, None] & k_ok[:, None, :])[:, None]

        ems = self.ems_self(ems, ems, pair(ems_ok, ems_ok))
        items = self.items_self(items, items, pair(item_ok, item_ok))
        return (self.ems_cross(ems, items, pair(ems_ok, item_ok)),
                self.items_cross(items, ems, pair(item_ok, ems_ok)))

    def features(self, obs):
        """``[B, C*E + I, d + 1]``: EMS then item embeddings, last channel = token is live."""
        ems, items, ems_ok, item_ok = self._embed(obs)
        ems, items = self._encode(ems, items, ems_ok, item_ok)
        live = jnp.concatenate([ems_ok, item_ok], axis=1)[..., None].astype(jnp.float32)
        return jnp.concatenate([jnp.concatenate([ems, items], axis=1), live], axis=-1)

    def _split(self, feats):
        ce = self.num_containers * self.num_ems
        emb, live = feats[..., :-1], feats[..., -1] > 0.5
        return emb[..., :ce, :], emb[..., ce:, :], live[..., :ce], live[..., ce:]

    def pair_scores(self, feats):
        """``[B, I, C*E]`` scores, ``NEG_INF`` on dead EMSs."""
        ems, items, ems_ok, _ = self._split(feats)
        scores = jnp.einsum("...ik,...ek->...ie", self.policy_items(items), self.policy_ems(ems))
        return jnp.where(ems_ok[..., None, :], scores, NEG_INF)

    def factor_logits(self, feats, prev_onehots, k):
        ems, items, ems_ok, _ = self._split(feats)
        n_i, n_c, n_e = self.num_items, self.num_containers, self.num_ems
        if k == 0:
            return jnp.squeeze(self.item_head(items), axis=-1)  # [B, I]
        item = self.policy_items(  # the chosen item's embedding, [B, d]
            jnp.einsum("...i,...id->...d", prev_onehots[..., :n_i], items))
        ems = ems.reshape(*ems.shape[:-2], n_c, n_e, ems.shape[-1])
        ems_ok = ems_ok.reshape(*ems_ok.shape[:-1], n_c, n_e)
        if n_c > 1 and k == 1:  # container | item: against each container's mean live EMS
            live = ems_ok[..., None].astype(ems.dtype)
            pooled = jnp.sum(ems * live, axis=-2) / jnp.maximum(jnp.sum(live, axis=-2), 1.0)
            return jnp.einsum("...d,...cd->...c", item, self.policy_ems(pooled))
        if n_c > 1:  # EMS | item, container: only the chosen container's EMSs
            onehot_c = prev_onehots[..., n_i:n_i + n_c]
            ems = jnp.einsum("...c,...ced->...ed", onehot_c, ems)
            ems_ok = jnp.einsum("...c,...ce->...e", onehot_c, ems_ok.astype(jnp.float32)) > 0.5
        else:
            ems, ems_ok = ems[..., 0, :, :], ems_ok[..., 0, :]
        scores = jnp.einsum("...d,...ed->...e", item, self.policy_ems(ems))
        return jnp.where(ems_ok, scores, NEG_INF)

    def value(self, feats):
        ems, items, ems_ok, item_ok = self._split(feats)
        def mean(x, ok):  # over live tokens (Jumanji sums; a mean keeps the scale fixed)
            return jnp.sum(x * ok[..., None], axis=-2) / jnp.maximum(
                jnp.sum(ok, axis=-1, keepdims=True), 1)

        pooled = jnp.concatenate([mean(ems, ems_ok), mean(items, item_ok)], axis=-1)
        return jnp.squeeze(self.value_out(nn.relu(self.value_hidden(pooled))), axis=-1)

    def __call__(self, obs, action_mask=None):
        feats = self.features(obs)
        self.factor_logits(feats, None, 0)  # creates the factored item head's params at init
        logits = self.pair_scores(feats).reshape(feats.shape[0], -1)
        if action_mask is not None:
            logits = mask_logits(logits, action_mask)
        return logits, self.value(feats)


class BinPackLatent(BinPackAttention):
    """:class:`BinPackAttention` with a trunk linear in the number of tokens: ``num_latents``
    learned tokens attend over every live item and EMS (one single-head block), then every item
    and EMS attends back to them (another). Cost ``(I + C*E) * num_latents`` per layer instead
    of the ``(I + C*E)^2``-order full attention, so at large ``C`` the flat head's ``I * C*E``
    scores, not the shared trunk, dominate a flat policy's cost. Same heads as the parent."""

    num_latents: int = 16

    def setup(self):
        super().setup()
        self.latents = self.param("latents", nn.initializers.normal(0.02),
                                  (self.num_latents, self.model_size))
        self.to_latents = _AttentionBlock(self.model_size, self.mlp_units)
        self.from_latents = _AttentionBlock(self.model_size, self.mlp_units)

    def _encode(self, ems, items, ems_ok, item_ok):
        tokens = jnp.concatenate([ems, items], axis=1)
        ok = jnp.concatenate([ems_ok, item_ok], axis=1)
        b, t = ok.shape
        lat = jnp.broadcast_to(self.latents, (b, self.num_latents, self.model_size))
        lat = self.to_latents(lat, tokens,
                              jnp.broadcast_to(ok[:, None, None, :], (b, 1, self.num_latents, t)))
        tokens = self.from_latents(tokens, lat,
                                   jnp.ones((b, 1, t, self.num_latents), jnp.bool_))
        n_ems = ems.shape[1]
        return tokens[:, :n_ems], tokens[:, n_ems:]


def compose_flat_action(choices, factor_widths) -> jax.Array:
    """Mixed-radix encode factor choices into the flat action index (matches the env's decoding)."""
    flat = jnp.zeros_like(choices[0])
    radix = 1
    for k in reversed(range(len(factor_widths))):
        flat = flat + choices[k] * radix
        radix *= factor_widths[k]
    return flat


def factored_sample(
    net, params, obs, query_mask_fn, factor_widths, key, initial_mask=None, greedy=False
):
    """Autoregressive sample under per-factor conditional masks queried from the env.

    ``query_mask_fn(prefix[B, K-1]) -> bool[B, pad]``. ``initial_mask`` (``bool[B, pad]``), when
    given, is the stage-0 mask the env already returned with the observation (the
    ``action_mask`` of a factored ``BinPack-v0`` without the flat mask): factor 0 is then
    sampled under it and only the ``K-1`` later stages are queried. ``greedy`` takes each
    factor's arg-max instead of sampling (``key`` is then unused). Returns
    ``(flat_action, total_log_prob, choices[B,K], stored_masks[B,K,pad], value)``.
    """
    feats = net.apply(params, obs, method=net.features)
    b = feats.shape[0]
    num_factors = len(factor_widths)
    keys = jax.random.split(key, num_factors)
    prefix = jnp.full((b, max(num_factors - 1, 1)), -1, dtype=jnp.int32)
    prev = jnp.zeros((b, 0))
    choices, logps, masks = [], [], []
    for k in range(num_factors):
        if k == 0 and initial_mask is not None:
            mask_pad = jnp.asarray(initial_mask, dtype=jnp.bool_)  # [B, pad], from recv
        else:
            mask_pad = query_mask_fn(prefix)  # [B, pad]
        logits = mask_logits(
            net.apply(params, feats, prev, k, method=net.factor_logits),
            mask_pad[:, : factor_widths[k]],
        )
        if greedy:
            a_k = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        else:
            a_k = sample_action(keys[k], logits)
        choices.append(a_k)
        logps.append(log_prob(logits, a_k))
        masks.append(mask_pad)
        prev = jnp.concatenate([prev, jax.nn.one_hot(a_k, factor_widths[k])], axis=-1)
        if k < num_factors - 1:
            prefix = prefix.at[:, k].set(a_k.astype(jnp.int32))
    return (
        compose_flat_action(choices, factor_widths).astype(jnp.int32),
        sum(logps),
        jnp.stack(choices, axis=1),
        jnp.stack(masks, axis=1),
        net.apply(params, feats, method=net.value),
    )


def factored_flat_logits(net, params, obs, factor_widths):
    """Compose the factored heads into the flat ``[B, prod(widths)]`` log-prior (+ value).

    A flat-action tree search (mctx over the full action space) needs the materialised flat
    prior. A factored *head* still helps: its parameters are ``O(sum widths)`` instead of a flat
    ``O(prod widths)`` head. Any number of factors; flat index order is the mixed radix with
    factor 0 most significant (the env's encoding).

    Every head is a single ``Dense`` over ``concat([feats, one_hot(prefix)...])``, i.e. affine,
    so ``head_k(feats, prefix) = head_k(feats, 0) + sum_j E_kj[prefix_j]`` with
    ``E_kj[v] = head_k(0, e_v in slot j) - head_k(0, 0)``. That builds the prior with one
    ``[B, w_k]`` head call per factor plus a few ``[w_j, w_k]`` tables -- never a head input per
    action prefix, which for 3+ factors would be ``B * |A| * sum(widths)`` floats.
    """
    widths = [int(w) for w in factor_widths]
    feats = net.apply(params, obs, method=net.features)
    b, hidden = feats.shape

    def head(k, f, prev):
        return net.apply(params, f, prev, k, method=net.factor_logits)

    logp = jnp.zeros((b,))
    for k, w in enumerate(widths):
        prev_dim = sum(widths[:k])
        logits = head(k, feats, jnp.zeros((b, prev_dim))).reshape((b,) + (1,) * k + (w,))
        if k:
            zero = head(k, jnp.zeros((1, hidden)), jnp.zeros((1, prev_dim)))  # [1, w]
            off = 0
            for j, wj in enumerate(widths[:k]):
                onehots = jnp.zeros((wj, prev_dim)).at[:, off:off + wj].set(jnp.eye(wj))
                table = head(k, jnp.zeros((wj, hidden)), onehots) - zero  # [w_j, w]
                shape = (1,) + tuple(wj if a == j else 1 for a in range(k)) + (w,)
                logits = logits + table.reshape(shape)
                off += wj
        logp = logp[..., None] + jax.nn.log_softmax(logits, axis=-1)
    return logp.reshape(b, -1), net.apply(params, feats, method=net.value)


def factored_log_prob_entropy(net, params, obs, choices, stored_masks, factor_widths):
    """Teacher-forced recompute of summed factored log-prob / entropy under the STORED masks."""
    feats = net.apply(params, obs, method=net.features)
    prev = jnp.zeros((feats.shape[0], 0))
    total_logp = jnp.zeros(feats.shape[0])
    total_ent = jnp.zeros(feats.shape[0])
    for k in range(len(factor_widths)):
        logits = mask_logits(
            net.apply(params, feats, prev, k, method=net.factor_logits),
            stored_masks[:, k, : factor_widths[k]],
        )
        total_logp = total_logp + log_prob(logits, choices[:, k])
        total_ent = total_ent + entropy(logits)
        prev = jnp.concatenate([prev, jax.nn.one_hot(choices[:, k], factor_widths[k])], axis=-1)
    return total_logp, total_ent, net.apply(params, feats, method=net.value)


def mask_logits(logits: jax.Array, action_mask: jax.Array) -> jax.Array:
    """Set logits of invalid actions to ``NEG_INF``.

    If a row has no valid action (can happen on a terminal/auto-reset frame), fall back to an
    all-valid row so the distribution stays normalizable.
    """
    mask = jnp.asarray(action_mask, dtype=jnp.bool_)
    any_valid = jnp.any(mask, axis=-1, keepdims=True)
    safe_mask = jnp.where(any_valid, mask, jnp.ones_like(mask))
    return jnp.where(safe_mask, logits, NEG_INF)


def sample_action(key: jax.Array, logits: jax.Array) -> jax.Array:
    """Sample one categorical action per row from (already masked) ``logits``."""
    return jax.random.categorical(key, logits).astype(jnp.int32)


def log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    """Log-probability of ``actions`` under categorical ``logits``."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, actions[..., None].astype(jnp.int32), axis=-1)[..., 0]


def entropy(logits: jax.Array) -> jax.Array:
    """Entropy of the categorical distribution defined by ``logits``."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)
