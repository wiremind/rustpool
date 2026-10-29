"""Factored vs flat action-space policy-step microbenchmark (PyTorch).

A factored (autoregressive) policy evaluates ``sum(factor_widths)`` action logits per step and
never materialises the flat ``prod(factor_widths)`` masked categorical. This measures the policy
step in isolation — head forward + masked-categorical sample + log-prob — for a flat head vs a
2-factor head, across growing action spaces, to show the compute/​memory the factorization saves.

It does NOT include the env: the flat policy gets its mask "for free" from the obs, while the
factored policy queries one conditional mask per factor from the env (an extra host round-trip
per step). So factored wins end-to-end only once the flat ``prod(widths)`` head/categorical cost
dominates that query overhead — i.e. for genuinely large combinatorial action spaces. The
``candidate logits`` column (``sum`` vs ``prod``) is the implementation-independent metric.

Run:
    python benchmarks/bench_factored.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
from torch_common import FactoredActorCritic, factored_sample, layer_init, mask_logits  # noqa: E402


class FlatPolicy(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden=128):
        super().__init__()
        self.torso = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.head = layer_init(nn.Linear(hidden, num_actions), std=0.01)

    def step(self, obs_flat, mask):
        logits = mask_logits(self.head(self.torso(obs_flat)), mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a)


def _time(fn, device, iters, warmup=10):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/step


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=200)
    cfg = p.parse_args(argv)
    device = torch.device(cfg.device)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    # (max_items, max_ems) pairs -> flat = product, factored candidates = sum.
    shapes = [(25, 50), (64, 128), (128, 256), (128, 512), (256, 512)]
    obs = torch.randn(cfg.batch, cfg.obs_dim, device=device)

    print(f"policy-step microbench | device={device} | batch={cfg.batch} | iters={cfg.iters}\n")
    hdr = f"{'factors':>14} {'flat |A|':>10} {'cand. logits':>13} {'flat ms':>9} {'factored ms':>12} {'speedup':>8}"
    print(hdr)
    print("-" * len(hdr))
    for w0, w1 in shapes:
        flat_n = w0 * w1
        cand = w0 + w1
        flat = FlatPolicy(cfg.obs_dim, flat_n).to(device)
        fac = FactoredActorCritic(cfg.obs_dim, (w0, w1)).to(device)
        flat_mask = torch.ones(cfg.batch, flat_n, dtype=torch.bool, device=device)
        pad = max(w0, w1)
        ones_pad = torch.ones(cfg.batch, pad, dtype=torch.bool, device=device)

        def flat_step(flat=flat, flat_mask=flat_mask):
            with torch.no_grad():
                flat.step(obs, flat_mask)

        def fac_step(fac=fac, ones_pad=ones_pad, w0=w0, w1=w1):
            with torch.no_grad():
                factored_sample(fac, obs, lambda prefix: ones_pad, (w0, w1))

        ms_flat = _time(flat_step, device, cfg.iters)
        ms_fac = _time(fac_step, device, cfg.iters)
        print(
            f"{w0}x{w1:>10} {flat_n:>10,} {cand:>6,} ({flat_n // cand:>3}x) "
            f"{ms_flat:>9.3f} {ms_fac:>12.3f} {ms_flat / ms_fac:>7.1f}x"
        )


if __name__ == "__main__":
    main()
