"""Smoke tests for the JAX bridge and the example training scripts.

Each example is run as an isolated subprocess with ``--smoke`` (a few env steps) on CPU, so the
base ``pytest`` job stays GPU-free and these are skipped unless the optional stacks are present.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _have(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _run(script: str, *args: str, extra_env: dict[str, str] | None = None) -> str:
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stderr[-3000:]}"
    return result.stdout


jax_stack = pytest.mark.skipif(
    not _have("jax", "flax", "optax"), reason="needs rustpool[examples] (jax/flax/optax)"
)
mctx_stack = pytest.mark.skipif(
    not _have("jax", "flax", "optax", "mctx"), reason="needs rustpool[examples] + mctx"
)
torch_stack = pytest.mark.skipif(not _have("torch"), reason="needs rustpool[torch]")


@pytest.mark.jax
@jax_stack
def test_jax_bridge_primitives():
    """reset/step under jit + snapshot/simulate_batch/release determinism."""
    assert "BRIDGE_OK" in _run("_bridge_check.py")


@pytest.mark.jax
@jax_stack
def test_ppo_smoke():
    assert "update" in _run("ppo_jax.py", "--smoke")


@pytest.mark.jax
@mctx_stack
def test_alphazero_smoke():
    assert "update" in _run("alphazero_jax.py", "--smoke")


@pytest.mark.torch
@torch_stack
def test_ppo_torch_smoke():
    assert "iter" in _run("ppo_torch.py", "--smoke", extra_env={"CUDA_VISIBLE_DEVICES": ""})


@pytest.mark.torch
@torch_stack
def test_alphazero_torch_smoke():
    assert "iter" in _run("alphazero_torch.py", "--smoke", extra_env={"CUDA_VISIBLE_DEVICES": ""})


# --- factored action space (BinPack item->ems) ----------------------------------------
@pytest.mark.jax
@jax_stack
def test_factored_ppo_jax_smoke():
    assert "update" in _run("factored_ppo_jax.py", "--smoke")


@pytest.mark.jax
@mctx_stack
def test_binpack_full_mask_alphazero_jax_smoke():
    assert "update" in _run("alphazero_jax.py", "--smoke", "--task", "BinPack-v0")


@pytest.mark.jax
@mctx_stack
def test_factored_alphazero_jax_smoke():
    """Real factored env (no full mask): sampled full actions per node."""
    out = _run("alphazero_jax.py", "--smoke", "--factored")
    assert "update" in out and "no full mask" in out


@pytest.mark.torch
@torch_stack
def test_factored_ppo_torch_smoke():
    assert "iter" in _run("factored_ppo_torch.py", "--smoke", extra_env={"CUDA_VISIBLE_DEVICES": ""})


@pytest.mark.torch
@torch_stack
def test_binpack_full_mask_alphazero_torch_smoke():
    assert "iter" in _run(
        "alphazero_torch.py", "--smoke", "--task", "BinPack-v0",
        extra_env={"CUDA_VISIBLE_DEVICES": ""},
    )
