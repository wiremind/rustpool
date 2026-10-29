"""Runs ``alphazero_candidates_checks.py`` (the ``--factored`` AlphaZero candidate-sampler tests)
in a separate pytest process.

Those tests jit the sampler and the network many times; the memory that leaves behind would
otherwise stay in the main pytest process, which then shares the CI job's 1 GiB limit with every
example smoke-test subprocess. A child process returns it all when it exits.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.jax
CHECKS = pathlib.Path(__file__).resolve().parent / "alphazero_candidates_checks.py"


@pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("jax", "flax", "mctx", "optax")),
    reason="needs rustpool[examples] (jax, flax, mctx, optax)",
)
def test_alphazero_candidate_sampler():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(CHECKS)],
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, f"candidate-sampler checks failed:\n{result.stdout[-6000:]}"
