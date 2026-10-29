"""Runs ``jax_multidevice_checks.py`` (JaxPool under pmap on two forced host devices) in a
child process: the device count has to be set before JAX is imported."""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.jax
CHECKS = pathlib.Path(__file__).resolve().parent / "jax_multidevice_checks.py"


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="needs rustpool[jax]")
def test_jax_pool_on_two_devices():
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
           "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(CHECKS)],
        capture_output=True, text=True, timeout=600, env=env, check=False,
    )
    assert result.returncode == 0, f"multi-device checks failed:\n{result.stdout[-6000:]}"
