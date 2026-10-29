import contextlib
import ctypes
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

# Run the suite on a few CPUs, set before anything starts a thread pool. XLA, torch, Rayon and
# rustpool's workers all size themselves to the CPUs this process may use: on the 192-CPU CI
# runner that was ~1,170 threads per process (pytest and each example subprocess) against a
# 1 GiB job memory limit. Restricting the affinity caps them all, and subprocesses inherit it.
# RUSTPOOL_TEST_CPUS overrides the count (0 = no cap).
_TEST_CPUS = int(os.environ.get("RUSTPOOL_TEST_CPUS", "4"))
if _TEST_CPUS > 0 and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:_TEST_CPUS])
# Fewer glibc malloc arenas (one per thread by default). Read at process start, so inherited by
# subprocesses via the env and applied to this process with mallopt (glibc only).
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
with contextlib.suppress(OSError, AttributeError):
    ctypes.CDLL("libc.so.6").mallopt(-8, int(os.environ["MALLOC_ARENA_MAX"]))  # M_ARENA_MAX

import pytest  # noqa: E402

import rustpool  # noqa: E402
from rustpool.envpool_api import _DEFAULT_CONFIGS  # noqa: E402

# The shared fixtures below drive envs through their action masks (playability, invalid-action
# handling). CartPole-v1 emits no mask (both actions are always valid), so it is left out here;
# tests/test_cartpole.py covers it (gymnasium parity, termination, truncation).
UNMASKED_TASKS = frozenset({"CartPole-v1"})


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    params: dict[str, Any]


def _build_task_specs() -> list[TaskSpec]:
    # Every registered masked task, so a new env cannot silently drop out of the generic tests.
    base_params = {"num_envs": 4, "batch_size": 2, "seed": 0}
    return [
        TaskSpec(task_id, {**_DEFAULT_CONFIGS.get(task_id, {}), **base_params})
        for task_id in rustpool.list_envs()
        if task_id not in UNMASKED_TASKS
    ]


TASK_SPECS = _build_task_specs()


@pytest.fixture(params=TASK_SPECS, ids=lambda spec: spec.task_id)
def task_spec(request: pytest.FixtureRequest) -> TaskSpec:
    return request.param


@pytest.fixture
def env_factory(task_spec: TaskSpec) -> Iterator[Callable[..., Any]]:
    created = []

    def _create(**overrides: Any) -> Any:
        params = {**task_spec.params, **overrides}
        pool = rustpool.make(task_spec.task_id, **params)
        created.append(pool)
        return pool

    yield _create

    for pool in reversed(created):
        with contextlib.suppress(Exception):
            pool.close()


@pytest.fixture
def env_pool(env_factory: Callable[..., Any]) -> Any:
    return env_factory()  # env_factory closes it


@pytest.fixture(scope="session")
def external_env(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """``examples/rust/external_env.rs`` built by Cargo as its own extension module and
    imported: a real external env crate (the docs' "Bring your own crate" guide)."""
    import json
    import pathlib
    import shutil
    import subprocess
    import sys
    import sysconfig

    if shutil.which("cargo") is None:
        pytest.skip("cargo not available")
    root = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run(
        ["cargo", "build", "--quiet", "--example", "external_env", "--message-format=json"],
        cwd=root, env={**os.environ, "PYO3_PYTHON": sys.executable}, check=True,
        capture_output=True, text=True).stdout
    built = [a for line in out.splitlines() if line.startswith("{")
             for a in [json.loads(line)]
             if a.get("reason") == "compiler-artifact" and a["target"]["name"] == "external_env"]
    so = next(f for f in built[-1]["filenames"] if f.endswith((".so", ".dylib", ".pyd")))
    mod_dir = tmp_path_factory.mktemp("external_env")
    shutil.copy(so, mod_dir / ("external_env" + sysconfig.get_config_var("EXT_SUFFIX")))
    sys.path.insert(0, str(mod_dir))
    try:
        import external_env
    finally:
        sys.path.remove(str(mod_dir))
    return external_env


# --- Optional stacks are required in CI -------------------------------------------------------
# Tests that need jax / torch / gymnasium / cargo skip when those are missing, which is right
# locally but would let a CI job that failed to install them pass green. With
# RUSTPOOL_REQUIRE_OPTIONAL=1 (set in CI), such a skip is reported as a failure.
_OPTIONAL_SKIP = re.compile(r"could not import|not installed|needs rustpool\[|cargo not available")


def _require_optional(report) -> None:
    if os.environ.get("RUSTPOOL_REQUIRE_OPTIONAL") != "1" or not report.skipped:
        return
    reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if _OPTIONAL_SKIP.search(str(reason)):
        report.outcome = "failed"
        report.longrepr = f"RUSTPOOL_REQUIRE_OPTIONAL=1 but an optional stack is missing: {reason}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _require_optional(outcome.get_result())


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    _require_optional(outcome.get_result())
