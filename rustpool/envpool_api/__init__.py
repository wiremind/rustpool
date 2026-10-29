"""EnvPool-flavoured convenience layer over the native ``rustpool.RlPool``.

This module is intentionally dependency-light: importing it does **not** import JAX or
PyTorch.
"""

from __future__ import annotations

from typing import Any

from rustpool import _rustpool as _core

# Python-side default kwargs applied per task before forwarding to the Rust constructor.
# (The Rust env configs also have their own serde defaults; these are convenience presets.)
_DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "BinPack-v0": {
        "max_items": 25,
        "max_ems": 50,
    },
    "CartPole-v1": {
        "max_episode_steps": 500,
    },
    "Maze-v0": {
        "width": 10,
        "height": 10,
        "max_episode_steps": 100,
    },
}


def list_envs() -> list[str]:
    """Return the task ids registered in the built-in environment registry."""
    return list(_core.registered_tasks())


def make(task_id: str, **kwargs: Any) -> _core.RlPool:
    """Create an asynchronous environment pool for ``task_id``.

    Merges the task's default config with ``kwargs`` and clamps ``batch_size`` to
    ``num_envs`` when it is missing or invalid (EnvPool behaviour). Returns a
    ``rustpool.RlPool``.

    See ``list_envs`` for the available task ids.
    """
    default_config = _DEFAULT_CONFIGS.get(task_id, {})
    config = {**default_config, **kwargs}

    # EnvPool-like behaviour: fix batch_size if invalid.
    if "num_envs" in config:
        if config.get("batch_size", 0) <= 0 or config["batch_size"] > config["num_envs"]:
            config["batch_size"] = config["num_envs"]

    config["task_id"] = task_id
    return _core.RlPool(**config)


__all__ = ["make", "list_envs"]
