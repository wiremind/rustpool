"""Type stubs for the compiled ``rustpool._rustpool`` extension module.

The native module is an implementation detail; prefer the public API in
:mod:`rustpool` (``rustpool.make``, ``rustpool.RlPool``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__version__: str

# Observation batch: one stacked array per observation key (leading axis = batch_size).
Obs = dict[str, npt.NDArray[Any]]
# (obs, reward[float32], done[bool], info). info carries "env_id" (and maybe "action_mask")
# from recv(); simulate_batch() instead returns the children's handles in info["state_id"].
StepReturn = tuple[Obs, npt.NDArray[np.float32], npt.NDArray[np.bool_], dict[str, Any]]

def registered_tasks() -> list[str]:
    """Task ids registered in the built-in environment registry."""
    ...

class RlPool:
    """Pool of native Rust RL environments with an async API + MCTS state substrate.

    Errors: an unknown or mistyped env option, or an array of the wrong dtype (ids, actions
    and state ids are int32, seeds int64), raises TypeError; a negative size, OverflowError;
    an out-of-range option, an unknown or released state_id, a shape mismatch, or a mask query
    on an env with no action factorization, ValueError; an env_id outside [0, num_envs),
    IndexError; a closed pool, RuntimeError. ``recv`` blocks forever when fewer than
    ``batch_size`` results are in flight.
    """

    @property
    def num_envs(self) -> int: ...
    @property
    def batch_size(self) -> int: ...
    # Rust worker threads spawned by this pool (num_threads / RUSTPOOL_NUM_THREADS / CPUs,
    # capped at num_envs).
    @property
    def num_threads(self) -> int: ...
    # How send / query_mask reach the workers: "auto" (default: per worker when a call has more
    # rows than worker threads, else per env), "per_env" or "per_worker". Results are identical.
    @property
    def dispatch(self) -> Literal["auto", "per_env", "per_worker"]: ...
    # Which call returns an episode's reset observation: "same_step" (default: the terminal
    # step itself, with done=True) or "next_step" (EnvPool's: the env's next send).
    @property
    def autoreset(self) -> Literal["same_step", "next_step"]: ...

    def __init__(
        self,
        num_envs: int = 4,
        batch_size: int = 4,
        seed: int = 0,
        task_id: str = "BinPack-v0",
        *,
        num_threads: int | None = None,
        dispatch: str = "auto",
        autoreset: str = "same_step",
        **kwargs: Any,
    ) -> None: ...
    def async_reset(self, seeds: npt.NDArray[np.int64] | None = ...) -> None: ...
    def send(
        self, actions: npt.NDArray[np.int32], env_ids: npt.NDArray[np.int32]
    ) -> None: ...
    def recv(self) -> StepReturn: ...
    def action_spec(self) -> tuple[int, ...]: ...
    def action_mask_width(self) -> int:
        """Width of ``info["action_mask"]``: num_actions (flat mask), pad_width (factored
        env without the flat mask: the stage-0 mask), or 0 (no mask emitted)."""
        ...
    def observation_spec(self) -> dict[str, tuple[list[int], str]]: ...
    def action_factorization(self) -> dict[str, Any] | None: ...
    def snapshot(self, env_ids: npt.NDArray[np.int32]) -> list[int]: ...
    def simulate_batch(
        self, state_ids: npt.NDArray[np.int32], actions: npt.NDArray[np.int32]
    ) -> StepReturn: ...
    def release_batch(self, state_ids: npt.NDArray[np.int32]) -> None: ...
    def query_state(self, env_id: int, key: str) -> Any | None: ...
    def query_mask(
        self, prefixes: npt.NDArray[np.int32], env_ids: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.bool_]: ...
    def query_state_mask(
        self,
        state_ids: npt.NDArray[np.int32],
        prefixes: npt.NDArray[np.int32],
        width: int | None = None,
    ) -> npt.NDArray[np.bool_]: ...
    def close(self) -> None: ...
    def __enter__(self) -> RlPool: ...
    def __exit__(self, *exc_info: Any) -> bool: ...
    def __len__(self) -> int: ...

# The pool's name before 0.2.0 (deprecated).
MyEnvPool = RlPool
