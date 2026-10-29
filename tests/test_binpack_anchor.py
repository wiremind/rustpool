"""Anchor: BinPack-v0 at its defaults must stay bit-identical across env refactors.

Seeded trajectories (64 envs x 300 steps, auto-reset included) are hashed -- every obs key,
reward, done flag, flat mask, factor masks and the specs -- and compared with hashes recorded
from the pre-refactor env (``tests/data/binpack_anchor.json``). Regenerate ONLY when a
behaviour change is intended:

    python tests/test_binpack_anchor.py --regen
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import rustpool

FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / "binpack_anchor.json"
NUM_ENVS, STEPS = 64, 300


class _Hasher:
    def __init__(self):
        self.h = hashlib.sha256()

    def add(self, x):
        if isinstance(x, dict):
            for k in sorted(x):
                self.h.update(k.encode())
                self.add(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                self.add(v)
        elif isinstance(x, np.ndarray):
            self.h.update(str((x.dtype.str, x.shape)).encode())
            self.h.update(np.ascontiguousarray(x).tobytes())
        else:
            self.h.update(repr(x).encode())

    def hexdigest(self):
        return self.h.hexdigest()


def _specs(pool):
    return {
        "obs": {k: (list(s), d) for k, (s, d) in pool.observation_spec().items()},
        "action": list(pool.action_spec()),
        "factorization": pool.action_factorization(),
        "mask_width": pool.action_mask_width(),
    }


def _pick(rng, mask_row):
    legal = np.flatnonzero(mask_row)
    return int(rng.choice(legal)) if legal.size else 0


def run_pool(kwargs, chooser):
    """Hash a seeded synchronous rollout. ``chooser`` is "flat" (random legal action from the
    flat mask) or "factored" (item from query_mask([-1]), then ems from query_mask([item]))."""
    pool = rustpool.make("BinPack-v0", num_envs=NUM_ENVS, batch_size=NUM_ENVS, seed=7, **kwargs)
    h = _Hasher()
    h.add(_specs(pool))
    rng = np.random.default_rng(123)
    ids = np.arange(NUM_ENVS, dtype=np.int32)
    pool.async_reset(np.arange(NUM_ENVS, dtype=np.int64) + 1000)
    obs, rew, done, info = pool.recv()
    total_reward, episodes = 0.0, 0
    me = kwargs.get("max_ems", 50)
    for _ in range(STEPS + 1):
        mask = info.get("action_mask")
        h.add([obs, rew, done, np.asarray(info["env_id"]), mask])
        total_reward += float(np.sum(rew))
        episodes += int(np.sum(done))
        if chooser == "flat":
            actions = np.array([_pick(rng, m) for m in mask], np.int32)
        else:
            m0 = pool.query_mask(np.full((NUM_ENVS, 1), -1, np.int32), ids)
            items = np.array([_pick(rng, m) for m in m0], np.int32)
            m1 = pool.query_mask(items[:, None], ids)
            ems = np.array([_pick(rng, m) for m in m1], np.int32)
            h.add([m0, m1])
            actions = items * me + ems
        pool.send(actions, ids)
        obs, rew, done, info = pool.recv()
    pool.close()
    return {"sha256": h.hexdigest(), "total_reward": round(total_reward, 4), "episodes": episodes}


CASES = {
    "flat": ({}, "flat"),
    "flat_sparse": ({"reward_type": "sparse"}, "flat"),
    "factored": ({"factored": True}, "factored"),
    "factored_with_flat_mask": ({"factored": True, "emit_flat_mask": True}, "factored"),
}


def compute_all():
    return {name: run_pool(kw, chooser) for name, (kw, chooser) in CASES.items()}


@pytest.fixture(scope="module")
def expected():
    return json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("name", list(CASES))
def test_binpack_defaults_bit_identical(name, expected):
    kw, chooser = CASES[name]
    assert run_pool(kw, chooser) == expected[name]


if __name__ == "__main__" and "--regen-case" in sys.argv:
    # Regenerate ONE case after an intended behaviour change, keeping the others' hashes, and
    # record why: python tests/test_binpack_anchor.py --regen-case NAME "reason"
    i = sys.argv.index("--regen-case")
    name, reason = sys.argv[i + 1], sys.argv[i + 2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    doc = json.loads(FIXTURE.read_text())
    kw, chooser = CASES[name]
    doc["cases"][name] = run_pool(kw, chooser)
    doc.setdefault("regenerated_cases", {})[name] = {"on_top_of_commit": commit.strip(),
                                                    "reason": reason}
    FIXTURE.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["cases"][name], indent=1))

if __name__ == "__main__" and "--regen" in sys.argv:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain", "src/envs/binpack"], capture_output=True,
                           text=True).stdout
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps({
        "generated_from_commit": commit.strip(),
        "binpack_src_dirty_at_generation": bool(dirty.strip()),
        "num_envs": NUM_ENVS,
        "steps": STEPS,
        "cases": compute_all(),
    }, indent=1) + "\n")
    print(FIXTURE.read_text())
