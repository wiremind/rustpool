"""The package, the compiled module and Cargo.toml (the single source of truth) agree."""

import pathlib
import tomllib

from packaging.version import Version

import rustpool
from rustpool import _rustpool

CARGO = pathlib.Path(__file__).resolve().parents[1] / "Cargo.toml"


def test_versions_match_cargo_toml():
    crate = Version(tomllib.loads(CARGO.read_text())["package"]["version"])
    assert Version(_rustpool.__version__) == crate
    assert Version(rustpool.__version__) == crate
