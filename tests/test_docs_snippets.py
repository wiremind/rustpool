"""Every code snippet on the documentation site and in the README runs against the current tree.

- Python: the ``python`` blocks of each page in ``docs/`` (and of README.md) run in order, in one namespace per
  page (a page reads like a notebook). A block preceded by ``<!-- docs-test: skip -->`` is not
  run. A page that imports JAX or PyTorch is skipped when that stack is missing; the "Bring your
  own crate" page runs against the example crate built by the ``external_env`` fixture.
- Rust: the hand-written signatures on ``docs/rust-api.md`` must appear in ``src/``. (The other
  Rust blocks are included from compiled files with ``--8<--``, so they cannot drift.)
"""

import importlib.util
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SKIP_MARK = "<!-- docs-test: skip -->"


def blocks(page, lang):
    """The ``lang`` fenced blocks of a page, minus those marked to skip."""
    out, lines, i = [], page.read_text().splitlines(), 0
    while i < len(lines):
        if lines[i].strip() == f"```{lang}":
            skip = i > 0 and lines[i - 1].strip() == SKIP_MARK
            j = i + 1
            while lines[j].strip() != "```":
                j += 1
            if not skip:
                out.append((i + 1, "\n".join(lines[i + 1:j])))
            i = j
        i += 1
    return out


PAGES = sorted(p for p in [*DOCS.rglob("*.md"), ROOT / "README.md"] if blocks(p, "python"))


def test_every_page_with_code_is_collected():
    names = {p.relative_to(ROOT).as_posix() for p in PAGES}
    assert {"README.md", "docs/index.md", "docs/quickstart.md", "docs/concepts/autoreset.md",
            "docs/concepts/mcts.md", "docs/guides/own-crate.md"} <= names


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_python_snippets_run(page, request, capsys):
    code = "\n".join(src for _, src in blocks(page, "python"))
    for mod in ("jax", "torch"):
        if re.search(rf"^\s*(import|from) {mod}\b", code, re.M) and \
                importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not installed")
    if "import external_env" in code:
        sys.modules.setdefault("external_env", request.getfixturevalue("external_env"))
    namespace = {"__name__": f"docs_{page.stem}"}
    for line, src in blocks(page, "python"):
        try:
            exec(compile(src, f"{page.relative_to(ROOT)}:{line}", "exec"), namespace)
        except Exception as err:  # name the block that failed
            raise AssertionError(f"{page.relative_to(ROOT)}, block at line {line}: {err!r}") \
                from err
    for value in list(namespace.values()):  # close what the page left open
        close = getattr(value, "close", None)
        if callable(close) and type(value).__module__.startswith(("rustpool", "external_env")):
            close()


def _normalize(text):
    text = re.sub(r"\s+", "", text)
    return text.replace(",)", ")").replace(",}", "}")


def test_rust_api_signatures_match_the_source():
    source = _normalize("\n".join(p.read_text() for p in (ROOT / "src").rglob("*.rs")))
    checked = 0
    for _, block in blocks(DOCS / "rust-api.md", "rust"):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line in ("}", "{"):
                continue
            needle = _normalize(line).rstrip(";,")
            assert needle in source, f"docs/rust-api.md: {line!r} not found in src/"
            checked += 1
    assert checked > 30
