#!/usr/bin/env python
"""Every number in the documentation traces to a file in results/published/.

    python scripts/check_docs.py          # verify; exits non-zero on a mismatch
    python scripts/check_docs.py --fix    # regenerate the embedded tables

Two checks, and the second is the one that earns its keep.

**Generated blocks match their source.**  Documentation embeds tables between
``<!-- table:name -->`` markers.  This regenerates each from
``results/published/`` and compares.  A block that has drifted from its data is
a failure, not a warning.

**Prose numbers exist in the published results.**  Every decimal with three or
more places, outside code fences and outside generated blocks, has to appear in
one of the published JSON files.  This is what stops a README sentence saying
"AUROC 0.9124" six months after the run that produced 0.8817, which is the way
documentation actually goes wrong -- tables get regenerated, sentences do not.

Numbers that are genuinely not results -- a version, a threshold from a config,
a licence year -- go in ``docs/number_allowlist.txt`` with a reason.  Making the
exception explicit and reviewable is the point; a check with a silent escape
hatch is not a check.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import make_tables

from edgeinspect import paths

BLOCK = re.compile(
    r"(?P<open><!--\s*table:(?P<name>[a-z_]+)\s*-->)"
    r"(?P<body>.*?)"
    r"(?P<close><!--\s*/table:(?P=name)\s*-->)",
    re.DOTALL,
)
FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
NUMBER = re.compile(r"(?<![\w.])\d+\.\d{3,}(?![\w.])")
ALLOWLIST = "docs/number_allowlist.txt"


def documentation_files() -> List[Path]:
    files = [paths.ROOT / "README.md"]
    files += sorted(paths.DOCS.glob("*.md"))
    return [f for f in files if f.exists()]


def load_allowlist() -> Dict[str, str]:
    path = paths.ROOT / ALLOWLIST
    allowed: Dict[str, str] = {}
    if not path.exists():
        return allowed
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value, _, reason = line.partition("#")
        allowed[value.strip()] = reason.strip() or "no reason given"
    return allowed


def _walk(node, out: Set[str]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _walk(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, out)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        number = float(node)
        if number != number:
            return
        for digits in range(1, 7):
            out.add("{0:.{1}f}".format(number, digits))
            out.add("{0:.{1}f}".format(abs(number), digits))
    elif isinstance(node, str):
        for match in NUMBER.finditer(node):
            out.add(match.group(0))


def published_numbers() -> Tuple[Set[str], List[str]]:
    """Every number in results/published/, at every rounding a document might use."""
    numbers: Set[str] = set()
    sources: List[str] = []
    for path in sorted(paths.PUBLISHED.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            _walk(json.load(handle), numbers)
        sources.append(paths.relative(path))
    return numbers, sources


def strip_uncheckable(text: str) -> str:
    """Remove code fences, inline code and generated blocks before scanning prose."""
    text = FENCE.sub(" ", text)
    text = BLOCK.sub(" ", text)
    return INLINE_CODE.sub(" ", text)


def check_blocks(fix: bool) -> Tuple[List[str], List[str]]:
    """Verify or rewrite every generated block.  Returns ``(problems, fixed)``."""
    problems: List[str] = []
    fixed: List[str] = []
    for path in documentation_files():
        original = path.read_text(encoding="utf-8")
        updated = original
        for match in BLOCK.finditer(original):
            name = match.group("name")
            if name not in make_tables.TABLES:
                problems.append("{0}: unknown table {1!r}.  Known: {2}".format(
                    paths.relative(path), name, ", ".join(make_tables.table_names())))
                continue
            expected = make_tables.build(name)
            if expected is None:
                problems.append(
                    "{0}: table {1!r} needs results/published/{2}, which does not exist. "
                    "Run the experiment and `make publish`.".format(
                        paths.relative(path), name, make_tables.TABLES[name].source))
                continue
            body = match.group("body").strip()
            if body != expected.strip():
                if fix:
                    replacement = "{0}\n\n{1}\n\n{2}".format(
                        match.group("open"), expected.strip(), match.group("close"))
                    updated = updated.replace(match.group(0), replacement)
                else:
                    problems.append(
                        "{0}: table {1!r} is out of date.  Run "
                        "`python scripts/check_docs.py --fix`.".format(
                            paths.relative(path), name))
        if fix and updated != original:
            path.write_text(updated, encoding="utf-8", newline="\n")
            fixed.append(paths.relative(path))
    return problems, fixed


def check_prose(numbers: Set[str], allowed: Dict[str, str]) -> List[str]:
    problems: List[str] = []
    for path in documentation_files():
        text = strip_uncheckable(path.read_text(encoding="utf-8"))
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in NUMBER.finditer(line):
                token = match.group(0)
                if token in numbers or token in allowed:
                    continue
                problems.append(
                    "{0}:{1}: {2} does not appear in results/published/ and is not in "
                    "{3}".format(paths.relative(path), line_number, token, ALLOWLIST))
    return problems


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="rewrite generated blocks from results/published/")
    parser.add_argument("--skip-prose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    numbers, sources = published_numbers()
    if not sources:
        print("results/published/ is empty.  Run the experiments and `make publish`.",
              file=sys.stderr)
        return 1

    problems, fixed = check_blocks(fix=args.fix)
    for path in fixed:
        print("rewrote " + path)
    if args.fix:
        # Re-read after rewriting so the prose check sees the new blocks.
        problems, _ = check_blocks(fix=False)

    if not args.skip_prose:
        problems += check_prose(numbers, load_allowlist())

    if problems:
        print("\n{0} documentation problem(s):\n".format(len(problems)), file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        return 1

    if not args.quiet:
        checked = ", ".join(paths.relative(p) for p in documentation_files())
        print("docs check passed")
        print("  sources: " + ", ".join(sources))
        print("  files:   " + checked)
    return 0


def iter_documents() -> Iterable[Path]:
    return documentation_files()


if __name__ == "__main__":
    raise SystemExit(main())
