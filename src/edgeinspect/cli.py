"""``edgeinspect`` -- one entry point for the scripts.

The scripts under ``scripts/`` are the real interface and each is runnable on
its own with ``python scripts/<name>.py``; that is what the README, the Makefile
and the runbook use, because a command someone can read, copy and modify beats a
subcommand whose implementation they then have to go and find.

This dispatcher exists for the installed package, where ``scripts/`` is not on
disk next to the module.  It executes the script by path in the current process,
so ``edgeinspect train --epochs 4`` and ``python scripts/train.py --epochs 4``
are the same code with the same arguments and the same exit code.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__, paths

COMMANDS: Dict[str, str] = {
    "dataset": "make_dataset.py",
    "train": "train.py",
    "qat": "qat.py",
    "export": "export_onnx.py",
    "quantise": "quantise.py",
    "engine": "export_tensorrt.py",
    "evaluate": "evaluate.py",
    "benchmark": "benchmark_latency.py",
    "edge": "simulate_edge.py",
    "drift": "detect_drift.py",
    "deploy": "deploy.py",
    "loop": "run_loop.py",
    "ablate": "ablate_quantisation.py",
    "tables": "make_tables.py",
    "check-docs": "check_docs.py",
}

PIPELINES: Dict[str, str] = {
    "retrain": "retrain.py",
}


def _usage() -> str:
    lines = ["edgeinspect {0}".format(__version__), "",
             "usage: edgeinspect <command> [options]", "", "commands:"]
    for name, script in sorted(COMMANDS.items()):
        lines.append("  {0:<12} scripts/{1}".format(name, script))
    for name, script in sorted(PIPELINES.items()):
        lines.append("  {0:<12} pipelines/{1}".format(name, script))
    lines += ["", "Every command forwards its arguments unchanged; try",
              "  edgeinspect train --help"]
    return "\n".join(lines)


def _resolve(command: str) -> Optional[Path]:
    if command in COMMANDS:
        return paths.ROOT / "scripts" / COMMANDS[command]
    if command in PIPELINES:
        return paths.ROOT / "pipelines" / PIPELINES[command]
    return None


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0

    command, rest = argv[0], argv[1:]
    target = _resolve(command)
    if target is None:
        print("unknown command {0!r}\n".format(command), file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    if not target.exists():
        print("cannot find {0}.  The scripts live in the repository, not in the "
              "installed package; run this from a checkout.".format(paths.relative(target)),
              file=sys.stderr)
        return 2

    # The scripts import _bootstrap from their own directory.
    script_dir = str(target.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    saved = sys.argv
    sys.argv = [str(target)] + rest
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
