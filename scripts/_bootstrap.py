"""Put ``src/`` on the path.

Imported first by every script so that ``python scripts/train.py`` works
without ``pip install -e .`` and without a ``PYTHONPATH`` that has to be
remembered.  The Makefile sets ``PYTHONPATH=src`` as well; this is the belt to
that pair of braces, and it is what makes the commands in the README copyable.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
