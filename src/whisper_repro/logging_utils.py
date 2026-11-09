"""Logging setup.

One configuration call at the CLI entry point; every module then uses
`get_logger(__name__)`. Logs go to stderr and to a rotating file under
the data root, so a crashed overnight run leaves a trace you can read.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .paths import DATA_ROOT

_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure(verbose: bool = False) -> None:
    root = logging.getLogger("whisper_repro")
    if root.handlers:  # already configured (e.g. called twice in a notebook)
        return
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(stream)

    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(
            DATA_ROOT / "run.log", maxBytes=5_000_000, backupCount=3,
            encoding="utf-8",
        )
        fileh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        fileh.setLevel(logging.DEBUG)
        root.addHandler(fileh)
    except OSError:  # read-only filesystem; stderr logging still works
        root.warning("could not open run.log; logging to stderr only")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("whisper_repro") else f"whisper_repro.{name}")
