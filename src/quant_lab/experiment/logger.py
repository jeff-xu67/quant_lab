from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", file: str | None = None, **_: object) -> None:
    """Configure structured logging for the framework."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
