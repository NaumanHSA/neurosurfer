"""Structured logging setup.

A single Rich-backed handler keyed off ``NEUROSURFER_LOG_LEVEL``. Library code
calls :func:`get_logger`; the CLI calls :func:`configure_logging` once at startup.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    lvl = (level or os.environ.get("NEUROSURFER_LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, lvl, logging.INFO)

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich always present, defensive
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    root = logging.getLogger("neurosurfer")
    root.handlers.clear()
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(numeric)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"neurosurfer.{name}")
