# Centralized logging configuration for PerceptionMetrics.

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# All child loggers (perceptionmetrics.models, perceptionmetrics.datasets, ...)
# inherit from this root automatically — no per-module handler setup needed.
_ROOT = "perceptionmetrics"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_initialised = False  # guards against re-initialising on repeated imports


def _init_root() -> None:
    """Set up the perceptionmetrics root logger exactly once."""
    global _initialised
    if _initialised:
        return

    root = logging.getLogger(_ROOT)
    root.setLevel(logging.INFO)
    root.propagate = False  # avoid double output via the Python root logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_FORMATTER)
    root.addHandler(console_handler)

    _initialised = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger under the perceptionmetrics hierarchy.

    Always pass ``__name__`` so log lines show which module they came from.
    """
    _init_root()
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def set_level(level: int) -> None:
    """Change the log level for the entire perceptionmetrics package."""
    _init_root()
    logging.getLogger(_ROOT).setLevel(level)


def add_file_handler(
    log_file: str,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> str:
    """Attach a rotating file handler to the perceptionmetrics root logger.

    Creates the log directory if needed and skips re-adding a handler for
    a path that's already attached. Returns the resolved absolute path.
    """
    _init_root()

    root = logging.getLogger(_ROOT)
    abs_path = os.path.abspath(log_file)

    for h in root.handlers:
        if (
            isinstance(h, RotatingFileHandler)
            and os.path.abspath(h.baseFilename) == abs_path
        ):
            return abs_path

    log_dir = os.path.dirname(abs_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        abs_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(_FORMATTER)
    root.addHandler(file_handler)
    root.info("File logging active → %s", abs_path)

    return abs_path
