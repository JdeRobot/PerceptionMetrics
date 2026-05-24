# Centralized logging configuration for PerceptionMetrics.


import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional



# Module-level state


# Root logger name for the entire package.
# All child loggers (perceptionmetrics.models, perceptionmetrics.datasets ...)
# inherit from this automatically — no per-module handler setup needed.
_ROOT = "perceptionmetrics"

# Single formatter reused by every handler
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Guards against re-initialising on repeated imports
_initialised = False


def _init_root() -> None:
    """Set up the perceptionmetrics root logger exactly once."""
    global _initialised
    if _initialised:
        return

    root = logging.getLogger(_ROOT)
    root.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_FORMATTER)
    # Flush console after every record so output is never buffered
    console_handler.terminator = "\n"
    root.addHandler(console_handler)

    # Prevent double output through the Python root logger
    root.propagate = False

    _initialised = True



# Public API


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger under the perceptionmetrics hierarchy.

    Always pass ``__name__`` so log lines show exactly which module they
    came from (e.g. ``perceptionmetrics.datasets.rellis3d``).

    The package root logger is initialised on the first call.
    Subsequent calls return the same logger with no duplicate handlers.
    """
    _init_root()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def set_level(level: int) -> None:
    """Change the log level for the entire perceptionmetrics package.

    Takes effect immediately — no restart needed.
    Affects all child loggers (models, datasets, cli, ...) at once.

    :param level: One of ``logging.DEBUG``, ``logging.INFO``,
        ``logging.WARNING``, ``logging.ERROR``.
    :type level: int
    """
    _init_root()
    logging.getLogger(_ROOT).setLevel(level)


def add_file_handler(
    log_file: str,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> str:
    """Attach a rotating file handler to the perceptionmetrics root logger.

    - Log directory is created automatically if it does not exist.
    - Rotation: once ``log_file`` hits ``max_bytes`` it is renamed to
      ``log_file.1`` and a fresh file starts. Up to ``backup_count``
      backups are kept then discarded.

    :param log_file: Path to write logs to, e.g. ``"logs/run.log"``.
        Resolved relative to the current working directory.
    :type log_file: str
    """
    _init_root()

    root     = logging.getLogger(_ROOT)
    abs_path = os.path.abspath(log_file)

    # Skip if a handler for this exact path already exists
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler):
            if os.path.abspath(h.baseFilename) == abs_path:
                return abs_path

    # Auto-create the log directory
    log_dir = os.path.dirname(abs_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        abs_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=False,   # open the file immediately, not lazily
    )
    file_handler.setFormatter(_FORMATTER)

    # Flush every record immediately — prevents empty file on crash or
    # when reading the file while the process is still running
    file_handler.flush = lambda: (
        file_handler.stream.flush() if file_handler.stream else None
    )

    root.addHandler(file_handler)

    # Print resolved path so user always knows where the file is
    print(
        f"[perceptionmetrics] File logging active → {abs_path}",
        file=sys.stdout,
        flush=True,
    )

    return abs_path