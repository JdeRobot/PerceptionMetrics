"""Centralized logging configuration for PerceptionMetrics.

No external dependencies — pure Python stdlib only.

Basic usage in any module:
    from perceptionmetrics.utils.logging_config import get_logger
    _logger = get_logger(__name__)
    _logger.info("Samples retrieved: %d", n)
    _logger.warning("Missing file for sample: %s", name)
    _logger.error("Failed to load ontology: %s", e)

To add file logging anywhere in your code:
    from perceptionmetrics.utils.logging_config import add_file_handler
    add_file_handler("logs/run.log")

To change log level anywhere in your code:
    import logging
    from perceptionmetrics.utils.logging_config import set_level
    set_level(logging.DEBUG)    # show everything
    set_level(logging.WARNING)  # show only problems
    set_level(logging.ERROR)    # show only failures
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger under the perceptionmetrics hierarchy.

    Always pass ``__name__`` so log lines show exactly which module they
    came from (e.g. ``perceptionmetrics.datasets.rellis3d``).

    The package root logger is initialised on the first call.
    Subsequent calls return the same logger with no duplicate handlers.

    :param name: Logger name — always pass ``__name__``.
    :type name: str
    :param level: Optional level override for this logger only.
        Use ``set_level()`` to change level for the whole package.
    :type level: int, optional
    :return: Configured logger.
    :rtype: logging.Logger

    Example::

        from perceptionmetrics.utils.logging_config import get_logger
        _logger = get_logger(__name__)
        _logger.info("Samples retrieved: %d", n)
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

    Example::

        import logging
        from perceptionmetrics.utils.logging_config import set_level

        set_level(logging.DEBUG)    # see everything
        set_level(logging.WARNING)  # see only problems
        set_level(logging.ERROR)    # see only failures
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
    - Safe to call multiple times — duplicate handlers for the same path
      are never added.
    - Flushes every record immediately so nothing is lost on crash.
    - Prints the resolved absolute path so you always know where the
      log file is being written.
    - Rotation: once ``log_file`` hits ``max_bytes`` it is renamed to
      ``log_file.1`` and a fresh file starts. Up to ``backup_count``
      backups are kept then discarded.

    :param log_file: Path to write logs to, e.g. ``"logs/run.log"``.
        Resolved relative to the current working directory.
    :type log_file: str
    :param max_bytes: File size limit before rotation. Default 5 MB.
    :type max_bytes: int
    :param backup_count: Number of backup files to keep. Default 3.
    :type backup_count: int
    :return: Resolved absolute path to the log file.
    :rtype: str

    Example::

        from perceptionmetrics.utils.logging_config import add_file_handler

        add_file_handler("logs/run.log")
        add_file_handler("logs/debug.log", max_bytes=10*1024*1024, backup_count=5)
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