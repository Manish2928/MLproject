"""
Logging setup for DementiaVoiceAI.

Configures structured logging to both console and file.
Privacy-safe: never logs raw audio content or full participant names.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started", extra={"epoch": 1})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def setup_logging(
    log_dir: str | Path = "logs",
    log_file: str = "dementiavoiceai.log",
    level: int = logging.INFO,
    console_level: int = logging.INFO,
) -> None:
    """Configure root logger with console and file handlers.

    Parameters
    ----------
    log_dir : str | Path
        Directory for log files. Created if it doesn't exist.
    log_file : str
        Name of the log file.
    level : int
        Root logger level.
    console_level : int
        Console handler level.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Formatter
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(
        log_path / log_file, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Automatically calls setup_logging() on first use.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.

    Returns
    -------
    logging.Logger
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
