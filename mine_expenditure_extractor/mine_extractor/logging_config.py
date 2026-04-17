"""
mine_extractor.logging_config
------------------------------
Central logging setup. One call to ``configure_logging`` wires both
console and rotating file handlers for the whole package.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from settings import LOG_DIR, settings

_CONFIGURED = False

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_file: str = "pipeline.log") -> None:
    """Initialise the root logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(settings.log_level)

    # Console
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(console)

    # File (rotates at 5 MB, keeps 3 backups)
    file_path = Path(LOG_DIR) / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(file_handler)

    # Quieter third-party libs
    for noisy in ("urllib3", "httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, auto-configuring on first call."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
