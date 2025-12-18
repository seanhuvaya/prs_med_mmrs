import os
import logging

from typing import Optional

def setup_logging(
        log_file: Optional[str] = None,
        level: int = logging.INFO,
):
    """
    Initialize logging configuration.

    Args:
        log_file: Path to the log file (if None, logs only to stdout)
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else None,
        ]
    )
