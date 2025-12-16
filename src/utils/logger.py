"""Logging setup for My Atlas RAG Chatbot."""

import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from .config import config


def setup_logger(name: str) -> logging.Logger:
    """Set up logger with JSON or text formatting.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set level from config
    level_name = config.log.level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # File handler
    log_file = config.paths.logs_dir / "app.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Format
    if config.log.format == 'json':
        # JSON formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={'asctime': 'timestamp', 'levelname': 'level'}
        )
    else:
        # Text formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Create module-specific loggers
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Module name (use __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)
