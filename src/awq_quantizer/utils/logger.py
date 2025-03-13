"""
Logging utility for AWQ Quantizer.
"""

import logging
import os
import sys
from typing import Optional


class Logger:
    """
    Logger class for AWQ Quantizer.
    """

    def __init__(
        self,
        name: str = "awq_quantizer",
        level: str = "INFO",
        to_file: bool = False,
        file_path: Optional[str] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            to_file: Whether to log to file
            file_path: Log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.propagate = False

        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Create file handler if needed
        if to_file:
            if file_path is None:
                file_path = "quantization.log"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
            
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str):
        """Log critical message."""
        self.logger.critical(msg)


def get_logger(
    name: str = "awq_quantizer",
    level: str = "INFO",
    to_file: bool = False,
    file_path: Optional[str] = None,
) -> Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        to_file: Whether to log to file
        file_path: Log file path

    Returns:
        Logger instance
    """
    return Logger(name=name, level=level, to_file=to_file, file_path=file_path) 