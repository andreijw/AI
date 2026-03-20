"""Logging utilities for training."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional


class Logger:
    """
    Simple logger for training metrics and events.

    Logs to both console and file, with optional structured logging support.
    """

    def __init__(
        self,
        name: str = "rl_cartpole",
        log_dir: str = "./logs",
        level: int = logging.INFO,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_dir: Directory to save log files
            level: Logging level
        """
        self.name = name
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create formatters
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self._handlers = [console_handler, file_handler]

        # Metrics log file (for structured data)
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.jsonl")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        # Log to structured metrics file
        try:
            with open(self.metrics_file, "a") as f:
                log_entry = {**metrics, "step": step} if step is not None else metrics
                f.write(json.dumps(log_entry) + "\n")
        except (OSError, TypeError, ValueError) as e:
            self.logger.warning("Failed to write metrics to %s: %s", self.metrics_file, e)

        # Log summary to console
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(f"Metrics: {metrics_str}")

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def close(self) -> None:
        """Close handlers created by this Logger instance."""
        for handler in self._handlers[:]:
            handler.close()
            if handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        self._handlers.clear()


def setup_logger(
    name: str = "rl_cartpole",
    log_dir: str = "./logs",
    level: int = logging.INFO,
) -> Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Configured Logger instance
    """
    return Logger(name=name, log_dir=log_dir, level=level)
