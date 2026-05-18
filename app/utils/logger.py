"""
Structured Logging Utility
"""

import logging
import sys
from typing import Any, Dict, Optional


def setup_logging(log_level: str = "INFO") -> None:
    """Configure basic logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        level=level,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


class RequestLogger:
    @staticmethod
    def log_request(method: str, path: str, client_ip: str, extra: Optional[Dict[str, Any]] = None) -> None:
        logger = get_logger("http.request")
        logger.info(f"Incoming request: {method} {path} from {client_ip}")

    @staticmethod
    def log_response(method: str, path: str, status_code: int, duration_ms: float, extra: Optional[Dict[str, Any]] = None) -> None:
        logger = get_logger("http.response")
        logger.info(f"Request completed: {method} {path} -> {status_code} ({duration_ms:.2f}ms)")
