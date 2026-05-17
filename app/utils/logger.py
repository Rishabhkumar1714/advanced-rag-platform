"""
Structured Logging Utility
Provides consistent JSON and console logging across the application.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from app.config import settings


def setup_logging() -> None:
    """Configure structlog and standard library logging."""

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structlog logger."""
    return structlog.get_logger(name)


class RequestLogger:
    """Middleware helper to log request/response metadata."""

    @staticmethod
    def log_request(
        method: str,
        path: str,
        client_ip: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger = get_logger("http.request")
        logger.info(
            "Incoming request",
            method=method,
            path=path,
            client_ip=client_ip,
            **(extra or {}),
        )

    @staticmethod
    def log_response(
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger = get_logger("http.response")
        log_fn = logger.warning if status_code >= 400 else logger.info
        log_fn(
            "Request completed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            **(extra or {}),
        )
