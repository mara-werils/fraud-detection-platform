"""Structured logging configuration with JSON output and correlation IDs."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    cid = _correlation_id.get()
    if not cid:
        cid = str(uuid4())
        _correlation_id.set(cid)
    return cid


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(correlation_id)


def new_correlation_id() -> str:
    """Generate and set a new correlation ID."""
    cid = str(uuid4())
    _correlation_id.set(cid)
    return cid


def _add_correlation_id(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor that adds the correlation ID to every log entry."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def _add_service_info(
    service_name: str,
) -> structlog.types.Processor:
    """Create a processor that adds service name to log entries."""

    def processor(
        logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        event_dict["service"] = service_name
        return event_dict

    return processor


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    log_format: str = "json",
) -> None:
    """Configure structlog with JSON output, correlation IDs, and timestamps.

    Args:
        service_name: Name of the service for log identification.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format - 'json' for production, 'console' for development.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_correlation_id,
        _add_service_info(service_name),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    for noisy_logger in ("aiokafka", "kafka", "urllib3", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound structlog logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A bound structlog logger instance.
    """
    return structlog.get_logger(name)
