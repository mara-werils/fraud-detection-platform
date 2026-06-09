"""Custom exception hierarchy for the scoring service.

Provides structured error types for different failure modes so that callers
can catch specific problems and API handlers can map them to HTTP status codes.
"""

from __future__ import annotations

__all__ = [
    "ScoringError",
    "ModelNotLoadedError",
    "ModelScoringError",
    "FeatureRetrievalError",
    "InvalidTransactionError",
    "RateLimitExceededError",
    "CaseNotFoundError",
    "WebhookDeliveryError",
    "ConfigurationError",
    "DriftDetectionError",
    "SanctionsMatchError",
    "EntityBlockedError",
]


class ScoringError(Exception):
    """Base exception for all scoring service errors."""

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ModelNotLoadedError(ScoringError):
    """Raised when a scoring model is not available."""


class ModelScoringError(ScoringError):
    """Raised when a model fails during inference."""


class FeatureRetrievalError(ScoringError):
    """Raised when feature data cannot be fetched from cache or store."""


class InvalidTransactionError(ScoringError):
    """Raised when a transaction fails validation before scoring."""


class RateLimitExceededError(ScoringError):
    """Raised when a client exceeds their configured rate limit."""


class CaseNotFoundError(ScoringError):
    """Raised when a fraud case cannot be found by ID."""


class WebhookDeliveryError(ScoringError):
    """Raised when a webhook payload cannot be delivered after retries."""


class ConfigurationError(ScoringError):
    """Raised for invalid or missing configuration values."""


class DriftDetectionError(ScoringError):
    """Raised when drift detection analysis fails."""
