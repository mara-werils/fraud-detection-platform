"""Shared library for the fraud detection platform."""

from shared.config import Settings
from shared.schemas import AlertEvent, FeatureVector, ScoredTransaction, Transaction

__all__ = [
    "AlertEvent",
    "FeatureVector",
    "ScoredTransaction",
    "Settings",
    "Transaction",
]
