"""PII masking utilities for log sanitization.

Masks sensitive fields in dictionaries and strings before they reach
log output or external systems.
"""

from __future__ import annotations

import re
from typing import Any

# Fields that should be masked in log payloads
SENSITIVE_FIELDS = frozenset({
    "card_number",
    "card_number_hash",
    "ssn",
    "social_security_number",
    "password",
    "secret",
    "api_key",
    "token",
    "authorization",
    "email",
    "phone",
    "phone_number",
    "account_number",
})

# Regex patterns for inline PII detection
_CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def mask_value(value: str, visible_chars: int = 4) -> str:
    """Mask a string, preserving only the last N characters."""
    if len(value) <= visible_chars:
        return "***"
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


def mask_dict(data: dict[str, Any], extra_fields: frozenset[str] | None = None) -> dict[str, Any]:
    """Return a copy of the dict with sensitive field values masked.

    Args:
        data: Input dictionary to sanitize.
        extra_fields: Additional field names to treat as sensitive.

    Returns:
        New dictionary with masked values.
    """
    fields = SENSITIVE_FIELDS | (extra_fields or frozenset())
    result = {}
    for key, value in data.items():
        if key.lower() in fields:
            result[key] = mask_value(str(value)) if value is not None else None
        elif isinstance(value, dict):
            result[key] = mask_dict(value, extra_fields)
        else:
            result[key] = value
    return result


def mask_card_numbers(text: str) -> str:
    """Replace card numbers in free text with masked versions."""
    return _CARD_RE.sub(lambda m: mask_value(m.group().replace(" ", "").replace("-", "")), text)


def mask_emails(text: str) -> str:
    """Replace email addresses in free text with masked versions."""
    return _EMAIL_RE.sub(lambda m: mask_value(m.group()), text)
