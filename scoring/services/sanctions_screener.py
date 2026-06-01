"""Sanctions and PEP (Politically Exposed Persons) screening service.

Provides screening against sanctions lists and PEP databases for
AML (Anti-Money Laundering) compliance. Uses an in-memory list by default,
with support for external API integration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MatchType(StrEnum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    PARTIAL = "partial"


class ListType(StrEnum):
    SANCTIONS = "sanctions"
    PEP = "pep"
    WATCHLIST = "watchlist"


@dataclass(frozen=True)
class ScreeningMatch:
    """A match found during screening."""

    entity_name: str
    list_type: ListType
    list_source: str
    match_type: MatchType
    match_score: float  # 0.0 - 1.0
    details: str


@dataclass
class ScreeningResult:
    """Result of a screening check."""

    query: str
    screened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_match: bool = False
    matches: list[ScreeningMatch] = field(default_factory=list)
    lists_checked: list[str] = field(default_factory=list)
    risk_level: str = "clear"  # clear, low, medium, high

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "screened_at": self.screened_at.isoformat(),
            "is_match": self.is_match,
            "risk_level": self.risk_level,
            "match_count": len(self.matches),
            "lists_checked": self.lists_checked,
            "matches": [
                {
                    "entity_name": m.entity_name,
                    "list_type": m.list_type.value,
                    "list_source": m.list_source,
                    "match_type": m.match_type.value,
                    "match_score": round(m.match_score, 4),
                    "details": m.details,
                }
                for m in self.matches
            ],
        }


# Sample sanctions/PEP entries for demonstration
# In production, these would come from OFAC SDN, EU Sanctions, UN Consolidated, etc.
_DEMO_SANCTIONS: list[dict[str, str]] = [
    {"name": "DEMO SANCTIONED ENTITY", "source": "demo-ofac-sdn", "type": "sanctions", "details": "Demo entry for testing"},
    {"name": "TEST BLOCKED PERSON", "source": "demo-ofac-sdn", "type": "sanctions", "details": "Demo entry for testing"},
    {"name": "EXAMPLE PEP OFFICIAL", "source": "demo-pep-list", "type": "pep", "details": "Demo PEP entry"},
    {"name": "SAMPLE WATCHLIST NAME", "source": "demo-watchlist", "type": "watchlist", "details": "Demo watchlist entry"},
]


class SanctionsScreener:
    """Screen entities against sanctions, PEP, and watchlist databases.

    Supports exact, partial, and fuzzy matching with configurable
    thresholds. In production, integrate with external APIs like
    ComplyAdvantage, Refinitiv World-Check, or OFAC SDN downloads.

    Args:
        fuzzy_threshold: Minimum similarity score for fuzzy matches (0-1).
        external_api_url: Optional URL for external screening API.
        external_api_key: Optional API key for external screening.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.8,
        external_api_url: str | None = None,
        external_api_key: str | None = None,
    ) -> None:
        self._fuzzy_threshold = fuzzy_threshold
        self._external_api_url = external_api_url
        self._external_api_key = external_api_key
        self._entries: list[dict[str, str]] = list(_DEMO_SANCTIONS)
        self._screening_count = 0

    def add_entry(self, name: str, source: str, list_type: str, details: str = "") -> None:
        """Add an entry to the screening lists."""
        self._entries.append({
            "name": name.upper(),
            "source": source,
            "type": list_type,
            "details": details,
        })

    def screen(self, name: str, check_types: list[str] | None = None) -> ScreeningResult:
        """Screen a name against all loaded lists.

        Args:
            name: The entity name to screen.
            check_types: Optional list of types to check (sanctions, pep, watchlist).

        Returns:
            ScreeningResult with any matches found.
        """
        self._screening_count += 1
        query = name.strip().upper()
        normalized_query = self._normalize(query)

        types_to_check = set(check_types or ["sanctions", "pep", "watchlist"])

        matches: list[ScreeningMatch] = []
        lists_checked: set[str] = set()

        for entry in self._entries:
            if entry["type"] not in types_to_check:
                continue

            lists_checked.add(entry["source"])
            entry_name = entry["name"].upper()
            normalized_entry = self._normalize(entry_name)

            # Exact match
            if normalized_query == normalized_entry:
                matches.append(ScreeningMatch(
                    entity_name=entry_name,
                    list_type=ListType(entry["type"]),
                    list_source=entry["source"],
                    match_type=MatchType.EXACT,
                    match_score=1.0,
                    details=entry.get("details", ""),
                ))
                continue

            # Partial match (query contained in entry or vice versa)
            if normalized_query in normalized_entry or normalized_entry in normalized_query:
                score = len(normalized_query) / max(len(normalized_entry), 1)
                if score > 0.5:
                    matches.append(ScreeningMatch(
                        entity_name=entry_name,
                        list_type=ListType(entry["type"]),
                        list_source=entry["source"],
                        match_type=MatchType.PARTIAL,
                        match_score=score,
                        details=entry.get("details", ""),
                    ))
                    continue

            # Fuzzy match (Levenshtein-based similarity)
            similarity = self._similarity(normalized_query, normalized_entry)
            if similarity >= self._fuzzy_threshold:
                matches.append(ScreeningMatch(
                    entity_name=entry_name,
                    list_type=ListType(entry["type"]),
                    list_source=entry["source"],
                    match_type=MatchType.FUZZY,
                    match_score=similarity,
                    details=entry.get("details", ""),
                ))

        # Determine risk level
        is_match = len(matches) > 0
        risk_level = "clear"
        if matches:
            max_score = max(m.match_score for m in matches)
            has_sanctions = any(m.list_type == ListType.SANCTIONS for m in matches)
            if has_sanctions and max_score >= 0.95:
                risk_level = "high"
            elif has_sanctions or max_score >= 0.9:
                risk_level = "medium"
            else:
                risk_level = "low"

        # Sort by score descending
        matches.sort(key=lambda m: m.match_score, reverse=True)

        result = ScreeningResult(
            query=name,
            is_match=is_match,
            matches=matches,
            lists_checked=sorted(lists_checked),
            risk_level=risk_level,
        )

        if is_match:
            logger.warning(
                "sanctions_match_found",
                query=name,
                match_count=len(matches),
                risk_level=risk_level,
            )

        return result

    @property
    def stats(self) -> dict[str, Any]:
        """Return screening statistics."""
        return {
            "total_screenings": self._screening_count,
            "lists_loaded": len(set(e["source"] for e in self._entries)),
            "total_entries": len(self._entries),
        }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().upper()

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        """Compute string similarity using Levenshtein distance ratio."""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)

        # Optimized Levenshtein with two-row approach
        prev_row = list(range(len2 + 1))
        for i in range(1, len1 + 1):
            curr_row = [i] + [0] * len2
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr_row[j] = min(
                    curr_row[j - 1] + 1,
                    prev_row[j] + 1,
                    prev_row[j - 1] + cost,
                )
            prev_row = curr_row

        distance = prev_row[len2]
        return 1.0 - (distance / max_len)
