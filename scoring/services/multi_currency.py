"""Multi-currency risk assessment engine.

Provides real-time multi-currency risk scoring including:
  - Real-time currency conversion with cached exchange rates
  - Cross-currency transaction pattern detection
  - Currency mismatch scoring (user home currency vs transaction currency)
  - High-risk currency pair identification
  - Structuring detection across currencies (smurfing)
  - Amount normalisation to base currency for consistent scoring
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

MULTI_CURRENCY_ASSESSMENTS = Counter(
    "multi_currency_assessments_total",
    "Total multi-currency risk assessments",
    ["assessment_type"],
)

CROSS_CURRENCY_PATTERNS = Counter(
    "multi_currency_cross_currency_patterns_total",
    "Cross-currency pattern detections",
    ["pattern_type"],
)

SMURFING_DETECTIONS = Counter(
    "multi_currency_smurfing_detections_total",
    "Cross-currency smurfing detections",
)

CURRENCY_MISMATCH_SCORE_HISTOGRAM = Histogram(
    "multi_currency_mismatch_score",
    "Distribution of currency mismatch scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ── Static data ──────────────────────────────────────────────────────────────

_DEFAULT_EXCHANGE_RATES: dict[str, Decimal] = {
    "USD": Decimal("1.0"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "JPY": Decimal("0.0067"),
    "CAD": Decimal("0.74"),
    "AUD": Decimal("0.65"),
    "CHF": Decimal("1.11"),
    "CNY": Decimal("0.138"),
    "HKD": Decimal("0.128"),
    "SGD": Decimal("0.74"),
    "SEK": Decimal("0.095"),
    "NOK": Decimal("0.094"),
    "DKK": Decimal("0.145"),
    "NZD": Decimal("0.60"),
    "MXN": Decimal("0.058"),
    "BRL": Decimal("0.198"),
    "INR": Decimal("0.012"),
    "KRW": Decimal("0.00073"),
    "ZAR": Decimal("0.054"),
    "TRY": Decimal("0.031"),
    "RUB": Decimal("0.011"),
    "AED": Decimal("0.272"),
    "SAR": Decimal("0.267"),
    "THB": Decimal("0.028"),
    "IDR": Decimal("0.000062"),
    "MYR": Decimal("0.213"),
    "PHP": Decimal("0.017"),
    "VND": Decimal("0.000040"),
    "PKR": Decimal("0.0036"),
    "BDT": Decimal("0.0091"),
    "NGN": Decimal("0.00065"),
    "KES": Decimal("0.0077"),
    "IRR": Decimal("0.000024"),
    "KPW": Decimal("0.0011"),
    "SYP": Decimal("0.000077"),
    "MMK": Decimal("0.00048"),
    "VES": Decimal("0.027"),
    "CUP": Decimal("0.042"),
    # Stablecoins
    "USDT": Decimal("1.0"),
    "USDC": Decimal("1.0"),
}

# High-risk currency pairs: (from, to) pairs that are inherently suspicious.
_HIGH_RISK_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("RUB", "USDT"),
        ("RUB", "CNY"),
        ("IRR", "AED"),
        ("IRR", "TRY"),
        ("KPW", "CNY"),
        ("SYP", "TRY"),
        ("SYP", "AED"),
        ("MMK", "THB"),
        ("MMK", "CNY"),
        ("VES", "USDT"),
        ("VES", "USD"),
        ("CUP", "MXN"),
        ("NGN", "USDT"),
        ("PKR", "AED"),
        # Reverse directions are also flagged
        ("USDT", "RUB"),
        ("CNY", "RUB"),
        ("AED", "IRR"),
        ("TRY", "IRR"),
        ("CNY", "KPW"),
        ("TRY", "SYP"),
        ("AED", "SYP"),
        ("THB", "MMK"),
        ("CNY", "MMK"),
        ("USDT", "VES"),
        ("USD", "VES"),
        ("MXN", "CUP"),
        ("USDT", "NGN"),
        ("AED", "PKR"),
    }
)

# Sanctioned or restricted currencies.
_SANCTIONED_CURRENCIES: frozenset[str] = frozenset(
    {"IRR", "KPW", "SYP", "CUP", "RUB"}
)

# Crypto / digital asset codes.
_CRYPTO_CURRENCIES: frozenset[str] = frozenset(
    {"BTC", "ETH", "XMR", "USDT", "USDC", "BNB", "LTC", "XRP", "DOGE", "SOL"}
)

# Currency zone groupings for mismatch detection.
_CURRENCY_ZONES: dict[str, str] = {
    "USD": "north_america",
    "CAD": "north_america",
    "MXN": "north_america",
    "EUR": "europe",
    "GBP": "europe",
    "CHF": "europe",
    "SEK": "europe",
    "NOK": "europe",
    "DKK": "europe",
    "JPY": "east_asia",
    "CNY": "east_asia",
    "KRW": "east_asia",
    "HKD": "east_asia",
    "SGD": "southeast_asia",
    "THB": "southeast_asia",
    "MYR": "southeast_asia",
    "IDR": "southeast_asia",
    "PHP": "southeast_asia",
    "VND": "southeast_asia",
    "INR": "south_asia",
    "PKR": "south_asia",
    "BDT": "south_asia",
    "AUD": "oceania",
    "NZD": "oceania",
    "BRL": "south_america",
    "AED": "middle_east",
    "SAR": "middle_east",
    "TRY": "middle_east",
    "IRR": "middle_east",
    "SYP": "middle_east",
    "NGN": "africa",
    "KES": "africa",
    "ZAR": "africa",
    "RUB": "russia_cis",
    "KPW": "east_asia",
    "MMK": "southeast_asia",
    "VES": "south_america",
    "CUP": "caribbean",
}

# CTR reporting threshold in USD.
_REPORTING_THRESHOLD_USD = Decimal("10000")

# Structuring band: flag amounts within this percentage below threshold.
_STRUCTURING_BAND_PCT = Decimal("0.15")


# ── Enumerations ─────────────────────────────────────────────────────────────


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternType(StrEnum):
    RAPID_CONVERSION = "rapid_conversion"
    CURRENCY_HOPPING = "currency_hopping"
    SPLIT_AND_CONVERT = "split_and_convert"
    ROUND_TRIP = "round_trip"


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CachedRate:
    """An exchange rate with a timestamp for cache invalidation."""

    rate: Decimal
    fetched_at: float  # time.monotonic() value


@dataclass(frozen=True)
class CurrencyMismatchResult:
    """Result of comparing a user's home currency against a transaction currency."""

    home_currency: str
    transaction_currency: str
    mismatch_score: float  # 0.0 – 1.0
    same_zone: bool
    is_high_risk_pair: bool
    risk_tier: RiskTier
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CrossCurrencyPattern:
    """Detected cross-currency pattern in a transaction sequence."""

    pattern_type: PatternType
    confidence: float  # 0.0 – 1.0
    involved_currencies: list[str]
    total_amount_usd: Decimal
    transaction_count: int
    description: str


@dataclass(frozen=True)
class SmurfingResult:
    """Result of cross-currency structuring (smurfing) analysis."""

    detected: bool
    confidence: float
    total_amount_usd: Decimal
    currencies_involved: list[str]
    flagged_amounts_usd: list[Decimal]
    threshold_usd: Decimal
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MultiCurrencyAssessment:
    """Comprehensive multi-currency risk assessment result."""

    overall_risk_score: float  # 0.0 – 1.0
    risk_tier: RiskTier
    normalized_amount_usd: Decimal
    mismatch: CurrencyMismatchResult | None
    patterns: list[CrossCurrencyPattern]
    smurfing: SmurfingResult | None
    high_risk_pairs_found: list[tuple[str, str]]
    notes: list[str]


# ── Service ──────────────────────────────────────────────────────────────────


class MultiCurrencyRiskEngine:
    """Multi-currency risk assessment engine.

    Provides cached exchange-rate lookups, cross-currency pattern detection,
    currency mismatch scoring, high-risk pair identification, smurfing
    detection, and amount normalisation to USD for consistent scoring.
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 300.0,
        reporting_threshold_usd: Decimal = _REPORTING_THRESHOLD_USD,
        structuring_band_pct: Decimal = _STRUCTURING_BAND_PCT,
    ) -> None:
        self._cache_ttl = cache_ttl_seconds
        self._threshold_usd = reporting_threshold_usd
        self._structuring_band = structuring_band_pct
        self._rate_cache: dict[str, CachedRate] = {}
        self._log = logger.bind(service="MultiCurrencyRiskEngine")

        # Seed cache from static defaults.
        now = time.monotonic()
        for code, rate in _DEFAULT_EXCHANGE_RATES.items():
            self._rate_cache[code] = CachedRate(rate=rate, fetched_at=now)

    # ── Exchange rate cache ──────────────────────────────────────────────────

    def get_rate(self, currency: str) -> Decimal:
        """Return the cached USD exchange rate for *currency*.

        If the rate is expired or unknown, falls back to the static default
        or returns Decimal('1.0') as a safe fallback.
        """
        code = currency.upper().strip()
        cached = self._rate_cache.get(code)
        now = time.monotonic()

        if cached is not None and (now - cached.fetched_at) < self._cache_ttl:
            return cached.rate

        # Fallback to static defaults when cache is stale / missing.
        default = _DEFAULT_EXCHANGE_RATES.get(code, Decimal("1.0"))
        self._rate_cache[code] = CachedRate(rate=default, fetched_at=now)
        self._log.debug("rate_cache_refreshed", currency=code, rate=str(default))
        return default

    def update_rate(self, currency: str, rate: Decimal) -> None:
        """Manually update the cached rate for *currency*."""
        code = currency.upper().strip()
        self._rate_cache[code] = CachedRate(rate=rate, fetched_at=time.monotonic())
        self._log.debug("rate_updated", currency=code, rate=str(rate))

    def is_rate_cached(self, currency: str) -> bool:
        """Check whether a non-expired rate exists in the cache."""
        code = currency.upper().strip()
        cached = self._rate_cache.get(code)
        if cached is None:
            return False
        return (time.monotonic() - cached.fetched_at) < self._cache_ttl

    # ── Amount normalisation ─────────────────────────────────────────────────

    def normalize_to_usd(self, amount: Decimal, currency: str) -> Decimal:
        """Convert *amount* in *currency* to its USD equivalent."""
        rate = self.get_rate(currency)
        return (amount * rate).quantize(Decimal("0.01"))

    # ── Currency mismatch scoring ────────────────────────────────────────────

    def score_currency_mismatch(
        self,
        home_currency: str,
        transaction_currency: str,
    ) -> CurrencyMismatchResult:
        """Score the risk of a transaction in *transaction_currency* by a user
        whose home currency is *home_currency*.

        Returns a CurrencyMismatchResult with a score from 0.0 (no mismatch)
        to 1.0 (maximum mismatch risk).
        """
        home = home_currency.upper().strip()
        txn = transaction_currency.upper().strip()

        MULTI_CURRENCY_ASSESSMENTS.labels(assessment_type="mismatch").inc()

        # No mismatch: same currency.
        if home == txn:
            return CurrencyMismatchResult(
                home_currency=home,
                transaction_currency=txn,
                mismatch_score=0.0,
                same_zone=True,
                is_high_risk_pair=False,
                risk_tier=RiskTier.LOW,
                notes=["No currency mismatch."],
            )

        notes: list[str] = []
        score = 0.0

        # Zone comparison.
        home_zone = _CURRENCY_ZONES.get(home, "unknown")
        txn_zone = _CURRENCY_ZONES.get(txn, "unknown")
        same_zone = home_zone == txn_zone and home_zone != "unknown"

        if not same_zone:
            score += 0.25
            notes.append(f"Cross-zone transaction: {home_zone} -> {txn_zone}.")

        # High-risk pair check.
        is_high_risk_pair = (home, txn) in _HIGH_RISK_PAIRS
        if is_high_risk_pair:
            score += 0.35
            notes.append(f"High-risk currency pair: {home} -> {txn}.")

        # Sanctioned currency involvement.
        if home in _SANCTIONED_CURRENCIES or txn in _SANCTIONED_CURRENCIES:
            score += 0.25
            notes.append("Sanctioned currency involved in mismatch.")

        # Crypto involvement adds moderate risk.
        if home in _CRYPTO_CURRENCIES or txn in _CRYPTO_CURRENCIES:
            score += 0.10
            notes.append("Crypto or digital asset involved.")

        # Base mismatch penalty.
        if not same_zone:
            score += 0.05

        score = round(min(score, 1.0), 4)
        tier = self._score_to_tier(score)

        CURRENCY_MISMATCH_SCORE_HISTOGRAM.observe(score)

        self._log.debug(
            "currency_mismatch_scored",
            home=home,
            txn=txn,
            score=score,
            tier=tier,
        )

        return CurrencyMismatchResult(
            home_currency=home,
            transaction_currency=txn,
            mismatch_score=score,
            same_zone=same_zone,
            is_high_risk_pair=is_high_risk_pair,
            risk_tier=tier,
            notes=notes,
        )

    # ── High-risk pair identification ────────────────────────────────────────

    def identify_high_risk_pairs(
        self, currencies: list[str],
    ) -> list[tuple[str, str]]:
        """Given a list of currency codes, return all high-risk pairs present."""
        codes = [c.upper().strip() for c in currencies]
        found: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for src in codes:
            for dst in codes:
                if src == dst:
                    continue
                pair = (src, dst)
                if pair in _HIGH_RISK_PAIRS and pair not in seen:
                    found.append(pair)
                    seen.add(pair)

        return found

    # ── Cross-currency pattern detection ─────────────────────────────────────

    def detect_cross_currency_patterns(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[CrossCurrencyPattern]:
        """Analyse a sequence of transactions for suspicious cross-currency patterns.

        Each transaction dict must contain at minimum:
            {"amount": Decimal, "currency": str, "timestamp": float}
        Optional: "user_id", "direction" ("in"/"out").

        Detects:
          - Rapid conversion: multiple currencies within a short window
          - Currency hopping: sequential use of 3+ distinct currencies
          - Split-and-convert: many small same-currency txns followed by conversion
        """
        if len(transactions) < 2:
            return []

        MULTI_CURRENCY_ASSESSMENTS.labels(assessment_type="pattern_detection").inc()

        patterns: list[CrossCurrencyPattern] = []

        # Sort by timestamp.
        sorted_txns = sorted(transactions, key=lambda t: t.get("timestamp", 0))

        # Extract currency sequence and amounts.
        currencies_in_order: list[str] = []
        total_usd = Decimal("0")
        for tx in sorted_txns:
            code = str(tx.get("currency", "USD")).upper()
            amount = tx.get("amount", Decimal("0"))
            currencies_in_order.append(code)
            total_usd += self.normalize_to_usd(amount, code)

        unique_currencies = list(dict.fromkeys(currencies_in_order))

        # ---- Rapid conversion: 3+ currencies within the window ----
        if len(sorted_txns) >= 3:
            window_seconds = 3600  # 1 hour
            first_ts = sorted_txns[0].get("timestamp", 0)
            last_ts = sorted_txns[-1].get("timestamp", 0)
            if (last_ts - first_ts) <= window_seconds and len(unique_currencies) >= 3:
                confidence = min(1.0, len(unique_currencies) / 5.0)
                patterns.append(
                    CrossCurrencyPattern(
                        pattern_type=PatternType.RAPID_CONVERSION,
                        confidence=round(confidence, 2),
                        involved_currencies=unique_currencies,
                        total_amount_usd=total_usd.quantize(Decimal("0.01")),
                        transaction_count=len(sorted_txns),
                        description=(
                            f"Rapid conversion across {len(unique_currencies)} currencies "
                            f"within {int(last_ts - first_ts)}s."
                        ),
                    )
                )
                CROSS_CURRENCY_PATTERNS.labels(pattern_type="rapid_conversion").inc()

        # ---- Currency hopping: sequential distinct currencies ----
        if len(unique_currencies) >= 3:
            # Check for a run of distinct consecutive currencies.
            max_run = 1
            current_run = 1
            for i in range(1, len(currencies_in_order)):
                if currencies_in_order[i] != currencies_in_order[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1

            if max_run >= 3:
                confidence = min(1.0, max_run / 5.0)
                patterns.append(
                    CrossCurrencyPattern(
                        pattern_type=PatternType.CURRENCY_HOPPING,
                        confidence=round(confidence, 2),
                        involved_currencies=unique_currencies,
                        total_amount_usd=total_usd.quantize(Decimal("0.01")),
                        transaction_count=len(sorted_txns),
                        description=(
                            f"Currency hopping detected: {max_run} consecutive "
                            f"distinct currencies."
                        ),
                    )
                )
                CROSS_CURRENCY_PATTERNS.labels(pattern_type="currency_hopping").inc()

        # ---- Split-and-convert: many small txns in one currency, then a conversion ----
        if len(sorted_txns) >= 4 and len(unique_currencies) >= 2:
            # Group consecutive txns by currency.
            groups: list[tuple[str, list[dict[str, Any]]]] = []
            current_group_curr = currencies_in_order[0]
            current_group: list[dict[str, Any]] = [sorted_txns[0]]
            for i in range(1, len(sorted_txns)):
                if currencies_in_order[i] == current_group_curr:
                    current_group.append(sorted_txns[i])
                else:
                    groups.append((current_group_curr, current_group))
                    current_group_curr = currencies_in_order[i]
                    current_group = [sorted_txns[i]]
            groups.append((current_group_curr, current_group))

            # A split-and-convert is a group of 3+ same-currency txns followed
            # by a different-currency group.
            for idx in range(len(groups) - 1):
                grp_curr, grp_txns = groups[idx]
                next_curr, _ = groups[idx + 1]
                if len(grp_txns) >= 3 and grp_curr != next_curr:
                    grp_total = sum(
                        self.normalize_to_usd(
                            t.get("amount", Decimal("0")), grp_curr
                        )
                        for t in grp_txns
                    )
                    confidence = min(1.0, len(grp_txns) / 6.0)
                    patterns.append(
                        CrossCurrencyPattern(
                            pattern_type=PatternType.SPLIT_AND_CONVERT,
                            confidence=round(confidence, 2),
                            involved_currencies=[grp_curr, next_curr],
                            total_amount_usd=grp_total.quantize(Decimal("0.01")),
                            transaction_count=len(grp_txns) + 1,
                            description=(
                                f"{len(grp_txns)} small {grp_curr} transactions "
                                f"(USD {grp_total.quantize(Decimal('0.01'))}) "
                                f"followed by conversion to {next_curr}."
                            ),
                        )
                    )
                    CROSS_CURRENCY_PATTERNS.labels(pattern_type="split_and_convert").inc()

        return patterns

    # ── Cross-currency structuring (smurfing) detection ──────────────────────

    def detect_smurfing(
        self,
        transactions: list[dict[str, Any]],
        threshold_usd: Decimal | None = None,
    ) -> SmurfingResult:
        """Detect structuring (smurfing) across multiple currencies.

        Examines whether a set of transactions in various currencies, when
        normalised to USD, appear designed to stay below the reporting
        threshold individually while collectively exceeding it.

        Each transaction dict requires: {"amount": Decimal, "currency": str}.
        """
        MULTI_CURRENCY_ASSESSMENTS.labels(assessment_type="smurfing").inc()

        effective_threshold = threshold_usd or self._threshold_usd
        lower_bound = effective_threshold * (1 - self._structuring_band)

        usd_amounts: list[Decimal] = []
        currencies_involved: set[str] = set()
        flagged: list[Decimal] = []

        for tx in transactions:
            code = str(tx.get("currency", "USD")).upper()
            amount = tx.get("amount", Decimal("0"))
            usd = self.normalize_to_usd(amount, code)
            usd_amounts.append(usd)
            currencies_involved.add(code)

            if lower_bound <= usd < effective_threshold:
                flagged.append(usd)

        total_usd = sum(usd_amounts, Decimal("0")).quantize(Decimal("0.01"))
        notes: list[str] = []

        # Detection criteria:
        # 1. Multiple individually-below-threshold amounts that sum above threshold.
        # 2. Near-threshold amounts across different currencies.
        detected = False
        confidence = 0.0

        if len(flagged) >= 2:
            detected = True
            confidence = min(1.0, len(flagged) / 4.0)
            notes.append(
                f"{len(flagged)} transactions near the USD {effective_threshold} "
                f"threshold across {len(currencies_involved)} currencies."
            )

        if (
            not detected
            and len(currencies_involved) >= 2
            and total_usd >= effective_threshold
            and all(a < effective_threshold for a in usd_amounts)
        ):
            detected = True
            confidence = min(1.0, 0.5 + (len(currencies_involved) - 1) * 0.15)
            notes.append(
                f"Individual amounts below threshold but aggregate "
                f"(USD {total_usd}) exceeds USD {effective_threshold} "
                f"across {len(currencies_involved)} currencies."
            )

        if detected:
            SMURFING_DETECTIONS.inc()
            self._log.warning(
                "smurfing_detected",
                flagged_count=len(flagged),
                total_usd=str(total_usd),
                currencies=sorted(currencies_involved),
                confidence=confidence,
            )

        return SmurfingResult(
            detected=detected,
            confidence=round(confidence, 4),
            total_amount_usd=total_usd,
            currencies_involved=sorted(currencies_involved),
            flagged_amounts_usd=flagged,
            threshold_usd=effective_threshold,
            notes=notes,
        )

    # ── Comprehensive assessment ─────────────────────────────────────────────

    def assess(
        self,
        transactions: list[dict[str, Any]],
        home_currency: str | None = None,
    ) -> MultiCurrencyAssessment:
        """Run a full multi-currency risk assessment on a set of transactions.

        Parameters
        ----------
        transactions:
            List of transaction dicts with keys: amount, currency, timestamp.
        home_currency:
            The user's home/primary currency. When provided, mismatch scoring
            is performed against the most recent transaction.

        Returns
        -------
        MultiCurrencyAssessment with aggregated risk scores and findings.
        """
        MULTI_CURRENCY_ASSESSMENTS.labels(assessment_type="full").inc()

        notes: list[str] = []
        all_currencies = [
            str(tx.get("currency", "USD")).upper() for tx in transactions
        ]
        unique_currencies = list(dict.fromkeys(all_currencies))

        # Normalise total.
        total_usd = Decimal("0")
        for tx in transactions:
            code = str(tx.get("currency", "USD")).upper()
            amount = tx.get("amount", Decimal("0"))
            total_usd += self.normalize_to_usd(amount, code)
        total_usd = total_usd.quantize(Decimal("0.01"))

        # Currency mismatch (against most recent transaction).
        mismatch: CurrencyMismatchResult | None = None
        if home_currency and transactions:
            last_txn_currency = str(
                transactions[-1].get("currency", "USD")
            ).upper()
            mismatch = self.score_currency_mismatch(home_currency, last_txn_currency)

        # Cross-currency patterns.
        patterns = self.detect_cross_currency_patterns(transactions)

        # Smurfing.
        smurfing = self.detect_smurfing(transactions)

        # High-risk pairs.
        hr_pairs = self.identify_high_risk_pairs(unique_currencies)

        # Compute overall risk score.
        score = 0.0

        if mismatch:
            score += mismatch.mismatch_score * 0.20

        if patterns:
            max_pattern_confidence = max(p.confidence for p in patterns)
            score += max_pattern_confidence * 0.30

        if smurfing and smurfing.detected:
            score += smurfing.confidence * 0.25

        if hr_pairs:
            score += min(0.25, len(hr_pairs) * 0.10)

        score = round(min(score, 1.0), 4)
        tier = self._score_to_tier(score)

        if patterns:
            notes.append(f"{len(patterns)} cross-currency pattern(s) detected.")
        if smurfing and smurfing.detected:
            notes.append("Cross-currency structuring (smurfing) detected.")
        if hr_pairs:
            notes.append(f"{len(hr_pairs)} high-risk currency pair(s) found.")

        self._log.info(
            "multi_currency_assessment_complete",
            score=score,
            tier=tier,
            currencies=unique_currencies,
            patterns=len(patterns),
            smurfing=smurfing.detected if smurfing else False,
        )

        return MultiCurrencyAssessment(
            overall_risk_score=score,
            risk_tier=tier,
            normalized_amount_usd=total_usd,
            mismatch=mismatch,
            patterns=patterns,
            smurfing=smurfing,
            high_risk_pairs_found=hr_pairs,
            notes=notes,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _score_to_tier(score: float) -> RiskTier:
        if score >= 0.75:
            return RiskTier.CRITICAL
        if score >= 0.50:
            return RiskTier.HIGH
        if score >= 0.25:
            return RiskTier.MEDIUM
        return RiskTier.LOW


# ── Module-level singleton ───────────────────────────────────────────────────

_default_engine: MultiCurrencyRiskEngine | None = None


def get_multi_currency_engine() -> MultiCurrencyRiskEngine:
    """Return the module-level singleton, creating it on first call."""
    global _default_engine
    if _default_engine is None:
        _default_engine = MultiCurrencyRiskEngine()
    return _default_engine
