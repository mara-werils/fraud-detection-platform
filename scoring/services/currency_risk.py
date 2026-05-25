"""Multi-currency risk scoring service for fraud detection.

Provides risk assessment across currency dimensions including:
  - Per-currency risk profiling (sanctions, FATF jurisdiction, crypto)
  - Cross-currency conversion risk and structuring detection
  - Normalisation of amounts to USD for threshold comparisons
  - Round-tripping detection across a transaction sequence
  - Jurisdiction risk scoring aligned with FATF grey/black lists
"""

from __future__ import annotations

import math
from decimal import Decimal
from enum import StrEnum
from typing import Any

import structlog
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

CURRENCY_ASSESSMENTS_TOTAL = Counter(
    "currency_risk_assessments_total",
    "Total currency risk assessments performed",
    ["currency_code", "risk_level"],
)

CONVERSION_ASSESSMENTS_TOTAL = Counter(
    "currency_risk_conversion_assessments_total",
    "Total cross-currency conversion risk assessments",
    ["from_currency", "to_currency"],
)

STRUCTURING_DETECTIONS_TOTAL = Counter(
    "currency_risk_structuring_detections_total",
    "Total structuring patterns detected",
    ["currency_code"],
)

ROUND_TRIP_DETECTIONS_TOTAL = Counter(
    "currency_risk_round_trip_detections_total",
    "Total round-tripping patterns detected",
)

CURRENCY_RISK_SCORE_HISTOGRAM = Histogram(
    "currency_risk_score_distribution",
    "Distribution of currency base risk scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

NORMALISED_AMOUNT_HISTOGRAM = Histogram(
    "currency_risk_normalised_amount_usd",
    "Distribution of normalised transaction amounts in USD",
    buckets=(100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000),
)

# ── Static risk data ──────────────────────────────────────────────────────────

# ISO 4217 codes → approximate USD conversion rates (updated periodically in
# production; these serve as safe fallback values for the service).
_USD_RATES: dict[str, Decimal] = {
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
    "GHS": Decimal("0.066"),
    "UGX": Decimal("0.00027"),
    "IRR": Decimal("0.000024"),  # Iranian rial
    "KPW": Decimal("0.0011"),   # North Korean won (parallel rate)
    "SYP": Decimal("0.000077"), # Syrian pound
    "SDG": Decimal("0.0017"),   # Sudanese pound
    "MMK": Decimal("0.00048"),  # Myanmar kyat
    "VES": Decimal("0.027"),    # Venezuelan bolivar
    "CUP": Decimal("0.042"),    # Cuban peso
    "LYD": Decimal("0.207"),    # Libyan dinar
    "YER": Decimal("0.0040"),   # Yemeni rial
    # Stablecoins / crypto proxies tracked for fiat-adjacent exposure
    "USDT": Decimal("1.0"),
    "USDC": Decimal("1.0"),
    "BUSD": Decimal("1.0"),
}

# FATF high-risk and other monitored jurisdictions (ISO 3166-1 alpha-2).
# Source: FATF Jurisdictions under increased monitoring + black list (2024-25).
_FATF_HIGH_RISK_COUNTRIES: frozenset[str] = frozenset(
    {
        # FATF Black List (Call for Action)
        "KP",  # North Korea
        "IR",  # Iran
        "MM",  # Myanmar
        # FATF Grey List (Increased Monitoring)
        "AF",  # Afghanistan
        "AL",  # Albania (periodic)
        "BB",  # Barbados
        "BF",  # Burkina Faso
        "CM",  # Cameroon
        "CD",  # DR Congo
        "HT",  # Haiti
        "JM",  # Jamaica
        "JO",  # Jordan
        "ML",  # Mali
        "MZ",  # Mozambique
        "NI",  # Nicaragua
        "NG",  # Nigeria
        "PK",  # Pakistan
        "PA",  # Panama
        "PH",  # Philippines
        "SN",  # Senegal
        "SO",  # Somalia
        "SS",  # South Sudan
        "SY",  # Syria
        "TZ",  # Tanzania
        "TT",  # Trinidad & Tobago
        "UG",  # Uganda
        "AE",  # UAE (periodic)
        "VN",  # Vietnam
        "YE",  # Yemen
    }
)

# Sanctioned country → primary currency mappings.
_SANCTIONED_CURRENCIES: frozenset[str] = frozenset(
    {"IRR", "KPW", "SYP", "SDG", "CUB", "RUB", "BYR"}
)

# Currency codes associated with crypto or digital-asset systems.
_CRYPTO_CURRENCIES: frozenset[str] = frozenset(
    {"BTC", "ETH", "XMR", "USDT", "USDC", "BUSD", "BNB", "LTC", "XRP", "DOGE", "SOL"}
)

# Country → primary currency (subset; used to map country risk → currency).
_COUNTRY_CURRENCY: dict[str, str] = {
    "KP": "KPW",
    "IR": "IRR",
    "MM": "MMK",
    "SY": "SYP",
    "SD": "SDG",
    "CU": "CUP",
    "LY": "LYD",
    "YE": "YER",
    "RU": "RUB",
    "VE": "VES",
}

# Per-country baseline jurisdiction risk score (0.0 – 1.0).
_JURISDICTION_RISK_SCORES: dict[str, float] = {
    "KP": 1.0,
    "IR": 1.0,
    "MM": 0.90,
    "SY": 0.95,
    "SD": 0.85,
    "AF": 0.85,
    "SO": 0.85,
    "SS": 0.85,
    "YE": 0.80,
    "LY": 0.80,
    "CU": 0.75,
    "VE": 0.70,
    "RU": 0.70,
    "BY": 0.65,
    "NG": 0.55,
    "PK": 0.55,
    "UA": 0.50,  # conflict zone
    "AE": 0.40,
    "PH": 0.40,
    "TT": 0.40,
    "PA": 0.35,
    "JM": 0.35,
    "HT": 0.65,
    # Low-risk reference points
    "US": 0.05,
    "GB": 0.05,
    "DE": 0.05,
    "JP": 0.05,
    "AU": 0.05,
    "CA": 0.05,
    "CH": 0.10,  # slightly elevated due to banking secrecy history
    "SG": 0.15,
}

# Approximate annualised volatility score (0.0 – 1.0) per currency.
_VOLATILITY_SCORES: dict[str, float] = {
    "USD": 0.0,
    "EUR": 0.05,
    "GBP": 0.08,
    "JPY": 0.07,
    "CHF": 0.06,
    "CAD": 0.09,
    "AUD": 0.12,
    "NZD": 0.12,
    "CNY": 0.05,
    "HKD": 0.02,
    "TRY": 0.75,
    "IRR": 0.90,
    "VES": 0.95,
    "SDG": 0.80,
    "MMK": 0.70,
    "KPW": 0.85,
    "SYP": 0.88,
    "RUB": 0.60,
    "BYR": 0.55,
    "NGN": 0.50,
    "PKR": 0.45,
    "BDT": 0.30,
    "BRL": 0.35,
    "MXN": 0.28,
    "INR": 0.18,
    "ZAR": 0.35,
    "KES": 0.35,
    "GHS": 0.55,
    "BTC": 1.0,
    "ETH": 0.95,
    "XMR": 0.95,
    "USDT": 0.02,
    "USDC": 0.01,
    "BUSD": 0.01,
    "BNB": 0.85,
    "LTC": 0.90,
    "XRP": 0.88,
    "DOGE": 0.98,
    "SOL": 0.92,
}

# CTR / SAR reporting threshold in USD (US FinCEN standard).
_DEFAULT_REPORTING_THRESHOLD_USD = Decimal("10000")

# Structuring detection window: amounts within this fraction below the threshold
# are flagged as potential structuring.
_STRUCTURING_FRACTION = Decimal("0.15")  # within 15 % below threshold


# ── Enumerations ──────────────────────────────────────────────────────────────


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Pydantic models ───────────────────────────────────────────────────────────


class CurrencyRiskProfile(BaseModel):
    """Risk profile for a single currency and transaction amount."""

    currency_code: str = Field(..., description="ISO 4217 currency code")
    risk_level: RiskLevel = Field(..., description="Aggregated risk tier")
    base_risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite risk score")

    is_sanctioned: bool = Field(False, description="Currency belongs to a sanctioned country")
    is_crypto: bool = Field(False, description="Currency is a crypto or digital asset")
    is_high_risk_jurisdiction: bool = Field(
        False, description="Currency's primary jurisdiction is FATF high-risk"
    )

    conversion_rate_to_usd: Decimal = Field(
        ..., description="Approximate conversion rate to USD"
    )
    volatility_score: float = Field(
        ..., ge=0.0, le=1.0, description="Annualised price-volatility score"
    )
    amount_usd: Decimal = Field(..., description="Transaction amount normalised to USD")


class CurrencyConversionRisk(BaseModel):
    """Risk assessment for a cross-currency conversion."""

    from_currency: str = Field(..., description="Source currency code")
    to_currency: str = Field(..., description="Destination currency code")
    conversion_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Composite conversion risk score"
    )

    is_cross_border: bool = Field(
        False, description="Conversion involves jurisdictions in different regulatory zones"
    )
    jurisdiction_risk: float = Field(
        ..., ge=0.0, le=1.0, description="Blended jurisdiction risk for both currencies"
    )
    structuring_indicator: bool = Field(
        False, description="Amount is positioned just below a reporting threshold"
    )
    amount_usd: Decimal = Field(..., description="Transaction amount in USD")
    risk_notes: list[str] = Field(default_factory=list, description="Human-readable risk factors")


# ── Service ───────────────────────────────────────────────────────────────────


class CurrencyRiskService:
    """Multi-currency risk scoring and structuring detection service.

    All public methods are synchronous and safe to call from async contexts
    via a thread-pool executor. The service is stateless aside from its
    configuration and the built-in static risk data loaded at import time.
    """

    def __init__(
        self,
        reporting_threshold_usd: Decimal = _DEFAULT_REPORTING_THRESHOLD_USD,
        structuring_fraction: Decimal = _STRUCTURING_FRACTION,
        unknown_currency_risk_score: float = 0.6,
    ) -> None:
        self._threshold_usd = reporting_threshold_usd
        self._structuring_fraction = structuring_fraction
        self._unknown_score = unknown_currency_risk_score
        self._log = logger.bind(service="CurrencyRiskService")

    # ── Public API ────────────────────────────────────────────────────────────

    def assess_currency_risk(
        self, currency: str, amount: Decimal
    ) -> CurrencyRiskProfile:
        """Return a full risk profile for *currency* and *amount*."""
        code = currency.upper().strip()
        rate = _USD_RATES.get(code, Decimal("1.0"))
        amount_usd = (amount * rate).quantize(Decimal("0.01"))

        is_sanctioned = code in _SANCTIONED_CURRENCIES
        is_crypto = code in _CRYPTO_CURRENCIES
        volatility = _VOLATILITY_SCORES.get(code, 0.5)
        is_high_risk_jur = self._currency_in_high_risk_jurisdiction(code)

        score = self._compute_base_risk_score(
            is_sanctioned=is_sanctioned,
            is_crypto=is_crypto,
            is_high_risk_jurisdiction=is_high_risk_jur,
            volatility=volatility,
            amount_usd=amount_usd,
            known=code in _USD_RATES,
        )
        risk_level = self._score_to_level(score)

        CURRENCY_ASSESSMENTS_TOTAL.labels(
            currency_code=code, risk_level=risk_level
        ).inc()
        CURRENCY_RISK_SCORE_HISTOGRAM.observe(score)
        NORMALISED_AMOUNT_HISTOGRAM.observe(float(amount_usd))

        self._log.debug(
            "currency_risk_assessed",
            currency=code,
            amount_usd=str(amount_usd),
            score=score,
            risk_level=risk_level,
        )

        return CurrencyRiskProfile(
            currency_code=code,
            risk_level=risk_level,
            base_risk_score=score,
            is_sanctioned=is_sanctioned,
            is_crypto=is_crypto,
            is_high_risk_jurisdiction=is_high_risk_jur,
            conversion_rate_to_usd=rate,
            volatility_score=volatility,
            amount_usd=amount_usd,
        )

    def assess_conversion_risk(
        self,
        from_curr: str,
        to_curr: str,
        amount: Decimal,
    ) -> CurrencyConversionRisk:
        """Assess the risk of converting *amount* from *from_curr* to *to_curr*."""
        src = from_curr.upper().strip()
        dst = to_curr.upper().strip()

        src_profile = self.assess_currency_risk(src, amount)
        dst_profile = self.assess_currency_risk(dst, amount)

        is_cross_border = self._is_cross_border(src, dst)
        jur_risk = (src_profile.base_risk_score + dst_profile.base_risk_score) / 2.0

        structuring = self.is_structuring(
            [src_profile.amount_usd], currency="USD", threshold=self._threshold_usd
        )

        notes: list[str] = []
        if src_profile.is_sanctioned or dst_profile.is_sanctioned:
            notes.append("One or both currencies are associated with sanctioned jurisdictions.")
        if src_profile.is_crypto or dst_profile.is_crypto:
            notes.append("Conversion involves a crypto or digital-asset currency.")
        if src_profile.is_high_risk_jurisdiction or dst_profile.is_high_risk_jurisdiction:
            notes.append("At least one currency originates from a FATF high-risk jurisdiction.")
        if is_cross_border:
            notes.append("Cross-border conversion detected between distinct regulatory zones.")
        if structuring:
            notes.append(
                f"Amount (USD {src_profile.amount_usd}) is within "
                f"{int(self._structuring_fraction * 100)}% of the "
                f"USD {self._threshold_usd} reporting threshold — potential structuring."
            )

        score = min(
            1.0,
            jur_risk
            + (0.15 if is_cross_border else 0.0)
            + (0.20 if structuring else 0.0)
            + (0.10 if src_profile.is_crypto or dst_profile.is_crypto else 0.0),
        )

        CONVERSION_ASSESSMENTS_TOTAL.labels(from_currency=src, to_currency=dst).inc()

        self._log.info(
            "conversion_risk_assessed",
            from_currency=src,
            to_currency=dst,
            score=score,
            structuring=structuring,
            cross_border=is_cross_border,
        )

        return CurrencyConversionRisk(
            from_currency=src,
            to_currency=dst,
            conversion_risk_score=round(score, 4),
            is_cross_border=is_cross_border,
            jurisdiction_risk=round(jur_risk, 4),
            structuring_indicator=structuring,
            amount_usd=src_profile.amount_usd,
            risk_notes=notes,
        )

    def normalize_amount(self, amount: Decimal, currency: str) -> Decimal:
        """Convert *amount* in *currency* to its USD equivalent."""
        code = currency.upper().strip()
        rate = _USD_RATES.get(code, Decimal("1.0"))
        return (amount * rate).quantize(Decimal("0.01"))

    def is_structuring(
        self,
        amounts: list[Decimal],
        currency: str,
        threshold: Decimal | None = None,
    ) -> bool:
        """Return True if any *amount* appears to be structured below *threshold*.

        Structuring (smurfing) is the practice of splitting transactions to stay
        just below reporting thresholds. We flag amounts in the band:
            threshold * (1 - structuring_fraction) <= amount < threshold
        """
        effective_threshold = threshold or self._threshold_usd
        lower_bound = effective_threshold * (1 - self._structuring_fraction)

        usd_amounts = [self.normalize_amount(a, currency) for a in amounts]

        flagged = any(lower_bound <= a < effective_threshold for a in usd_amounts)
        if flagged:
            STRUCTURING_DETECTIONS_TOTAL.labels(currency_code=currency.upper()).inc()
            self._log.warning(
                "structuring_pattern_detected",
                currency=currency.upper(),
                threshold_usd=str(effective_threshold),
                amounts_usd=[str(a) for a in usd_amounts],
            )
        return flagged

    def get_high_risk_currencies(self) -> list[str]:
        """Return all currency codes that score HIGH risk on their own profile."""
        high_risk: list[str] = []
        for code in _USD_RATES:
            profile = self.assess_currency_risk(code, Decimal("1"))
            if profile.risk_level == RiskLevel.HIGH:
                high_risk.append(code)
        return sorted(high_risk)

    def get_jurisdiction_risk(self, country_code: str) -> float:
        """Return the jurisdiction risk score (0.0 – 1.0) for an ISO 3166-1 alpha-2 code."""
        code = country_code.upper().strip()
        score = _JURISDICTION_RISK_SCORES.get(code)
        if score is None:
            # Unknown jurisdiction: apply a moderate default.
            self._log.debug("unknown_jurisdiction", country_code=code)
            return 0.30
        return score

    def detect_round_tripping(self, transactions: list[dict[str, Any]]) -> bool:
        """Detect round-tripping: funds leave and return via a different currency path.

        Expects each transaction dict to have at minimum:
            {"amount": Decimal, "currency": str, "direction": "in" | "out",
             "counterparty_country": str}

        Returns True when inflows and outflows are within 5 % of each other in USD
        and involve at least two distinct non-USD currencies or high-risk jurisdictions.
        """
        if len(transactions) < 2:
            return False

        in_usd = Decimal("0")
        out_usd = Decimal("0")
        currencies_seen: set[str] = set()
        high_risk_involved = False

        for tx in transactions:
            code = str(tx.get("currency", "USD")).upper()
            amount = tx.get("amount", Decimal("0"))
            direction = str(tx.get("direction", "out")).lower()
            country = str(tx.get("counterparty_country", "")).upper()

            usd = self.normalize_amount(amount, code)
            currencies_seen.add(code)

            if direction == "in":
                in_usd += usd
            else:
                out_usd += usd

            if country in _FATF_HIGH_RISK_COUNTRIES or code in _SANCTIONED_CURRENCIES:
                high_risk_involved = True

        if in_usd == 0 or out_usd == 0:
            return False

        # Ratio test: amounts should nearly cancel.
        ratio = float(min(in_usd, out_usd) / max(in_usd, out_usd))
        amount_match = ratio >= 0.95

        # Structural test: multiple currencies or high-risk routing.
        non_usd_currencies = currencies_seen - {"USD", "USDT", "USDC", "BUSD"}
        structural_indicator = len(non_usd_currencies) >= 2 or high_risk_involved

        detected = amount_match and structural_indicator
        if detected:
            ROUND_TRIP_DETECTIONS_TOTAL.inc()
            self._log.warning(
                "round_tripping_detected",
                in_usd=str(in_usd),
                out_usd=str(out_usd),
                ratio=round(ratio, 4),
                currencies=sorted(currencies_seen),
            )
        return detected

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_base_risk_score(
        self,
        *,
        is_sanctioned: bool,
        is_crypto: bool,
        is_high_risk_jurisdiction: bool,
        volatility: float,
        amount_usd: Decimal,
        known: bool,
    ) -> float:
        """Composite risk score assembled from individual signal weights."""
        score = 0.0

        if is_sanctioned:
            score += 0.50
        if is_high_risk_jurisdiction:
            score += 0.25
        if is_crypto:
            score += 0.15
        if not known:
            score += self._unknown_score * 0.20  # partial penalty for unknown codes

        # Volatility contributes up to 0.15 of the score.
        score += volatility * 0.15

        # Large amounts incrementally raise risk (up to +0.10).
        if amount_usd >= Decimal("100000"):
            score += 0.10
        elif amount_usd >= Decimal("10000"):
            score += 0.05

        return round(min(score, 1.0), 4)

    @staticmethod
    def _score_to_level(score: float) -> RiskLevel:
        if score >= 0.60:
            return RiskLevel.HIGH
        if score >= 0.30:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @staticmethod
    def _currency_in_high_risk_jurisdiction(code: str) -> bool:
        """Return True if *code* is issued by a FATF high-risk jurisdiction."""
        for country, currency in _COUNTRY_CURRENCY.items():
            if currency == code and country in _FATF_HIGH_RISK_COUNTRIES:
                return True
        # Also flag currencies whose ISO code implicitly maps to a high-risk country
        # e.g. MMK → MM, SYP → SY.
        if len(code) == 3:
            country_guess = code[:2]
            if country_guess in _FATF_HIGH_RISK_COUNTRIES:
                return True
        return False

    @staticmethod
    def _is_cross_border(src: str, dst: str) -> bool:
        """Return True when *src* and *dst* currencies belong to different major zones."""
        _ZONE_USD = {"USD", "CAD", "MXN", "BRL", "ARS"}
        _ZONE_EUR = {"EUR", "GBP", "CHF", "SEK", "NOK", "DKK"}
        _ZONE_ASIA = {"JPY", "CNY", "HKD", "SGD", "KRW", "THB", "MYR", "IDR", "PHP", "VND"}
        _ZONE_CRYPTO = _CRYPTO_CURRENCIES

        def _zone(code: str) -> str:
            if code in _ZONE_USD:
                return "usd_zone"
            if code in _ZONE_EUR:
                return "eur_zone"
            if code in _ZONE_ASIA:
                return "asia_zone"
            if code in _ZONE_CRYPTO:
                return "crypto"
            return "other"

        return _zone(src) != _zone(dst)


# ── Module-level singleton ────────────────────────────────────────────────────

_default_service: CurrencyRiskService | None = None


def get_currency_risk_service() -> CurrencyRiskService:
    """Return the module-level singleton, creating it on first call."""
    global _default_service
    if _default_service is None:
        _default_service = CurrencyRiskService()
    return _default_service
