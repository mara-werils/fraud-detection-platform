"""Tests for MultiCurrencyRiskEngine."""

from decimal import Decimal

import pytest

from scoring.services.multi_currency import (
    MultiCurrencyRiskEngine,
    PatternType,
    RiskTier,
    get_multi_currency_engine,
)


@pytest.fixture
def engine() -> MultiCurrencyRiskEngine:
    return MultiCurrencyRiskEngine(cache_ttl_seconds=60.0)


# ── Exchange rate cache ──────────────────────────────────────────────────────


class TestExchangeRateCache:
    def test_get_known_rate(self, engine: MultiCurrencyRiskEngine):
        rate = engine.get_rate("EUR")
        assert rate == Decimal("1.08")

    def test_get_unknown_rate_defaults_to_one(self, engine: MultiCurrencyRiskEngine):
        rate = engine.get_rate("XYZ")
        assert rate == Decimal("1.0")

    def test_case_insensitive_lookup(self, engine: MultiCurrencyRiskEngine):
        assert engine.get_rate("eur") == engine.get_rate("EUR")

    def test_update_rate(self, engine: MultiCurrencyRiskEngine):
        engine.update_rate("EUR", Decimal("1.15"))
        assert engine.get_rate("EUR") == Decimal("1.15")

    def test_is_rate_cached(self, engine: MultiCurrencyRiskEngine):
        assert engine.is_rate_cached("USD") is True
        # Unknown currency gets cached on first get.
        engine.get_rate("UNKNOWN")
        assert engine.is_rate_cached("UNKNOWN") is True

    def test_cache_expiry(self):
        engine = MultiCurrencyRiskEngine(cache_ttl_seconds=0.0)
        # With a 0s TTL every fetch refreshes from defaults.
        rate = engine.get_rate("GBP")
        assert rate == Decimal("1.27")


# ── Amount normalisation ─────────────────────────────────────────────────────


class TestNormalization:
    def test_usd_passthrough(self, engine: MultiCurrencyRiskEngine):
        assert engine.normalize_to_usd(Decimal("100"), "USD") == Decimal("100.00")

    def test_eur_to_usd(self, engine: MultiCurrencyRiskEngine):
        result = engine.normalize_to_usd(Decimal("100"), "EUR")
        assert result == Decimal("108.00")

    def test_jpy_to_usd(self, engine: MultiCurrencyRiskEngine):
        result = engine.normalize_to_usd(Decimal("100000"), "JPY")
        assert result == Decimal("670.00")

    def test_unknown_currency_treated_as_one_to_one(self, engine: MultiCurrencyRiskEngine):
        result = engine.normalize_to_usd(Decimal("500"), "FAKE")
        assert result == Decimal("500.00")


# ── Currency mismatch scoring ────────────────────────────────────────────────


class TestCurrencyMismatch:
    def test_same_currency_no_mismatch(self, engine: MultiCurrencyRiskEngine):
        result = engine.score_currency_mismatch("USD", "USD")
        assert result.mismatch_score == 0.0
        assert result.same_zone is True
        assert result.risk_tier == RiskTier.LOW

    def test_same_zone_lower_score(self, engine: MultiCurrencyRiskEngine):
        result = engine.score_currency_mismatch("EUR", "GBP")
        assert result.same_zone is True
        # Same zone, no sanctioned/crypto => score should be low.
        assert result.mismatch_score < 0.25

    def test_cross_zone_higher_score(self, engine: MultiCurrencyRiskEngine):
        result = engine.score_currency_mismatch("USD", "JPY")
        assert result.same_zone is False
        assert result.mismatch_score > 0.0

    def test_high_risk_pair_flagged(self, engine: MultiCurrencyRiskEngine):
        result = engine.score_currency_mismatch("RUB", "USDT")
        assert result.is_high_risk_pair is True
        assert result.mismatch_score >= 0.35

    def test_sanctioned_currency_raises_score(self, engine: MultiCurrencyRiskEngine):
        result = engine.score_currency_mismatch("USD", "IRR")
        assert result.mismatch_score >= 0.25
        assert any("Sanctioned" in n for n in result.notes)

    def test_crypto_involvement(self, engine: MultiCurrencyRiskEngine):
        result = engine.score_currency_mismatch("USD", "BTC")
        assert result.mismatch_score > 0.0
        assert any("Crypto" in n or "digital" in n for n in result.notes)


# ── High-risk pair identification ────────────────────────────────────────────


class TestHighRiskPairs:
    def test_find_known_pair(self, engine: MultiCurrencyRiskEngine):
        pairs = engine.identify_high_risk_pairs(["RUB", "USDT", "USD"])
        assert ("RUB", "USDT") in pairs
        assert ("USDT", "RUB") in pairs

    def test_no_pairs_for_safe_currencies(self, engine: MultiCurrencyRiskEngine):
        pairs = engine.identify_high_risk_pairs(["USD", "EUR", "GBP"])
        assert pairs == []

    def test_single_currency_no_pairs(self, engine: MultiCurrencyRiskEngine):
        pairs = engine.identify_high_risk_pairs(["USD"])
        assert pairs == []


# ── Cross-currency pattern detection ─────────────────────────────────────────


class TestCrossCurrencyPatterns:
    def test_rapid_conversion(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("1000"), "currency": "USD", "timestamp": 1000.0},
            {"amount": Decimal("900"), "currency": "EUR", "timestamp": 1100.0},
            {"amount": Decimal("800"), "currency": "GBP", "timestamp": 1200.0},
        ]
        patterns = engine.detect_cross_currency_patterns(txns)
        types = [p.pattern_type for p in patterns]
        assert PatternType.RAPID_CONVERSION in types

    def test_currency_hopping(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("500"), "currency": "USD", "timestamp": 1.0},
            {"amount": Decimal("500"), "currency": "EUR", "timestamp": 2.0},
            {"amount": Decimal("500"), "currency": "GBP", "timestamp": 3.0},
            {"amount": Decimal("500"), "currency": "JPY", "timestamp": 4.0},
        ]
        patterns = engine.detect_cross_currency_patterns(txns)
        types = [p.pattern_type for p in patterns]
        assert PatternType.CURRENCY_HOPPING in types

    def test_split_and_convert(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("100"), "currency": "USD", "timestamp": 1.0},
            {"amount": Decimal("100"), "currency": "USD", "timestamp": 2.0},
            {"amount": Decimal("100"), "currency": "USD", "timestamp": 3.0},
            {"amount": Decimal("300"), "currency": "EUR", "timestamp": 4.0},
        ]
        patterns = engine.detect_cross_currency_patterns(txns)
        types = [p.pattern_type for p in patterns]
        assert PatternType.SPLIT_AND_CONVERT in types

    def test_no_patterns_for_single_currency(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("100"), "currency": "USD", "timestamp": 1.0},
            {"amount": Decimal("200"), "currency": "USD", "timestamp": 2.0},
        ]
        patterns = engine.detect_cross_currency_patterns(txns)
        assert patterns == []

    def test_too_few_transactions(self, engine: MultiCurrencyRiskEngine):
        txns = [{"amount": Decimal("100"), "currency": "USD", "timestamp": 1.0}]
        assert engine.detect_cross_currency_patterns(txns) == []


# ── Smurfing detection ───────────────────────────────────────────────────────


class TestSmurfingDetection:
    def test_near_threshold_across_currencies(self, engine: MultiCurrencyRiskEngine):
        # Two amounts near the USD 10,000 threshold in different currencies.
        txns = [
            {"amount": Decimal("9500"), "currency": "USD"},
            {"amount": Decimal("8800"), "currency": "EUR"},  # ~9504 USD
        ]
        result = engine.detect_smurfing(txns)
        assert result.detected is True
        assert len(result.flagged_amounts_usd) >= 1
        assert result.confidence > 0.0

    def test_aggregate_exceeds_threshold(self, engine: MultiCurrencyRiskEngine):
        # Individual amounts below threshold but aggregate above.
        txns = [
            {"amount": Decimal("5000"), "currency": "USD"},
            {"amount": Decimal("3000"), "currency": "EUR"},  # ~3240 USD
            {"amount": Decimal("400000"), "currency": "JPY"},  # ~2680 USD
        ]
        result = engine.detect_smurfing(txns)
        assert result.detected is True
        assert result.total_amount_usd >= Decimal("10000")

    def test_no_smurfing_for_small_amounts(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("50"), "currency": "USD"},
            {"amount": Decimal("30"), "currency": "EUR"},
        ]
        result = engine.detect_smurfing(txns)
        assert result.detected is False

    def test_single_large_transaction_no_smurfing(self, engine: MultiCurrencyRiskEngine):
        txns = [{"amount": Decimal("15000"), "currency": "USD"}]
        result = engine.detect_smurfing(txns)
        # Single txn above threshold is not smurfing (not split).
        assert result.detected is False

    def test_custom_threshold(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("4500"), "currency": "USD"},
            {"amount": Decimal("4500"), "currency": "USD"},
        ]
        result = engine.detect_smurfing(txns, threshold_usd=Decimal("5000"))
        assert result.detected is True


# ── Full assessment ──────────────────────────────────────────────────────────


class TestFullAssessment:
    def test_benign_transactions(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("50"), "currency": "USD", "timestamp": 1.0},
            {"amount": Decimal("30"), "currency": "USD", "timestamp": 2.0},
        ]
        result = engine.assess(txns, home_currency="USD")
        assert result.risk_tier == RiskTier.LOW
        assert result.overall_risk_score < 0.25

    def test_suspicious_multi_currency(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("9500"), "currency": "RUB", "timestamp": 1.0},
            {"amount": Decimal("9000"), "currency": "USDT", "timestamp": 2.0},
            {"amount": Decimal("8500"), "currency": "CNY", "timestamp": 3.0},
        ]
        result = engine.assess(txns, home_currency="USD")
        assert result.overall_risk_score > 0.0
        assert len(result.high_risk_pairs_found) > 0

    def test_assessment_without_home_currency(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("100"), "currency": "EUR", "timestamp": 1.0},
            {"amount": Decimal("200"), "currency": "GBP", "timestamp": 2.0},
        ]
        result = engine.assess(txns)
        assert result.mismatch is None
        assert result.normalized_amount_usd > Decimal("0")

    def test_assessment_includes_patterns_and_smurfing(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("9500"), "currency": "USD", "timestamp": 1.0},
            {"amount": Decimal("8800"), "currency": "EUR", "timestamp": 100.0},
            {"amount": Decimal("9200"), "currency": "GBP", "timestamp": 200.0},
        ]
        result = engine.assess(txns, home_currency="USD")
        # Should have patterns (rapid conversion / currency hopping) and smurfing.
        assert result.smurfing is not None
        assert len(result.notes) > 0

    def test_normalized_amount_correct(self, engine: MultiCurrencyRiskEngine):
        txns = [
            {"amount": Decimal("100"), "currency": "USD", "timestamp": 1.0},
            {"amount": Decimal("100"), "currency": "EUR", "timestamp": 2.0},
        ]
        result = engine.assess(txns)
        expected = Decimal("100.00") + Decimal("108.00")
        assert result.normalized_amount_usd == expected


# ── Singleton ────────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_multi_currency_engine()
        b = get_multi_currency_engine()
        assert a is b
