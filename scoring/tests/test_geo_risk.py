from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scoring.services.geo_risk import (
    GeoLocation,
    GeoRiskLevel,
    GeoRiskService,
    TransactionGeoData,
    get_geo_risk_service,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

_NYC = GeoLocation(latitude=40.7128, longitude=-74.0060, country_code="US")
_LONDON = GeoLocation(latitude=51.5074, longitude=-0.1278, country_code="GB")
_TOKYO = GeoLocation(latitude=35.6762, longitude=139.6503, country_code="JP")
_PARIS = GeoLocation(latitude=48.8566, longitude=2.3522, country_code="FR")
_PYONGYANG = GeoLocation(latitude=39.0392, longitude=125.7625, country_code="KP")
_TEHRAN = GeoLocation(latitude=35.6892, longitude=51.3890, country_code="IR")
_LAGOS = GeoLocation(latitude=6.5244, longitude=3.3792, country_code="NG")
_SAO_PAULO = GeoLocation(latitude=-23.5505, longitude=-46.6333, country_code="BR")

_NOW = datetime(2026, 6, 8, 14, 0, 0, tzinfo=UTC)


def _make_txn(
    location: GeoLocation,
    timestamp: datetime | None = None,
    user_id: str = "user-1",
    transaction_id: str = "txn-1",
) -> TransactionGeoData:
    return TransactionGeoData(
        transaction_id=transaction_id,
        location=location,
        timestamp=timestamp or _NOW,
        user_id=user_id,
    )


# ── Distance calculation ─────────────────────────────────────────────────────


class TestCalculateDistance:
    def test_same_location_returns_zero(self) -> None:
        svc = GeoRiskService()
        assert svc.calculate_distance(_NYC, _NYC) == pytest.approx(0.0, abs=0.01)

    def test_nyc_to_london(self) -> None:
        svc = GeoRiskService()
        distance = svc.calculate_distance(_NYC, _LONDON)
        # Expected ~5570 km
        assert 5500 < distance < 5700

    def test_tokyo_to_nyc(self) -> None:
        svc = GeoRiskService()
        distance = svc.calculate_distance(_TOKYO, _NYC)
        # Expected ~10850 km
        assert 10700 < distance < 11000

    def test_symmetry(self) -> None:
        svc = GeoRiskService()
        d1 = svc.calculate_distance(_PARIS, _TOKYO)
        d2 = svc.calculate_distance(_TOKYO, _PARIS)
        assert d1 == pytest.approx(d2, rel=1e-6)


# ── Impossible travel detection ──────────────────────────────────────────────


class TestImpossibleTravel:
    def test_possible_travel_long_gap(self) -> None:
        """NYC -> London in 8 hours is possible by plane."""
        svc = GeoRiskService()
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        curr = _make_txn(_LONDON, _NOW + timedelta(hours=8), transaction_id="txn-curr")
        result = svc.check_impossible_travel(curr, prev)
        assert not result.is_impossible
        assert result.distance_km > 5000

    def test_impossible_travel_short_gap(self) -> None:
        """NYC -> Tokyo in 30 minutes is impossible."""
        svc = GeoRiskService()
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        curr = _make_txn(_TOKYO, _NOW + timedelta(minutes=30), transaction_id="txn-curr")
        result = svc.check_impossible_travel(curr, prev)
        assert result.is_impossible
        assert result.required_speed_kmh > 1200

    def test_very_short_time_gap_skipped(self) -> None:
        """Transactions within 60 seconds are not flagged as impossible."""
        svc = GeoRiskService()
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        curr = _make_txn(_TOKYO, _NOW + timedelta(seconds=30), transaction_id="txn-curr")
        result = svc.check_impossible_travel(curr, prev)
        assert not result.is_impossible
        assert result.required_speed_kmh == 0.0

    def test_same_location_not_impossible(self) -> None:
        svc = GeoRiskService()
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        curr = _make_txn(_NYC, _NOW + timedelta(minutes=5), transaction_id="txn-curr")
        result = svc.check_impossible_travel(curr, prev)
        assert not result.is_impossible

    def test_batch_detection(self) -> None:
        """Batch check with three transactions: first pair possible, second impossible."""
        svc = GeoRiskService()
        txns = [
            _make_txn(_NYC, _NOW, transaction_id="t1"),
            _make_txn(_LONDON, _NOW + timedelta(hours=8), transaction_id="t2"),
            _make_txn(_TOKYO, _NOW + timedelta(hours=8, minutes=15), transaction_id="t3"),
        ]
        results = svc.detect_impossible_travel_batch(txns)
        assert len(results) == 2
        assert not results[0].is_impossible  # NYC -> London in 8h
        assert results[1].is_impossible       # London -> Tokyo in 15min

    def test_custom_max_speed(self) -> None:
        """With a very high max speed, nothing is impossible."""
        svc = GeoRiskService(max_plausible_speed_kmh=200_000.0)
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        curr = _make_txn(_TOKYO, _NOW + timedelta(minutes=5), transaction_id="txn-curr")
        result = svc.check_impossible_travel(curr, prev)
        assert not result.is_impossible


# ── Country risk scoring ─────────────────────────────────────────────────────


class TestCountryRisk:
    def test_tier_1_country(self) -> None:
        svc = GeoRiskService()
        assert svc.get_country_risk_score("KP") == 1.0
        assert svc.get_country_risk_score("IR") == 1.0

    def test_tier_2_country(self) -> None:
        svc = GeoRiskService()
        assert svc.get_country_risk_score("AF") == 0.75
        assert svc.get_country_risk_score("RU") == 0.75

    def test_tier_3_country(self) -> None:
        svc = GeoRiskService()
        assert svc.get_country_risk_score("NG") == 0.45

    def test_unlisted_country_defaults_to_tier_4(self) -> None:
        svc = GeoRiskService()
        assert svc.get_country_risk_score("US") == 0.10
        assert svc.get_country_risk_score("DE") == 0.10

    def test_case_insensitive(self) -> None:
        svc = GeoRiskService()
        assert svc.get_country_risk_score("kp") == svc.get_country_risk_score("KP")

    def test_custom_tiers(self) -> None:
        custom_tiers = {1: frozenset({"XX"})}
        custom_scores = {1: 0.99, 4: 0.01}
        svc = GeoRiskService(country_risk_tiers=custom_tiers, tier_scores=custom_scores)
        assert svc.get_country_risk_score("XX") == 0.99
        assert svc.get_country_risk_score("US") == 0.01


# ── Cross-border detection ───────────────────────────────────────────────────


class TestCrossBorder:
    def test_same_country(self) -> None:
        svc = GeoRiskService()
        assert not svc.is_cross_border("US", "US")

    def test_different_countries(self) -> None:
        svc = GeoRiskService()
        assert svc.is_cross_border("US", "GB")

    def test_case_insensitive(self) -> None:
        svc = GeoRiskService()
        assert not svc.is_cross_border("us", "US")
        assert svc.is_cross_border("us", "gb")


# ── Timezone anomaly detection ───────────────────────────────────────────────


class TestTimezoneAnomaly:
    def test_normal_daytime_no_anomaly(self) -> None:
        """14:00 UTC in the US (e.g. 9 AM ET) is normal."""
        svc = GeoRiskService()
        ts = datetime(2026, 6, 8, 14, 0, 0, tzinfo=UTC)
        assert not svc.detect_timezone_anomaly(ts, "US")

    def test_deep_night_anomaly(self) -> None:
        """01:00 UTC is 2-3 AM in GB — anomalous."""
        svc = GeoRiskService()
        ts = datetime(2026, 6, 8, 1, 0, 0, tzinfo=UTC)
        assert svc.detect_timezone_anomaly(ts, "GB")

    def test_multi_timezone_country_no_anomaly(self) -> None:
        """Even at 02:00 UTC, some US timezones are in evening hours."""
        svc = GeoRiskService()
        # 02:00 UTC => e.g. 6 PM previous day in Hawaii (-10)
        ts = datetime(2026, 6, 8, 15, 0, 0, tzinfo=UTC)
        assert not svc.detect_timezone_anomaly(ts, "US")

    def test_unknown_country_no_anomaly(self) -> None:
        """Unknown countries return no anomaly (insufficient data)."""
        svc = GeoRiskService()
        ts = datetime(2026, 6, 8, 2, 0, 0, tzinfo=UTC)
        assert not svc.detect_timezone_anomaly(ts, "ZZ")

    def test_japan_late_night_anomaly(self) -> None:
        """18:00 UTC is 03:00 JST — anomalous for Japan."""
        svc = GeoRiskService()
        ts = datetime(2026, 6, 8, 18, 0, 0, tzinfo=UTC)
        assert svc.detect_timezone_anomaly(ts, "JP")


# ── Full assessment ──────────────────────────────────────────────────────────


class TestAssess:
    def test_low_risk_domestic(self) -> None:
        """A domestic US transaction during daytime should be low risk."""
        svc = GeoRiskService()
        txn = _make_txn(_NYC, _NOW)
        result = svc.assess(txn)
        assert result.risk_level in (GeoRiskLevel.LOW, GeoRiskLevel.MEDIUM)
        assert not result.is_cross_border
        assert result.country_risk_tier == 4

    def test_high_risk_country(self) -> None:
        """Transaction from a tier-1 country should produce a high score."""
        svc = GeoRiskService()
        txn = _make_txn(_PYONGYANG, _NOW)
        result = svc.assess(txn)
        assert result.risk_score >= 0.5
        assert "high_risk_country" in result.risk_signals
        assert result.country_risk_tier == 1

    def test_cross_border_flagged(self) -> None:
        svc = GeoRiskService()
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        curr = _make_txn(_LONDON, _NOW + timedelta(hours=10), transaction_id="txn-curr")
        result = svc.assess(curr, prev)
        assert result.is_cross_border
        assert "cross_border_transaction" in result.risk_signals

    def test_impossible_travel_in_assessment(self) -> None:
        """Impossible travel should elevate the risk score significantly."""
        svc = GeoRiskService()
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        curr = _make_txn(_TOKYO, _NOW + timedelta(minutes=20), transaction_id="txn-curr")
        result = svc.assess(curr, prev)
        assert result.impossible_travel is not None
        assert result.impossible_travel.is_impossible
        assert "impossible_travel" in result.risk_signals
        assert result.risk_score >= 0.5

    def test_timezone_anomaly_in_assessment(self) -> None:
        """Transaction at 3 AM local time should flag timezone anomaly."""
        svc = GeoRiskService()
        # 18:00 UTC => 03:00 JST
        txn = _make_txn(_TOKYO, datetime(2026, 6, 8, 18, 0, 0, tzinfo=UTC))
        result = svc.assess(txn)
        assert result.timezone_anomaly
        assert "timezone_anomaly" in result.risk_signals

    def test_no_previous_transaction(self) -> None:
        """Assessment without a previous transaction should still work."""
        svc = GeoRiskService()
        txn = _make_txn(_PARIS, _NOW)
        result = svc.assess(txn)
        assert not result.is_cross_border
        assert result.impossible_travel is None

    def test_combined_signals_elevate_score(self) -> None:
        """Impossible travel from a high-risk country should produce critical score."""
        svc = GeoRiskService()
        prev = _make_txn(_NYC, _NOW, transaction_id="txn-prev")
        # Pyongyang in 20 min from NYC: impossible travel + tier 1 country
        curr = _make_txn(
            _PYONGYANG,
            _NOW + timedelta(minutes=20),
            transaction_id="txn-curr",
        )
        result = svc.assess(curr, prev)
        assert result.risk_level in (GeoRiskLevel.HIGH, GeoRiskLevel.CRITICAL)
        assert "impossible_travel" in result.risk_signals
        assert "high_risk_country" in result.risk_signals


# ── Singleton ────────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_geo_risk_service_returns_same_instance(self) -> None:
        svc1 = get_geo_risk_service()
        svc2 = get_geo_risk_service()
        assert svc1 is svc2

    def test_singleton_is_geo_risk_service(self) -> None:
        svc = get_geo_risk_service()
        assert isinstance(svc, GeoRiskService)
