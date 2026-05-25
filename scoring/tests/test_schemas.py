"""Schema validation and serialisation tests.

Covers Transaction, FeatureVector, and ScoredTransaction Pydantic models:
validation rules, field defaults, serialisation round-trips, and
rejection of invalid data.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

import orjson
import pytest
from pydantic import ValidationError

from shared.schemas import (
    AlertEvent,
    AlertSeverity,
    AlertStatus,
    FeatureVector,
    ScoredTransaction,
    Transaction,
    TransactionType,
)

# ---------------------------------------------------------------------------
# Transaction model
# ---------------------------------------------------------------------------


class TestTransactionValidation:
    """Pydantic validation for the Transaction schema."""

    def test_minimal_valid_transaction(self) -> None:
        """Only required fields should be sufficient."""
        txn = Transaction(
            user_id=uuid4(),
            amount=Decimal("42.50"),
            transaction_type=TransactionType.PURCHASE,
        )
        assert txn.currency == "USD"
        assert txn.merchant_id is None
        assert isinstance(txn.transaction_id, UUID)
        assert isinstance(txn.timestamp, datetime)

    def test_transaction_id_auto_generated(self) -> None:
        """Each Transaction should get a unique auto-generated UUID."""
        t1 = Transaction(
            user_id=uuid4(),
            amount=Decimal("1.00"),
            transaction_type=TransactionType.DEPOSIT,
        )
        t2 = Transaction(
            user_id=uuid4(),
            amount=Decimal("1.00"),
            transaction_type=TransactionType.DEPOSIT,
        )
        assert t1.transaction_id != t2.transaction_id

    def test_amount_must_be_non_negative(self) -> None:
        """Negative amount should raise ValidationError."""
        with pytest.raises(ValidationError):
            Transaction(
                user_id=uuid4(),
                amount=Decimal("-0.01"),
                transaction_type=TransactionType.PURCHASE,
            )

    def test_amount_zero_is_valid(self) -> None:
        """Zero amount is a valid edge case."""
        txn = Transaction(
            user_id=uuid4(),
            amount=Decimal("0.00"),
            transaction_type=TransactionType.REFUND,
        )
        assert txn.amount == Decimal("0.00")

    def test_currency_default_is_usd(self) -> None:
        """Default currency must be USD."""
        txn = Transaction(
            user_id=uuid4(),
            amount=Decimal("10.00"),
            transaction_type=TransactionType.TRANSFER,
        )
        assert txn.currency == "USD"

    def test_currency_max_length_enforced(self) -> None:
        """Currency longer than 3 characters should fail validation."""
        with pytest.raises(ValidationError):
            Transaction(
                user_id=uuid4(),
                amount=Decimal("10.00"),
                transaction_type=TransactionType.PURCHASE,
                currency="DOLLARS",
            )

    def test_all_transaction_types_accepted(self) -> None:
        """Every TransactionType enum value should be accepted."""
        for txn_type in TransactionType:
            txn = Transaction(
                user_id=uuid4(),
                amount=Decimal("1.00"),
                transaction_type=txn_type,
            )
            assert txn.transaction_type == txn_type

    def test_invalid_transaction_type_rejected(self) -> None:
        """Unknown transaction_type string should raise ValidationError."""
        with pytest.raises(ValidationError):
            Transaction(
                user_id=uuid4(),
                amount=Decimal("1.00"),
                transaction_type="airdrop",
            )

    def test_latitude_bounds_enforced(self) -> None:
        """Latitude outside [-90, 90] must be rejected."""
        with pytest.raises(ValidationError):
            Transaction(
                user_id=uuid4(),
                amount=Decimal("1.00"),
                transaction_type=TransactionType.PURCHASE,
                latitude=91.0,
            )

    def test_longitude_bounds_enforced(self) -> None:
        """Longitude outside [-180, 180] must be rejected."""
        with pytest.raises(ValidationError):
            Transaction(
                user_id=uuid4(),
                amount=Decimal("1.00"),
                transaction_type=TransactionType.PURCHASE,
                longitude=-181.0,
            )

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """metadata field should default to an empty dict."""
        txn = Transaction(
            user_id=uuid4(),
            amount=Decimal("1.00"),
            transaction_type=TransactionType.PURCHASE,
        )
        assert txn.metadata == {}

    def test_metadata_accepts_arbitrary_keys(self) -> None:
        """metadata should store any JSON-serialisable key-value pairs."""
        txn = Transaction(
            user_id=uuid4(),
            amount=Decimal("1.00"),
            transaction_type=TransactionType.PURCHASE,
            metadata={"channel": "mobile", "session_id": "abc123", "count": 5},
        )
        assert txn.metadata["channel"] == "mobile"


# ---------------------------------------------------------------------------
# FeatureVector defaults and validation
# ---------------------------------------------------------------------------


class TestFeatureVectorDefaults:
    """FeatureVector defaults and field constraints."""

    def test_minimal_feature_vector(self) -> None:
        """Only transaction_id and user_id are required."""
        fv = FeatureVector(transaction_id=uuid4(), user_id=uuid4())
        assert fv.txn_count_1h == 0
        assert fv.txn_count_24h == 0
        assert fv.txn_amount_sum_1h == Decimal("0")
        assert fv.txn_amount_avg_24h == Decimal("0")
        assert fv.distance_from_last_txn_km == 0.0
        assert fv.is_new_device is False
        assert fv.is_new_merchant is False
        assert fv.amount_zscore == 0.0
        assert fv.unique_countries_24h == 0

    def test_feature_vector_with_all_fields(self) -> None:
        """All fields should be settable without errors."""
        fv = FeatureVector(
            transaction_id=uuid4(),
            user_id=uuid4(),
            txn_count_1h=5,
            txn_count_24h=20,
            txn_amount_sum_1h=Decimal("500.00"),
            txn_amount_sum_24h=Decimal("2000.00"),
            txn_amount_avg_24h=Decimal("100.00"),
            unique_merchants_24h=3,
            unique_countries_24h=2,
            max_amount_24h=Decimal("400.00"),
            time_since_last_txn_seconds=120.0,
            distance_from_last_txn_km=250.0,
            is_new_device=True,
            is_new_merchant=True,
            amount_zscore=2.5,
            amount_to_avg_ratio=4.0,
        )
        assert fv.txn_count_1h == 5
        assert fv.is_new_device is True
        assert fv.unique_countries_24h == 2

    def test_timestamp_auto_populated(self) -> None:
        """timestamp field should default to a datetime."""
        fv = FeatureVector(transaction_id=uuid4(), user_id=uuid4())
        assert isinstance(fv.timestamp, datetime)


# ---------------------------------------------------------------------------
# ScoredTransaction serialisation
# ---------------------------------------------------------------------------


class TestScoredTransactionSerialization:
    """model_dump, model_dump(mode='json'), and JSON round-trip."""

    def _make_scored(self, score: float = 0.3) -> ScoredTransaction:
        uid = uuid4()
        txn_id = uuid4()
        return ScoredTransaction(
            transaction_id=txn_id,
            user_id=uid,
            amount=Decimal("75.00"),
            currency="EUR",
            transaction_type=TransactionType.TRANSFER,
            timestamp=datetime(2025, 6, 15, 10, 0, 0),
            fraud_score=score,
            model_version="ensemble-v1",
            scoring_latency_ms=12.5,
        )

    def test_model_dump_has_expected_keys(self) -> None:
        """model_dump should contain all schema-level fields."""
        st = self._make_scored()
        dumped = st.model_dump()
        required_keys = {
            "transaction_id",
            "user_id",
            "amount",
            "currency",
            "transaction_type",
            "timestamp",
            "fraud_score",
            "model_version",
            "scoring_latency_ms",
            "scored_at",
            "is_flagged",
        }
        assert required_keys.issubset(dumped.keys())

    def test_model_dump_json_mode_serializeable(self) -> None:
        """model_dump(mode='json') must produce a JSON-safe dict."""
        st = self._make_scored()
        json_dict = st.model_dump(mode="json")
        # Should not raise
        serialized = orjson.dumps(json_dict)
        assert isinstance(serialized, bytes)

    def test_fraud_score_bounds_accepted(self) -> None:
        """fraud_score at boundaries 0.0 and 1.0 must be valid."""
        for score in (0.0, 1.0):
            st = self._make_scored(score=score)
            assert st.fraud_score == score

    def test_fraud_score_out_of_range_rejected(self) -> None:
        """fraud_score > 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            self._make_scored(score=1.1)

    def test_negative_fraud_score_rejected(self) -> None:
        """Negative fraud_score must raise ValidationError."""
        with pytest.raises(ValidationError):
            self._make_scored(score=-0.01)

    def test_features_field_is_optional(self) -> None:
        """features field defaults to None."""
        st = self._make_scored()
        assert st.features is None

    def test_features_field_can_be_set(self) -> None:
        """features field accepts a FeatureVector instance."""
        uid = uuid4()
        txn_id = uuid4()
        fv = FeatureVector(transaction_id=txn_id, user_id=uid)
        st = ScoredTransaction(
            transaction_id=txn_id,
            user_id=uid,
            amount=Decimal("10.00"),
            currency="USD",
            transaction_type=TransactionType.PURCHASE,
            timestamp=datetime.utcnow(),
            fraud_score=0.5,
            model_version="ensemble-v1",
            scoring_latency_ms=5.0,
            features=fv,
        )
        assert st.features is not None
        assert st.features.transaction_id == txn_id


# ---------------------------------------------------------------------------
# orjson serialisation round-trip
# ---------------------------------------------------------------------------


class TestOrjsonRoundTrip:
    """to_json_bytes / from_json_bytes round-trip for all schema types."""

    def test_transaction_round_trip(self) -> None:
        """Transaction serialises to bytes and deserialises back correctly."""
        original = Transaction(
            user_id=uuid4(),
            amount=Decimal("123.45"),
            transaction_type=TransactionType.WITHDRAWAL,
            currency="GBP",
            ip_address="192.168.1.1",
        )
        raw = original.to_json_bytes()
        restored = Transaction.from_json_bytes(raw)
        assert restored.amount == original.amount
        assert restored.currency == original.currency
        assert restored.transaction_type == original.transaction_type

    def test_feature_vector_round_trip(self) -> None:
        """FeatureVector serialises and deserialises without data loss."""
        uid = uuid4()
        txn_id = uuid4()
        original = FeatureVector(
            transaction_id=txn_id,
            user_id=uid,
            txn_count_1h=8,
            txn_amount_avg_24h=Decimal("200.00"),
            is_new_device=True,
            unique_countries_24h=3,
        )
        raw = original.to_json_bytes()
        restored = FeatureVector.from_json_bytes(raw)
        assert restored.txn_count_1h == 8
        assert restored.is_new_device is True
        assert restored.unique_countries_24h == 3

    def test_scored_transaction_round_trip(self) -> None:
        """ScoredTransaction round-trips through orjson without loss."""
        uid = uuid4()
        txn_id = uuid4()
        original = ScoredTransaction(
            transaction_id=txn_id,
            user_id=uid,
            amount=Decimal("999.99"),
            currency="USD",
            transaction_type=TransactionType.PURCHASE,
            timestamp=datetime(2025, 6, 15, 12, 0, 0),
            fraud_score=0.85,
            model_version="ensemble-v1",
            scoring_latency_ms=7.3,
            is_flagged=True,
            flag_reason="block | high_amount",
        )
        raw = original.to_json_bytes()
        restored = ScoredTransaction.from_json_bytes(raw)
        assert restored.fraud_score == pytest.approx(0.85)
        assert restored.is_flagged is True
        assert restored.flag_reason == "block | high_amount"

    def test_to_json_bytes_returns_bytes(self) -> None:
        """to_json_bytes must return a bytes object."""
        txn = Transaction(
            user_id=uuid4(),
            amount=Decimal("1.00"),
            transaction_type=TransactionType.DEPOSIT,
        )
        assert isinstance(txn.to_json_bytes(), bytes)

    def test_json_bytes_parseable_by_orjson(self) -> None:
        """to_json_bytes output should be valid JSON readable by orjson."""
        txn = Transaction(
            user_id=uuid4(),
            amount=Decimal("55.55"),
            transaction_type=TransactionType.PURCHASE,
        )
        parsed = orjson.loads(txn.to_json_bytes())
        assert isinstance(parsed, dict)
        assert "amount" in parsed

    def test_invalid_bytes_raise_on_deserialise(self) -> None:
        """from_json_bytes should raise when data is garbage."""
        with pytest.raises(Exception):
            Transaction.from_json_bytes(b"not-json-at-all")
