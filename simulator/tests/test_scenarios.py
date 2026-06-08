"""Tests for simulator.scenarios fraud simulation classes."""

from __future__ import annotations

from decimal import Decimal

import pytest

from shared.schemas import Transaction, TransactionType
from simulator.generators import build_user_pool
from simulator.scenarios import (
    SCENARIO_CLASSES,
    AccountTakeoverSimulation,
    CardTestingAttack,
    FriendlyFraudSimulation,
    MerchantCollusionSimulation,
    MoneyMuleChain,
    RefundAbuseSimulation,
    SyntheticIdentityFraud,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def user():
    """Create a single simulated user for scenario tests."""
    pool = build_user_pool(1)
    return pool[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_all_transactions(txns: list[Transaction]) -> None:
    """Common assertions that apply to every scenario output."""
    assert isinstance(txns, list)
    assert len(txns) > 0
    for txn in txns:
        assert isinstance(txn, Transaction)
        assert txn.metadata.get("is_fraud") is True
        assert "fraud_scenario" in txn.metadata
        assert txn.amount >= Decimal("0")


def _assert_sorted_by_time(txns: list[Transaction]) -> None:
    """Verify transactions are sorted chronologically."""
    timestamps = [t.timestamp for t in txns]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# CardTestingAttack
# ---------------------------------------------------------------------------


class TestCardTestingAttack:
    def test_generates_probes_and_cashout(self, user):
        scenario = CardTestingAttack(user, probe_count=10)
        txns = scenario.generate()

        _assert_all_transactions(txns)
        _assert_sorted_by_time(txns)
        # 10 probes + 1 cashout
        assert len(txns) == 11

    def test_probes_are_small_amounts(self, user):
        scenario = CardTestingAttack(user, probe_count=5)
        txns = scenario.generate()

        probes = [t for t in txns if t.metadata.get("phase") == "probe"]
        assert len(probes) == 5
        for p in probes:
            assert p.amount <= Decimal("5.00")

    def test_cashout_is_large(self, user):
        scenario = CardTestingAttack(user)
        txns = scenario.generate()

        cashouts = [t for t in txns if t.metadata.get("phase") == "cashout"]
        assert len(cashouts) == 1
        assert cashouts[0].amount >= Decimal("1500")

    def test_default_probe_count_range(self, user):
        scenario = CardTestingAttack(user)
        txns = scenario.generate()
        # Default 8-20 probes + 1 cashout
        assert 9 <= len(txns) <= 21

    def test_all_share_same_user_id(self, user):
        txns = CardTestingAttack(user).generate()
        for t in txns:
            assert t.user_id == user.user_id

    def test_scenario_label(self, user):
        txns = CardTestingAttack(user).generate()
        for t in txns:
            assert t.metadata["fraud_scenario"] == "card_testing_attack"


# ---------------------------------------------------------------------------
# AccountTakeoverSimulation
# ---------------------------------------------------------------------------


class TestAccountTakeoverSimulation:
    def test_generates_three_steps(self, user):
        txns = AccountTakeoverSimulation(user).generate()

        _assert_all_transactions(txns)
        _assert_sorted_by_time(txns)
        assert len(txns) == 3

    def test_step_sequence(self, user):
        txns = AccountTakeoverSimulation(user).generate()
        steps = [t.metadata["step"] for t in txns]
        assert steps == ["login", "settings_change", "transfer"]

    def test_login_uses_new_device(self, user):
        txns = AccountTakeoverSimulation(user).generate()
        login = txns[0]
        assert login.metadata.get("new_device") is True
        assert login.metadata.get("new_ip") is True

    def test_transfer_is_high_value(self, user):
        txns = AccountTakeoverSimulation(user).generate()
        transfer = [t for t in txns if t.metadata.get("step") == "transfer"][0]
        assert transfer.transaction_type == TransactionType.TRANSFER
        assert transfer.amount > Decimal(str(user.avg_spend))

    def test_scenario_label(self, user):
        txns = AccountTakeoverSimulation(user).generate()
        for t in txns:
            assert t.metadata["fraud_scenario"] == "account_takeover"

    def test_attacker_uses_consistent_device(self, user):
        txns = AccountTakeoverSimulation(user).generate()
        device_ids = {t.device_id for t in txns}
        # Attacker uses the same new device across all steps
        assert len(device_ids) == 1
        assert device_ids.pop() != user.device_id


# ---------------------------------------------------------------------------
# FriendlyFraudSimulation
# ---------------------------------------------------------------------------


class TestFriendlyFraudSimulation:
    def test_generates_purchase_and_chargeback(self, user):
        txns = FriendlyFraudSimulation(user).generate()

        _assert_all_transactions(txns)
        assert len(txns) == 2

    def test_purchase_then_refund(self, user):
        txns = FriendlyFraudSimulation(user).generate()
        assert txns[0].transaction_type == TransactionType.PURCHASE
        assert txns[1].transaction_type == TransactionType.REFUND

    def test_same_amount(self, user):
        txns = FriendlyFraudSimulation(user).generate()
        assert txns[0].amount == txns[1].amount

    def test_same_merchant(self, user):
        txns = FriendlyFraudSimulation(user).generate()
        assert txns[0].merchant_id == txns[1].merchant_id

    def test_chargeback_after_purchase(self, user):
        txns = FriendlyFraudSimulation(user).generate()
        assert txns[1].timestamp > txns[0].timestamp

    def test_uses_user_device(self, user):
        txns = FriendlyFraudSimulation(user).generate()
        for t in txns:
            assert t.device_id == user.device_id

    def test_scenario_label(self, user):
        txns = FriendlyFraudSimulation(user).generate()
        for t in txns:
            assert t.metadata["fraud_scenario"] == "friendly_fraud"


# ---------------------------------------------------------------------------
# SyntheticIdentityFraud
# ---------------------------------------------------------------------------


class TestSyntheticIdentityFraud:
    def test_generates_buildup_and_bustout(self):
        scenario = SyntheticIdentityFraud(buildup_count=4)
        txns = scenario.generate()

        _assert_all_transactions(txns)
        _assert_sorted_by_time(txns)
        # 4 buildup + 1 bustout
        assert len(txns) == 5

    def test_buildup_transactions_are_small(self):
        txns = SyntheticIdentityFraud(buildup_count=5).generate()
        buildups = [t for t in txns if t.metadata.get("step") == "credit_building"]
        assert len(buildups) == 5
        for b in buildups:
            assert b.amount <= Decimal("80.00")

    def test_bustout_is_large(self):
        txns = SyntheticIdentityFraud(buildup_count=3).generate()
        bustouts = [t for t in txns if t.metadata.get("step") == "bustout"]
        assert len(bustouts) == 1
        assert bustouts[0].amount >= Decimal("3000")

    def test_synthetic_name_present(self):
        txns = SyntheticIdentityFraud().generate()
        for t in txns:
            assert "synthetic_name" in t.metadata
            # Name should have first and last name
            name = t.metadata["synthetic_name"]
            assert " " in name

    def test_all_share_synthetic_user_id(self):
        txns = SyntheticIdentityFraud().generate()
        user_ids = {t.user_id for t in txns}
        assert len(user_ids) == 1

    def test_scenario_label(self):
        txns = SyntheticIdentityFraud().generate()
        for t in txns:
            assert t.metadata["fraud_scenario"] == "synthetic_identity"


# ---------------------------------------------------------------------------
# MoneyMuleChain
# ---------------------------------------------------------------------------


class TestMoneyMuleChain:
    def test_generates_chain_plus_withdrawal(self):
        scenario = MoneyMuleChain(chain_length=4, initial_amount=10000.0)
        txns = scenario.generate()

        _assert_all_transactions(txns)
        _assert_sorted_by_time(txns)
        # 4 hops + 1 final withdrawal
        assert len(txns) == 5

    def test_amounts_decrease_along_chain(self):
        txns = MoneyMuleChain(chain_length=4, initial_amount=10000.0).generate()
        hop_txns = [t for t in txns if t.metadata.get("step") != "final_withdrawal"]
        amounts = [float(t.amount) for t in hop_txns]
        # Each hop skims, so amounts should decrease
        for i in range(1, len(amounts)):
            assert amounts[i] < amounts[i - 1]

    def test_final_withdrawal(self):
        txns = MoneyMuleChain(chain_length=3).generate()
        final = [t for t in txns if t.metadata.get("step") == "final_withdrawal"]
        assert len(final) == 1
        assert final[0].transaction_type == TransactionType.WITHDRAWAL

    def test_transfers_use_transfer_type(self):
        txns = MoneyMuleChain(chain_length=3).generate()
        hops = [t for t in txns if t.metadata.get("step") != "final_withdrawal"]
        for h in hops:
            assert h.transaction_type == TransactionType.TRANSFER

    def test_each_hop_has_different_sender(self):
        txns = MoneyMuleChain(chain_length=4).generate()
        hops = [t for t in txns if t.metadata.get("step") != "final_withdrawal"]
        senders = [t.metadata["sender_id"] for t in hops]
        assert len(set(senders)) == len(senders)

    def test_default_chain_length_range(self):
        txns = MoneyMuleChain().generate()
        # Default 3-5 hops + 1 withdrawal
        assert 4 <= len(txns) <= 6

    def test_scenario_label(self):
        txns = MoneyMuleChain().generate()
        for t in txns:
            assert t.metadata["fraud_scenario"] == "money_mule_chain"


# ---------------------------------------------------------------------------
# RefundAbuseSimulation
# ---------------------------------------------------------------------------


class TestRefundAbuseSimulation:
    def test_generates_purchase_refund_pairs(self, user):
        scenario = RefundAbuseSimulation(user, cycle_count=3)
        txns = scenario.generate()

        _assert_all_transactions(txns)
        _assert_sorted_by_time(txns)
        assert len(txns) == 6  # 3 purchases + 3 refunds

    def test_each_cycle_has_matching_amounts(self, user):
        scenario = RefundAbuseSimulation(user, cycle_count=4)
        txns = scenario.generate()

        purchases = [t for t in txns if t.metadata.get("step") == "purchase"]
        refunds = [t for t in txns if t.metadata.get("step") == "refund"]
        assert len(purchases) == 4
        assert len(refunds) == 4

        # Match by cycle number and check amounts
        for cycle in range(4):
            p = [t for t in purchases if t.metadata.get("cycle") == cycle][0]
            r = [t for t in refunds if t.metadata.get("cycle") == cycle][0]
            assert p.amount == r.amount

    def test_refund_reasons_present(self, user):
        txns = RefundAbuseSimulation(user, cycle_count=3).generate()
        refunds = [t for t in txns if t.metadata.get("step") == "refund"]
        valid_reasons = {"item_not_received", "item_damaged", "wrong_item", "quality_issue"}
        for r in refunds:
            assert r.metadata.get("refund_reason") in valid_reasons

    def test_refund_after_purchase_per_cycle(self, user):
        txns = RefundAbuseSimulation(user, cycle_count=2).generate()
        for cycle in range(2):
            p = [t for t in txns if t.metadata.get("step") == "purchase" and t.metadata.get("cycle") == cycle][0]
            r = [t for t in txns if t.metadata.get("step") == "refund" and t.metadata.get("cycle") == cycle][0]
            assert r.timestamp > p.timestamp

    def test_scenario_label(self, user):
        txns = RefundAbuseSimulation(user).generate()
        for t in txns:
            assert t.metadata["fraud_scenario"] == "refund_abuse"


# ---------------------------------------------------------------------------
# MerchantCollusionSimulation
# ---------------------------------------------------------------------------


class TestMerchantCollusionSimulation:
    def test_generates_ring_transactions(self, user):
        scenario = MerchantCollusionSimulation(user, ring_size=4)
        txns = scenario.generate()

        _assert_all_transactions(txns)
        _assert_sorted_by_time(txns)
        assert len(txns) == 4

    def test_circular_sender_receiver(self, user):
        scenario = MerchantCollusionSimulation(user, ring_size=3)
        txns = scenario.generate()

        senders = [t.metadata["sender_id"] for t in txns]
        receivers = [t.metadata["receiver_id"] for t in txns]

        # Last receiver should equal first sender (circular)
        assert receivers[-1] == senders[0]

    def test_amounts_are_similar(self, user):
        scenario = MerchantCollusionSimulation(user, ring_size=4)
        txns = scenario.generate()

        amounts = [float(t.amount) for t in txns]
        avg = sum(amounts) / len(amounts)
        for a in amounts:
            # Each amount should be within ~10% of average
            assert abs(a - avg) / avg < 0.15

    def test_all_transfers(self, user):
        txns = MerchantCollusionSimulation(user, ring_size=3).generate()
        for t in txns:
            assert t.transaction_type == TransactionType.TRANSFER

    def test_uses_colluding_merchants(self, user):
        txns = MerchantCollusionSimulation(user, ring_size=4).generate()
        merchants = {t.metadata.get("colluding_merchant") for t in txns}
        # Should use a limited set of colluding merchants
        assert 1 <= len(merchants) <= 3

    def test_scenario_label(self, user):
        txns = MerchantCollusionSimulation(user).generate()
        for t in txns:
            assert t.metadata["fraud_scenario"] == "merchant_collusion"


# ---------------------------------------------------------------------------
# SCENARIO_CLASSES registry
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    def test_all_seven_scenarios_registered(self):
        assert len(SCENARIO_CLASSES) == 7

    def test_all_classes_are_subclasses(self):
        from simulator.scenarios import FraudScenario

        for cls in SCENARIO_CLASSES:
            assert issubclass(cls, FraudScenario)

    def test_all_scenarios_can_be_instantiated_and_generate(self, user):
        """Smoke test: every registered scenario produces valid output."""
        for cls in SCENARIO_CLASSES:
            # Some scenarios need a user, some don't
            if cls in (SyntheticIdentityFraud, MoneyMuleChain):
                scenario = cls()
            else:
                scenario = cls(user)

            txns = scenario.generate()
            _assert_all_transactions(txns)
            assert len(txns) >= 2
