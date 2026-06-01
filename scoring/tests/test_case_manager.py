"""Tests for CaseManager enhancements."""

from scoring.services.case_manager import CaseManager


class TestCaseManager:
    def test_create_and_get(self):
        cm = CaseManager()
        case = cm.create_case(transaction_id="tx1", user_id="u1", fraud_score=0.9)
        assert case.transaction_id == "tx1"
        assert cm.get_case(case.case_id) is not None

    def test_duplicate_transaction_returns_existing(self):
        cm = CaseManager()
        c1 = cm.create_case(transaction_id="tx1", user_id="u1")
        c2 = cm.create_case(transaction_id="tx1", user_id="u1")
        assert c1.case_id == c2.case_id

    def test_bulk_update_status(self):
        cm = CaseManager()
        cases = [
            cm.create_case(transaction_id=f"tx{i}", user_id="u1")
            for i in range(5)
        ]
        ids = [c.case_id for c in cases]

        success, not_found = cm.bulk_update_status(ids, "investigating", actor="analyst")
        assert success == 5
        assert not_found == 0

        for c in cases:
            updated = cm.get_case(c.case_id)
            assert updated is not None
            assert updated.status.value == "investigating"

    def test_bulk_update_with_missing_ids(self):
        cm = CaseManager()
        case = cm.create_case(transaction_id="tx1", user_id="u1")
        success, not_found = cm.bulk_update_status(
            [case.case_id, "nonexistent-id"], "closed"
        )
        assert success == 1
        assert not_found == 1

    def test_stats_empty(self):
        cm = CaseManager()
        stats = cm.stats()
        assert stats["total"] == 0

    def test_assign(self):
        cm = CaseManager()
        case = cm.create_case(transaction_id="tx1", user_id="u1")
        updated = cm.assign(case.case_id, "alice")
        assert updated is not None
        assert updated.assigned_to == "alice"
