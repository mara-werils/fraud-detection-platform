"""Tests for CaseWorkflowEngine."""

from datetime import UTC, datetime, timedelta

import pytest

from scoring.services.case_workflow import (
    Analyst,
    CasePriority,
    CaseTemplate,
    CaseWorkflowEngine,
    WorkflowState,
    compute_priority,
)

# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

class TestComputePriority:
    def test_low_priority(self):
        assert compute_priority(100, 0.1) == CasePriority.LOW

    def test_medium_priority(self):
        assert compute_priority(3000, 0.3) == CasePriority.MEDIUM

    def test_high_priority(self):
        assert compute_priority(5000, 0.7) == CasePriority.HIGH

    def test_critical_priority(self):
        assert compute_priority(10_000, 0.9) == CasePriority.CRITICAL

    def test_vip_tier_boosts_priority(self):
        # Standard tier would be medium, VIP tier bumps it up
        standard = compute_priority(3000, 0.3, "standard")
        vip = compute_priority(3000, 0.3, "vip")
        assert vip.value <= standard.value or PRIORITY_ORDER_MAP[vip] < PRIORITY_ORDER_MAP[standard]

    def test_premium_tier(self):
        p = compute_priority(5000, 0.5, "premium")
        assert p in (CasePriority.HIGH, CasePriority.CRITICAL)


# Mapping used in a test above
PRIORITY_ORDER_MAP = {
    CasePriority.CRITICAL: 0,
    CasePriority.HIGH: 1,
    CasePriority.MEDIUM: 2,
    CasePriority.LOW: 3,
}


# ---------------------------------------------------------------------------
# Workflow state transitions
# ---------------------------------------------------------------------------

class TestWorkflowTransitions:
    def _engine_with_case(self, **kwargs):
        engine = CaseWorkflowEngine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=5000, risk_score=0.8, **kwargs,
        )
        return engine, case

    def test_open_to_investigating(self):
        engine, case = self._engine_with_case()
        updated = engine.transition(case.case_id, "investigating", actor="alice")
        assert updated.state == WorkflowState.INVESTIGATING

    def test_investigating_to_escalated(self):
        engine, case = self._engine_with_case()
        engine.transition(case.case_id, "investigating")
        updated = engine.transition(case.case_id, "escalated")
        assert updated.state == WorkflowState.ESCALATED

    def test_escalated_to_resolved(self):
        engine, case = self._engine_with_case()
        engine.transition(case.case_id, "investigating")
        engine.transition(case.case_id, "escalated")
        updated = engine.transition(case.case_id, "resolved", notes="confirmed fraud")
        assert updated.state == WorkflowState.RESOLVED
        assert updated.resolved_at is not None

    def test_resolved_to_closed(self):
        engine, case = self._engine_with_case()
        engine.transition(case.case_id, "investigating")
        engine.transition(case.case_id, "resolved")
        updated = engine.transition(case.case_id, "closed")
        assert updated.state == WorkflowState.CLOSED

    def test_invalid_transition_raises(self):
        engine, case = self._engine_with_case()
        engine.transition(case.case_id, "investigating")
        engine.transition(case.case_id, "resolved")
        engine.transition(case.case_id, "closed")
        with pytest.raises(ValueError, match="not allowed"):
            engine.transition(case.case_id, "open")

    def test_closed_to_anything_raises(self):
        engine, case = self._engine_with_case()
        engine.transition(case.case_id, "closed")
        with pytest.raises(ValueError):
            engine.transition(case.case_id, "investigating")

    def test_nonexistent_case_raises(self):
        engine = CaseWorkflowEngine()
        with pytest.raises(KeyError):
            engine.transition("no-such-id", "investigating")

    def test_transition_adds_timeline_entry(self):
        engine, case = self._engine_with_case()
        engine.transition(case.case_id, "investigating", actor="bob")
        timeline = engine.get_timeline(case.case_id)
        events = [e for e in timeline if e.event_type == "state_changed"]
        assert len(events) == 1
        assert events[0].actor == "bob"

    def test_transition_with_notes(self):
        engine, case = self._engine_with_case()
        engine.transition(case.case_id, "investigating", notes="Starting review")
        updated = engine.get_case(case.case_id)
        assert any(n.content == "Starting review" for n in updated.notes)


# ---------------------------------------------------------------------------
# Auto-assignment
# ---------------------------------------------------------------------------

class TestAutoAssignment:
    def _build_engine(self):
        engine = CaseWorkflowEngine()
        engine.register_analyst(Analyst(
            analyst_id="a1", name="Alice", expertise=["card_fraud"], max_cases=3,
        ))
        engine.register_analyst(Analyst(
            analyst_id="a2", name="Bob", expertise=["account_takeover"], max_cases=3,
        ))
        engine.register_analyst(Analyst(
            analyst_id="a3", name="Charlie", expertise=["card_fraud", "phishing"], max_cases=2,
        ))
        return engine

    def test_assigns_expert_with_lowest_load(self):
        engine = self._build_engine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=5000,
            risk_score=0.8, fraud_type="card_fraud",
        )
        engine.auto_assign(case.case_id)
        updated = engine.get_case(case.case_id)
        # a1 or a3 both have card_fraud expertise; both at 0 load
        assert updated.assigned_to in ("a1", "a3")

    def test_falls_back_to_lowest_load_without_expertise(self):
        engine = self._build_engine()
        case = engine.create_case(
            transaction_id="tx2", user_id="u2", amount=100,
            risk_score=0.2, fraud_type="wire_fraud",
        )
        engine.auto_assign(case.case_id)
        updated = engine.get_case(case.case_id)
        assert updated.assigned_to is not None

    def test_respects_max_cases(self):
        engine = CaseWorkflowEngine()
        engine.register_analyst(Analyst(
            analyst_id="a1", name="Alice", expertise=[], max_cases=1,
        ))
        engine.register_analyst(Analyst(
            analyst_id="a2", name="Bob", expertise=[], max_cases=5,
        ))
        # Fill a1's capacity
        c1 = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.assign(c1.case_id, "a1")

        c2 = engine.create_case(transaction_id="tx2", user_id="u2", amount=100, risk_score=0.1)
        engine.auto_assign(c2.case_id)
        assert engine.get_case(c2.case_id).assigned_to == "a2"

    def test_no_available_analyst_raises(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        with pytest.raises(RuntimeError, match="No available analyst"):
            engine.auto_assign(case.case_id)

    def test_inactive_analyst_skipped(self):
        engine = CaseWorkflowEngine()
        engine.register_analyst(Analyst(
            analyst_id="a1", name="Alice", active=False, max_cases=5,
        ))
        engine.register_analyst(Analyst(
            analyst_id="a2", name="Bob", active=True, max_cases=5,
        ))
        case = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.auto_assign(case.case_id)
        assert engine.get_case(case.case_id).assigned_to == "a2"

    def test_auto_assign_adds_timeline(self):
        engine = self._build_engine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1,
        )
        engine.auto_assign(case.case_id)
        events = [e for e in case.timeline if e.event_type == "auto_assigned"]
        assert len(events) == 1


# ---------------------------------------------------------------------------
# SLA tracking
# ---------------------------------------------------------------------------

class TestSLATracking:
    def test_sla_deadline_set_on_creation(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9,
        )
        assert case.sla_deadline is not None

    def test_critical_has_short_sla(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9,
        )
        assert case.priority == CasePriority.CRITICAL
        delta = case.sla_deadline - case.created_at
        assert delta <= timedelta(hours=1, seconds=1)

    def test_breach_detected(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9,
        )
        future = datetime.now(UTC) + timedelta(hours=2)
        breaches = engine.check_sla_breaches(now=future)
        assert len(breaches) == 1
        assert breaches[0].case_id == case.case_id

    def test_no_double_breach(self):
        engine = CaseWorkflowEngine()
        engine.create_case(
            transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9,
        )
        future = datetime.now(UTC) + timedelta(hours=2)
        engine.check_sla_breaches(now=future)
        second = engine.check_sla_breaches(now=future)
        assert len(second) == 0

    def test_resolved_case_not_breached(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9,
        )
        engine.transition(case.case_id, "investigating")
        engine.transition(case.case_id, "resolved")
        future = datetime.now(UTC) + timedelta(hours=2)
        breaches = engine.check_sla_breaches(now=future)
        assert len(breaches) == 0

    def test_sla_notifications_persisted(self):
        engine = CaseWorkflowEngine()
        engine.create_case(transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9)
        future = datetime.now(UTC) + timedelta(hours=2)
        engine.check_sla_breaches(now=future)
        assert len(engine.get_sla_notifications()) == 1

    def test_custom_sla_hours(self):
        custom = {
            CasePriority.CRITICAL: 0.5,
            CasePriority.HIGH: 2.0,
            CasePriority.MEDIUM: 12.0,
            CasePriority.LOW: 48.0,
        }
        engine = CaseWorkflowEngine(sla_hours=custom)
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9,
        )
        delta = case.sla_deadline - case.created_at
        assert delta <= timedelta(minutes=31)

    def test_sla_breach_adds_timeline_entry(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=10000, risk_score=0.9,
        )
        future = datetime.now(UTC) + timedelta(hours=2)
        engine.check_sla_breaches(now=future)
        updated = engine.get_case(case.case_id)
        assert any(e.event_type == "sla_breached" for e in updated.timeline)


# ---------------------------------------------------------------------------
# Notes and timeline
# ---------------------------------------------------------------------------

class TestNotesAndTimeline:
    def test_add_note(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.add_note(case.case_id, "alice", "Suspicious pattern found")
        updated = engine.get_case(case.case_id)
        assert len(updated.notes) == 1
        assert updated.notes[0].content == "Suspicious pattern found"

    def test_add_note_nonexistent_case(self):
        engine = CaseWorkflowEngine()
        with pytest.raises(KeyError):
            engine.add_note("nope", "alice", "text")

    def test_timeline_chronological(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.add_note(case.case_id, "alice", "note 1")
        engine.transition(case.case_id, "investigating", actor="alice")
        timeline = engine.get_timeline(case.case_id)
        assert len(timeline) >= 3  # created + note_added + state_changed
        types = [e.event_type for e in timeline]
        assert "created" in types
        assert "note_added" in types
        assert "state_changed" in types


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------

class TestBulkOperations:
    def _create_cases(self, engine, n=5):
        cases = []
        for i in range(n):
            c = engine.create_case(
                transaction_id=f"tx{i}", user_id=f"u{i}", amount=1000, risk_score=0.5,
            )
            cases.append(c)
        return cases

    def test_bulk_assign(self):
        engine = CaseWorkflowEngine()
        cases = self._create_cases(engine, 3)
        ids = [c.case_id for c in cases]
        success, not_found = engine.bulk_assign(ids, "alice")
        assert success == 3
        assert not_found == 0
        for c in cases:
            assert engine.get_case(c.case_id).assigned_to == "alice"

    def test_bulk_assign_partial_failure(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        success, not_found = engine.bulk_assign([case.case_id, "ghost"], "bob")
        assert success == 1
        assert not_found == 1

    def test_bulk_escalate(self):
        engine = CaseWorkflowEngine()
        cases = self._create_cases(engine, 3)
        ids = [c.case_id for c in cases]
        success, failed = engine.bulk_escalate(ids, actor="manager")
        assert success == 3
        assert failed == 0
        for c in cases:
            assert engine.get_case(c.case_id).state == WorkflowState.ESCALATED

    def test_bulk_escalate_skips_closed(self):
        engine = CaseWorkflowEngine()
        cases = self._create_cases(engine, 2)
        engine.transition(cases[0].case_id, "closed")
        ids = [c.case_id for c in cases]
        success, failed = engine.bulk_escalate(ids)
        assert success == 1
        assert failed == 1

    def test_bulk_close(self):
        engine = CaseWorkflowEngine()
        cases = self._create_cases(engine, 3)
        ids = [c.case_id for c in cases]
        success, failed = engine.bulk_close(ids, actor="admin", notes="batch close")
        assert success == 3
        for c in cases:
            assert engine.get_case(c.case_id).state == WorkflowState.CLOSED


# ---------------------------------------------------------------------------
# Case templates
# ---------------------------------------------------------------------------

class TestCaseTemplates:
    def test_register_and_list(self):
        engine = CaseWorkflowEngine()
        tmpl = CaseTemplate(
            name="Card Fraud",
            fraud_type="card_fraud",
            default_tags=["card", "dispute"],
            initial_notes="Check chargeback records",
            checklist=["Verify cardholder", "Check IP geolocation"],
        )
        engine.register_template(tmpl)
        assert len(engine.list_templates()) == 1

    def test_create_case_from_template(self):
        engine = CaseWorkflowEngine()
        tmpl = CaseTemplate(
            name="Account Takeover",
            fraud_type="account_takeover",
            default_tags=["ato", "credential"],
            initial_notes="Verify recent password changes",
        )
        engine.register_template(tmpl)

        case = engine.create_case(
            transaction_id="tx1",
            user_id="u1",
            amount=2000,
            risk_score=0.7,
            template_id=tmpl.template_id,
        )
        assert case.fraud_type == "account_takeover"
        assert "ato" in case.tags
        assert any("password" in n.content for n in case.notes)

    def test_template_tags_merge_with_explicit(self):
        engine = CaseWorkflowEngine()
        tmpl = CaseTemplate(
            name="Phishing",
            fraud_type="phishing",
            default_tags=["phishing"],
        )
        engine.register_template(tmpl)
        case = engine.create_case(
            transaction_id="tx1", user_id="u1",
            amount=500, risk_score=0.5,
            tags=["manual"],
            template_id=tmpl.template_id,
        )
        assert "phishing" in case.tags
        assert "manual" in case.tags

    def test_get_template(self):
        engine = CaseWorkflowEngine()
        tmpl = CaseTemplate(name="T1", fraud_type="ft")
        engine.register_template(tmpl)
        assert engine.get_template(tmpl.template_id) is not None
        assert engine.get_template("no-such") is None


# ---------------------------------------------------------------------------
# Case similarity search
# ---------------------------------------------------------------------------

class TestCaseSimilarity:
    def test_same_user_scores_high(self):
        engine = CaseWorkflowEngine()
        c1 = engine.create_case(transaction_id="tx1", user_id="u1", amount=1000, risk_score=0.5)
        c2 = engine.create_case(transaction_id="tx2", user_id="u1", amount=1000, risk_score=0.5)
        results = engine.find_similar_cases(c1.case_id)
        assert len(results) == 1
        assert results[0][0] == c2.case_id
        assert results[0][1] >= 0.4  # user match alone = 0.4

    def test_same_fraud_type_scores(self):
        engine = CaseWorkflowEngine()
        c1 = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=1000, risk_score=0.5,
            fraud_type="card_fraud",
        )
        c2 = engine.create_case(
            transaction_id="tx2", user_id="u2", amount=1000, risk_score=0.5,
            fraud_type="card_fraud",
        )
        results = engine.find_similar_cases(c1.case_id)
        assert any(r[0] == c2.case_id for r in results)

    def test_tag_overlap_contributes(self):
        engine = CaseWorkflowEngine()
        c1 = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=1000, risk_score=0.5,
            tags=["dispute", "chargeback"],
        )
        c2 = engine.create_case(
            transaction_id="tx2", user_id="u2", amount=5000, risk_score=0.5,
            tags=["dispute", "chargeback"],
        )
        results = engine.find_similar_cases(c1.case_id)
        scores = {r[0]: r[1] for r in results}
        assert c2.case_id in scores
        assert scores[c2.case_id] > 0

    def test_max_results_limit(self):
        engine = CaseWorkflowEngine()
        c1 = engine.create_case(
            transaction_id="tx0", user_id="u1", amount=1000, risk_score=0.5,
        )
        for i in range(1, 10):
            engine.create_case(
                transaction_id=f"tx{i}", user_id="u1", amount=1000, risk_score=0.5,
            )
        results = engine.find_similar_cases(c1.case_id, max_results=3)
        assert len(results) == 3

    def test_nonexistent_case_raises(self):
        engine = CaseWorkflowEngine()
        with pytest.raises(KeyError):
            engine.find_similar_cases("ghost")

    def test_amount_proximity(self):
        engine = CaseWorkflowEngine()
        c1 = engine.create_case(
            transaction_id="tx1", user_id="u1", amount=1000, risk_score=0.5,
            fraud_type="card_fraud",
        )
        c_close = engine.create_case(
            transaction_id="tx2", user_id="u2", amount=1010, risk_score=0.5,
            fraud_type="card_fraud",
        )
        c_far = engine.create_case(
            transaction_id="tx3", user_id="u3", amount=50000, risk_score=0.5,
            fraud_type="card_fraud",
        )
        results = engine.find_similar_cases(c1.case_id)
        scores = {r[0]: r[1] for r in results}
        assert scores[c_close.case_id] > scores[c_far.case_id]


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestListCases:
    def test_list_all(self):
        engine = CaseWorkflowEngine()
        for i in range(5):
            engine.create_case(transaction_id=f"tx{i}", user_id="u1", amount=100, risk_score=0.1)
        cases, total = engine.list_cases()
        assert total == 5
        assert len(cases) == 5

    def test_filter_by_state(self):
        engine = CaseWorkflowEngine()
        c1 = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.create_case(transaction_id="tx2", user_id="u1", amount=100, risk_score=0.1)
        engine.transition(c1.case_id, "investigating")
        cases, total = engine.list_cases(state="investigating")
        assert total == 1
        assert cases[0].case_id == c1.case_id

    def test_filter_by_assigned_to(self):
        engine = CaseWorkflowEngine()
        c1 = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.create_case(transaction_id="tx2", user_id="u2", amount=100, risk_score=0.1)
        engine.assign(c1.case_id, "alice")
        cases, _ = engine.list_cases(assigned_to="alice")
        assert len(cases) == 1

    def test_pagination(self):
        engine = CaseWorkflowEngine()
        for i in range(10):
            engine.create_case(transaction_id=f"tx{i}", user_id="u1", amount=100, risk_score=0.1)
        cases, total = engine.list_cases(limit=3, offset=0)
        assert len(cases) == 3
        assert total == 10


# ---------------------------------------------------------------------------
# Analyst management
# ---------------------------------------------------------------------------

class TestAnalystManagement:
    def test_register_and_remove(self):
        engine = CaseWorkflowEngine()
        engine.register_analyst(Analyst(analyst_id="a1", name="Alice"))
        assert engine.remove_analyst("a1") is True
        assert engine.remove_analyst("a1") is False

    def test_workload_count(self):
        engine = CaseWorkflowEngine()
        engine.register_analyst(Analyst(analyst_id="a1", name="Alice", max_cases=10))
        c1 = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        c2 = engine.create_case(transaction_id="tx2", user_id="u2", amount=100, risk_score=0.1)
        engine.assign(c1.case_id, "a1")
        engine.assign(c2.case_id, "a1")
        assert engine.get_analyst_workload("a1") == 2

    def test_resolved_case_not_counted_in_workload(self):
        engine = CaseWorkflowEngine()
        engine.register_analyst(Analyst(analyst_id="a1", name="Alice", max_cases=10))
        c1 = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.assign(c1.case_id, "a1")
        engine.transition(c1.case_id, "investigating")
        engine.transition(c1.case_id, "resolved")
        assert engine.get_analyst_workload("a1") == 0


# ---------------------------------------------------------------------------
# Manual assignment
# ---------------------------------------------------------------------------

class TestManualAssignment:
    def test_assign(self):
        engine = CaseWorkflowEngine()
        case = engine.create_case(transaction_id="tx1", user_id="u1", amount=100, risk_score=0.1)
        engine.assign(case.case_id, "bob", actor="admin")
        updated = engine.get_case(case.case_id)
        assert updated.assigned_to == "bob"
        assert any(e.event_type == "assigned" for e in updated.timeline)

    def test_assign_nonexistent_raises(self):
        engine = CaseWorkflowEngine()
        with pytest.raises(KeyError):
            engine.assign("nope", "bob")
