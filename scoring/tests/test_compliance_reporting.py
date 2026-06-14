"""Tests for ComplianceReportingService."""

import json
from datetime import UTC, datetime, timedelta

from scoring.services.compliance_reporting import (
    CTR_THRESHOLD_USD,
    ComplianceReportingService,
    ExportFormat,
    GDPRRequestStatus,
    GDPRRequestType,
    Jurisdiction,
    JURISDICTION_TEMPLATES,
    PCIRequirementStatus,
    ReportStatus,
    ReportType,
    ScheduleFrequency,
)


def _make_service() -> ComplianceReportingService:
    return ComplianceReportingService()


def _sample_sar_kwargs(
    jurisdiction: Jurisdiction = Jurisdiction.US,
) -> dict:
    now = datetime.now(UTC)
    return {
        "subject_name": "John Doe",
        "subject_id": "subj-001",
        "subject_account": "acct-999",
        "suspicious_activity_description": "Structured deposits under $10k across 5 accounts",
        "transaction_ids": ["tx-1", "tx-2", "tx-3"],
        "total_amount": 48_500.00,
        "currency": "USD",
        "activity_start_date": now - timedelta(days=30),
        "activity_end_date": now,
        "narrative": "Subject made 5 deposits of $9,700 each over 3 days.",
        "filing_institution": "Acme Bank",
        "jurisdiction": jurisdiction,
        "created_by": "analyst-1",
    }


# ---------------------------------------------------------------------------
# SAR Tests
# ---------------------------------------------------------------------------


class TestSARGeneration:
    def test_generate_sar_creates_draft(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())

        assert sar.report_id
        assert sar.status == ReportStatus.DRAFT
        assert sar.subject_name == "John Doe"
        assert sar.total_amount == 48_500.00
        assert sar.jurisdiction == Jurisdiction.US
        assert sar.filing_deadline is not None

    def test_sar_stored_and_retrievable(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        fetched = svc.get_sar(sar.report_id)
        assert fetched is not None
        assert fetched.report_id == sar.report_id

    def test_get_sar_not_found(self):
        svc = _make_service()
        assert svc.get_sar("nonexistent") is None

    def test_review_sar_approve(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        reviewed = svc.review_sar(sar.report_id, reviewer="mgr-1", approved=True, notes="Looks good")
        assert reviewed.status == ReportStatus.APPROVED
        assert reviewed.reviewed_by == "mgr-1"
        assert reviewed.review_notes == "Looks good"

    def test_review_sar_reject(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        reviewed = svc.review_sar(sar.report_id, reviewer="mgr-1", approved=False, notes="Insufficient evidence")
        assert reviewed.status == ReportStatus.REJECTED

    def test_review_sar_not_found_raises(self):
        svc = _make_service()
        try:
            svc.review_sar("bad-id", reviewer="mgr-1", approved=True)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_review_filed_sar_raises(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        svc.review_sar(sar.report_id, reviewer="mgr-1", approved=True)
        svc.file_sar(sar.report_id)
        try:
            svc.review_sar(sar.report_id, reviewer="mgr-2", approved=True)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_file_sar(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        svc.review_sar(sar.report_id, reviewer="mgr-1", approved=True)
        filed = svc.file_sar(sar.report_id)
        assert filed.status == ReportStatus.FILED
        assert filed.filing_date is not None

    def test_file_sar_not_approved_raises(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        try:
            svc.file_sar(sar.report_id)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "APPROVED" in str(exc)

    def test_list_sars_filter_by_status(self):
        svc = _make_service()
        svc.generate_sar(**_sample_sar_kwargs())
        sar2 = svc.generate_sar(**_sample_sar_kwargs())
        svc.review_sar(sar2.report_id, reviewer="mgr-1", approved=True)

        drafts = svc.list_sars(status=ReportStatus.DRAFT)
        assert len(drafts) == 1
        approved = svc.list_sars(status=ReportStatus.APPROVED)
        assert len(approved) == 1

    def test_list_sars_filter_by_jurisdiction(self):
        svc = _make_service()
        svc.generate_sar(**_sample_sar_kwargs(Jurisdiction.US))
        svc.generate_sar(**_sample_sar_kwargs(Jurisdiction.EU))

        us = svc.list_sars(jurisdiction=Jurisdiction.US)
        eu = svc.list_sars(jurisdiction=Jurisdiction.EU)
        assert len(us) == 1
        assert len(eu) == 1

    def test_overdue_sars(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        # Force the deadline to the past
        sar.filing_deadline = datetime.now(UTC) - timedelta(days=1)
        overdue = svc.get_overdue_sars()
        assert len(overdue) == 1
        assert overdue[0].report_id == sar.report_id

    def test_filed_sar_not_overdue(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        svc.review_sar(sar.report_id, reviewer="mgr-1", approved=True)
        svc.file_sar(sar.report_id)
        sar.filing_deadline = datetime.now(UTC) - timedelta(days=1)
        assert len(svc.get_overdue_sars()) == 0

    def test_sar_jurisdiction_uk(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs(Jurisdiction.UK))
        assert sar.jurisdiction == Jurisdiction.UK
        # UK has 28-day deadline
        assert sar.filing_deadline is not None


# ---------------------------------------------------------------------------
# CTR Tests
# ---------------------------------------------------------------------------


class TestCTRGeneration:
    def test_generate_ctr_above_threshold(self):
        svc = _make_service()
        ctr = svc.generate_ctr(
            transaction_id="tx-100",
            customer_name="Jane Smith",
            customer_id="cust-001",
            transaction_amount=15_000.00,
            transaction_date=datetime.now(UTC),
        )
        assert ctr is not None
        assert ctr.transaction_amount == 15_000.00
        assert ctr.threshold_amount == CTR_THRESHOLD_USD

    def test_generate_ctr_below_threshold_returns_none(self):
        svc = _make_service()
        ctr = svc.generate_ctr(
            transaction_id="tx-101",
            customer_name="Jane Smith",
            customer_id="cust-001",
            transaction_amount=5_000.00,
            transaction_date=datetime.now(UTC),
        )
        assert ctr is None

    def test_generate_ctr_at_exact_threshold(self):
        svc = _make_service()
        ctr = svc.generate_ctr(
            transaction_id="tx-102",
            customer_name="Jane Smith",
            customer_id="cust-001",
            transaction_amount=CTR_THRESHOLD_USD,
            transaction_date=datetime.now(UTC),
        )
        assert ctr is not None

    def test_ctr_stored_and_retrievable(self):
        svc = _make_service()
        ctr = svc.generate_ctr(
            transaction_id="tx-100",
            customer_name="Jane Smith",
            customer_id="cust-001",
            transaction_amount=15_000.00,
            transaction_date=datetime.now(UTC),
        )
        assert ctr is not None
        fetched = svc.get_ctr(ctr.report_id)
        assert fetched is not None
        assert fetched.report_id == ctr.report_id

    def test_get_ctr_not_found(self):
        svc = _make_service()
        assert svc.get_ctr("nonexistent") is None

    def test_list_ctrs_by_jurisdiction(self):
        svc = _make_service()
        svc.generate_ctr(
            transaction_id="tx-200",
            customer_name="A",
            customer_id="c1",
            transaction_amount=20_000.00,
            transaction_date=datetime.now(UTC),
            jurisdiction=Jurisdiction.US,
        )
        svc.generate_ctr(
            transaction_id="tx-201",
            customer_name="B",
            customer_id="c2",
            transaction_amount=20_000.00,
            transaction_date=datetime.now(UTC),
            jurisdiction=Jurisdiction.EU,
        )
        assert len(svc.list_ctrs(jurisdiction=Jurisdiction.US)) == 1
        assert len(svc.list_ctrs(jurisdiction=Jurisdiction.EU)) == 1
        assert len(svc.list_ctrs()) == 2


# ---------------------------------------------------------------------------
# PCI DSS Tests
# ---------------------------------------------------------------------------


class TestPCIDSSAssessment:
    def test_assessment_returns_report(self):
        svc = _make_service()
        report = svc.run_pci_dss_assessment(assessor="auditor-1")
        assert report.report_id
        assert report.assessor == "auditor-1"
        assert report.pass_count > 0
        assert report.fail_count > 0
        assert len(report.requirements) == 12

    def test_assessment_overall_not_compliant_when_failures(self):
        svc = _make_service()
        report = svc.run_pci_dss_assessment()
        assert report.overall_compliant is False
        assert report.fail_count >= 1

    def test_requirement_statuses(self):
        svc = _make_service()
        report = svc.run_pci_dss_assessment()
        statuses = {r.requirement_id: r.status for r in report.requirements}
        assert statuses["3.1"] == PCIRequirementStatus.PASS
        assert statuses["6.1"] == PCIRequirementStatus.FAIL
        assert statuses["9.1"] == PCIRequirementStatus.NOT_APPLICABLE

    def test_counts_add_up(self):
        svc = _make_service()
        report = svc.run_pci_dss_assessment()
        total = report.pass_count + report.fail_count + report.not_applicable_count
        assert total == len(report.requirements)


# ---------------------------------------------------------------------------
# GDPR Tests
# ---------------------------------------------------------------------------


class TestGDPRRequests:
    def test_create_access_request(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS,
            subject_id="user-123",
            subject_email="user@example.com",
        )
        assert req.request_id
        assert req.status == GDPRRequestStatus.RECEIVED
        assert req.request_type == GDPRRequestType.ACCESS
        assert req.deadline > datetime.now(UTC) - timedelta(seconds=5)

    def test_create_erasure_request(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.ERASURE,
            subject_id="user-456",
        )
        assert req.request_type == GDPRRequestType.ERASURE

    def test_process_gdpr_request_complete(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS,
            subject_id="user-123",
        )
        processed = svc.process_gdpr_request(
            req.request_id,
            processed_by="dpo-1",
            response_data={"transactions": [], "profile": {"name": "Test"}},
        )
        assert processed.status == GDPRRequestStatus.COMPLETED
        assert processed.completed_at is not None
        assert processed.processed_by == "dpo-1"
        assert "profile" in processed.response_data

    def test_process_gdpr_request_deny(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.ERASURE,
            subject_id="user-789",
        )
        denied = svc.process_gdpr_request(
            req.request_id,
            processed_by="dpo-1",
            denied=True,
            denial_reason="Legal hold in effect",
        )
        assert denied.status == GDPRRequestStatus.DENIED
        assert denied.denial_reason == "Legal hold in effect"

    def test_process_already_completed_raises(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS,
            subject_id="user-123",
        )
        svc.process_gdpr_request(req.request_id, processed_by="dpo-1")
        try:
            svc.process_gdpr_request(req.request_id, processed_by="dpo-2")
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_process_not_found_raises(self):
        svc = _make_service()
        try:
            svc.process_gdpr_request("bad-id", processed_by="dpo-1")
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_get_gdpr_request(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.PORTABILITY,
            subject_id="user-555",
        )
        fetched = svc.get_gdpr_request(req.request_id)
        assert fetched is not None
        assert fetched.request_type == GDPRRequestType.PORTABILITY

    def test_get_gdpr_request_not_found(self):
        svc = _make_service()
        assert svc.get_gdpr_request("nonexistent") is None

    def test_list_gdpr_requests_filter_status(self):
        svc = _make_service()
        svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS, subject_id="u1"
        )
        req2 = svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS, subject_id="u2"
        )
        svc.process_gdpr_request(req2.request_id, processed_by="dpo-1")

        received = svc.list_gdpr_requests(status=GDPRRequestStatus.RECEIVED)
        completed = svc.list_gdpr_requests(status=GDPRRequestStatus.COMPLETED)
        assert len(received) == 1
        assert len(completed) == 1

    def test_overdue_gdpr_requests(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS, subject_id="u1"
        )
        req.deadline = datetime.now(UTC) - timedelta(days=1)
        overdue = svc.get_overdue_gdpr_requests()
        assert len(overdue) == 1

    def test_completed_request_not_overdue(self):
        svc = _make_service()
        req = svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS, subject_id="u1"
        )
        svc.process_gdpr_request(req.request_id, processed_by="dpo-1")
        req.deadline = datetime.now(UTC) - timedelta(days=1)
        assert len(svc.get_overdue_gdpr_requests()) == 0


# ---------------------------------------------------------------------------
# Audit Trail Tests
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_audit_trail_from_sar_operations(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        svc.review_sar(sar.report_id, reviewer="mgr-1", approved=True)
        svc.file_sar(sar.report_id)

        now = datetime.now(UTC)
        trail = svc.generate_audit_trail(
            period_start=now - timedelta(minutes=5),
            period_end=now + timedelta(minutes=5),
            jurisdiction=Jurisdiction.US,
        )
        assert trail.total_entries >= 3
        actions = [e.action for e in trail.entries]
        assert "sar.generated" in actions
        assert "sar.reviewed" in actions
        assert "sar.filed" in actions

    def test_audit_trail_filter_by_action(self):
        svc = _make_service()
        svc.generate_sar(**_sample_sar_kwargs())
        svc.generate_ctr(
            transaction_id="tx-100",
            customer_name="X",
            customer_id="c1",
            transaction_amount=20_000.00,
            transaction_date=datetime.now(UTC),
        )

        now = datetime.now(UTC)
        trail = svc.generate_audit_trail(
            period_start=now - timedelta(minutes=5),
            period_end=now + timedelta(minutes=5),
            action_filter="ctr",
        )
        assert trail.total_entries >= 1
        for e in trail.entries:
            assert "ctr" in e.action

    def test_audit_trail_filter_by_jurisdiction(self):
        svc = _make_service()
        svc.generate_sar(**_sample_sar_kwargs(Jurisdiction.US))
        svc.create_gdpr_request(
            request_type=GDPRRequestType.ACCESS, subject_id="u1"
        )

        now = datetime.now(UTC)
        us_trail = svc.generate_audit_trail(
            period_start=now - timedelta(minutes=5),
            period_end=now + timedelta(minutes=5),
            jurisdiction=Jurisdiction.US,
        )
        eu_trail = svc.generate_audit_trail(
            period_start=now - timedelta(minutes=5),
            period_end=now + timedelta(minutes=5),
            jurisdiction=Jurisdiction.EU,
        )
        assert us_trail.total_entries >= 1
        assert eu_trail.total_entries >= 1

    def test_audit_trail_summary_structure(self):
        svc = _make_service()
        svc.generate_sar(**_sample_sar_kwargs())

        now = datetime.now(UTC)
        trail = svc.generate_audit_trail(
            period_start=now - timedelta(minutes=5),
            period_end=now + timedelta(minutes=5),
        )
        assert "by_action" in trail.summary
        assert "by_actor" in trail.summary


# ---------------------------------------------------------------------------
# Scheduling Tests
# ---------------------------------------------------------------------------


class TestReportScheduling:
    def test_schedule_daily_report(self):
        svc = _make_service()
        sched = svc.schedule_report(
            report_type=ReportType.SAR,
            frequency=ScheduleFrequency.DAILY,
            recipients=["compliance@example.com"],
        )
        assert sched.schedule_id
        assert sched.frequency == ScheduleFrequency.DAILY
        assert sched.enabled is True
        assert sched.next_run is not None

    def test_schedule_weekly_report(self):
        svc = _make_service()
        sched = svc.schedule_report(
            report_type=ReportType.AUDIT_TRAIL,
            frequency=ScheduleFrequency.WEEKLY,
        )
        assert sched.frequency == ScheduleFrequency.WEEKLY
        # Next run should be ~7 days from now
        delta = sched.next_run - datetime.now(UTC)
        assert 6 <= delta.days <= 7

    def test_schedule_monthly_report(self):
        svc = _make_service()
        sched = svc.schedule_report(
            report_type=ReportType.PCI_DSS,
            frequency=ScheduleFrequency.MONTHLY,
        )
        assert sched.frequency == ScheduleFrequency.MONTHLY

    def test_get_schedule(self):
        svc = _make_service()
        sched = svc.schedule_report(
            report_type=ReportType.CTR,
            frequency=ScheduleFrequency.DAILY,
        )
        fetched = svc.get_schedule(sched.schedule_id)
        assert fetched is not None

    def test_list_schedules(self):
        svc = _make_service()
        svc.schedule_report(report_type=ReportType.SAR, frequency=ScheduleFrequency.DAILY)
        svc.schedule_report(report_type=ReportType.CTR, frequency=ScheduleFrequency.WEEKLY)
        assert len(svc.list_schedules()) == 2

    def test_disable_schedule(self):
        svc = _make_service()
        sched = svc.schedule_report(
            report_type=ReportType.SAR,
            frequency=ScheduleFrequency.DAILY,
        )
        disabled = svc.disable_schedule(sched.schedule_id)
        assert disabled.enabled is False

    def test_disable_nonexistent_raises(self):
        svc = _make_service()
        try:
            svc.disable_schedule("bad-id")
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_get_due_schedules(self):
        svc = _make_service()
        sched = svc.schedule_report(
            report_type=ReportType.SAR,
            frequency=ScheduleFrequency.DAILY,
        )
        # Force next_run to past
        sched.next_run = datetime.now(UTC) - timedelta(hours=1)
        due = svc.get_due_schedules()
        assert len(due) == 1

    def test_disabled_schedule_not_due(self):
        svc = _make_service()
        sched = svc.schedule_report(
            report_type=ReportType.SAR,
            frequency=ScheduleFrequency.DAILY,
        )
        sched.next_run = datetime.now(UTC) - timedelta(hours=1)
        svc.disable_schedule(sched.schedule_id)
        assert len(svc.get_due_schedules()) == 0


# ---------------------------------------------------------------------------
# Jurisdiction Template Tests
# ---------------------------------------------------------------------------


class TestJurisdictionTemplates:
    def test_us_template(self):
        tmpl = ComplianceReportingService.get_jurisdiction_template(Jurisdiction.US)
        assert tmpl["regulator"] == "FinCEN"
        assert tmpl["ctr_threshold"] == 10_000.00
        assert tmpl["currency"] == "USD"

    def test_eu_template(self):
        tmpl = ComplianceReportingService.get_jurisdiction_template(Jurisdiction.EU)
        assert tmpl["regulator"] == "National FIU"
        assert tmpl["currency"] == "EUR"
        assert tmpl["gdpr_applies"] is True

    def test_uk_template(self):
        tmpl = ComplianceReportingService.get_jurisdiction_template(Jurisdiction.UK)
        assert tmpl["regulator"] == "NCA (National Crime Agency)"
        assert tmpl["currency"] == "GBP"
        assert tmpl["filing_deadline_days"] == 28

    def test_all_jurisdictions_have_templates(self):
        for j in Jurisdiction:
            tmpl = ComplianceReportingService.get_jurisdiction_template(j)
            assert "regulator" in tmpl
            assert "ctr_threshold" in tmpl


# ---------------------------------------------------------------------------
# Export Format Tests
# ---------------------------------------------------------------------------


class TestExportFormats:
    def test_export_json(self):
        svc = _make_service()
        data = {"report_type": "SAR", "amount": 50_000}
        result = svc.export_report(data, ExportFormat.JSON)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["report_type"] == "SAR"

    def test_export_csv_single_dict(self):
        svc = _make_service()
        data = {"name": "John", "amount": 10_000}
        result = svc.export_report(data, ExportFormat.CSV)
        assert isinstance(result, str)
        assert "name" in result
        assert "John" in result

    def test_export_csv_list(self):
        svc = _make_service()
        data = [
            {"id": "1", "amount": 100},
            {"id": "2", "amount": 200},
        ]
        result = svc.export_report(data, ExportFormat.CSV)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_export_csv_empty_list(self):
        svc = _make_service()
        result = svc.export_report([], ExportFormat.CSV)
        assert result == ""

    def test_export_pdf_structured(self):
        svc = _make_service()
        data = {"title": "Test Report"}
        result = svc.export_report(data, ExportFormat.PDF_STRUCTURED)
        assert isinstance(result, dict)
        assert result["format"] == "pdf_structured"
        assert result["content"] == data
        assert result["metadata"]["page_size"] == "A4"

    def test_export_sar_json(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        result = svc.export_sar(sar.report_id, ExportFormat.JSON)
        parsed = json.loads(result)
        assert parsed["subject_name"] == "John Doe"

    def test_export_sar_csv(self):
        svc = _make_service()
        sar = svc.generate_sar(**_sample_sar_kwargs())
        result = svc.export_sar(sar.report_id, ExportFormat.CSV)
        assert "John Doe" in result

    def test_export_sar_not_found_raises(self):
        svc = _make_service()
        try:
            svc.export_sar("bad-id", ExportFormat.JSON)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_export_ctr_json(self):
        svc = _make_service()
        ctr = svc.generate_ctr(
            transaction_id="tx-100",
            customer_name="Jane",
            customer_id="c1",
            transaction_amount=20_000.00,
            transaction_date=datetime.now(UTC),
        )
        assert ctr is not None
        result = svc.export_ctr(ctr.report_id, ExportFormat.JSON)
        parsed = json.loads(result)
        assert parsed["customer_name"] == "Jane"

    def test_export_ctr_not_found_raises(self):
        svc = _make_service()
        try:
            svc.export_ctr("bad-id", ExportFormat.JSON)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
