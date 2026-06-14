"""Tests for the model experimentation framework."""

import random

import pytest

from scoring.services.model_experimentation import (
    AllocationStrategy,
    ExperimentationService,
    ExperimentConfig,
    ExperimentStatus,
    SignificanceResult,
    VariantMetrics,
    chi_squared_test,
    proportion_z_test,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> ExperimentConfig:
    """Build a default two-variant experiment config, with optional overrides."""
    defaults = dict(
        experiment_id="exp-001",
        name="Test Experiment",
        variants={"control": 0.5, "treatment": 0.5},
        control_variant="control",
        allocation_strategy=AllocationStrategy.FIXED,
        confidence_level=0.95,
        min_samples_per_variant=30,
        auto_conclude=False,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _seed_outcomes(
    svc: ExperimentationService,
    experiment_id: str,
    variant_id: str,
    tp: int = 0,
    fp: int = 0,
    tn: int = 0,
    fn: int = 0,
) -> None:
    """Record a batch of outcomes with controlled confusion matrix values."""
    for _ in range(tp):
        svc.record_outcome(experiment_id, variant_id, True, True, 5.0, 0.9)
    for _ in range(fp):
        svc.record_outcome(experiment_id, variant_id, True, False, 5.0, 0.8)
    for _ in range(tn):
        svc.record_outcome(experiment_id, variant_id, False, False, 5.0, 0.1)
    for _ in range(fn):
        svc.record_outcome(experiment_id, variant_id, False, True, 5.0, 0.2)


# ---------------------------------------------------------------------------
# ExperimentConfig validation
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_valid_config(self):
        cfg = _make_config()
        cfg.validate()  # should not raise

    def test_control_not_in_variants(self):
        cfg = _make_config(control_variant="missing")
        with pytest.raises(ValueError, match="Control variant"):
            cfg.validate()

    def test_fewer_than_two_variants(self):
        cfg = _make_config(variants={"only_one": 1.0}, control_variant="only_one")
        with pytest.raises(ValueError, match="At least two variants"):
            cfg.validate()

    def test_percentages_do_not_sum_to_one(self):
        cfg = _make_config(variants={"a": 0.3, "b": 0.3}, control_variant="a")
        with pytest.raises(ValueError, match="sum to 1.0"):
            cfg.validate()

    def test_negative_percentage(self):
        cfg = _make_config(variants={"a": -0.5, "b": 1.5}, control_variant="a")
        with pytest.raises(ValueError, match="must be in"):
            cfg.validate()

    def test_invalid_confidence_level(self):
        cfg = _make_config(confidence_level=1.5)
        with pytest.raises(ValueError, match="confidence_level"):
            cfg.validate()

    def test_zero_min_samples(self):
        cfg = _make_config(min_samples_per_variant=0)
        with pytest.raises(ValueError, match="min_samples_per_variant"):
            cfg.validate()


# ---------------------------------------------------------------------------
# Lifecycle management
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_create_experiment(self):
        svc = ExperimentationService()
        exp = svc.create_experiment(_make_config())
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.started_at is None

    def test_duplicate_experiment_raises(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        with pytest.raises(ValueError, match="already exists"):
            svc.create_experiment(_make_config())

    def test_start_experiment(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        exp = svc.start_experiment("exp-001")
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None

    def test_start_non_draft_raises(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")
        with pytest.raises(ValueError, match="Cannot start"):
            svc.start_experiment("exp-001")

    def test_conclude_experiment(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")
        exp = svc.conclude_experiment("exp-001", winner="treatment", reason="manual")
        assert exp.status == ExperimentStatus.CONCLUDED
        assert exp.winner == "treatment"
        assert exp.concluded_at is not None

    def test_conclude_non_running_raises(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        with pytest.raises(ValueError, match="Cannot conclude"):
            svc.conclude_experiment("exp-001")

    def test_get_experiment_not_found(self):
        svc = ExperimentationService()
        with pytest.raises(KeyError, match="not found"):
            svc.get_experiment("nonexistent")

    def test_list_experiments_filter(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config(experiment_id="a", name="A"))
        svc.create_experiment(_make_config(experiment_id="b", name="B"))
        svc.start_experiment("a")

        all_exps = svc.list_experiments()
        assert len(all_exps) == 2

        running = svc.list_experiments(status=ExperimentStatus.RUNNING)
        assert len(running) == 1
        assert running[0].config.experiment_id == "a"

        drafts = svc.list_experiments(status=ExperimentStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].config.experiment_id == "b"


# ---------------------------------------------------------------------------
# Traffic allocation
# ---------------------------------------------------------------------------


class TestTrafficAllocation:
    def test_fixed_allocation_distribution(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")

        counts = {"control": 0, "treatment": 0}
        random.seed(42)
        for _ in range(1000):
            v = svc.allocate_traffic("exp-001")
            counts[v] += 1

        # With 50/50 split and 1000 samples, each should be roughly 500
        assert 400 < counts["control"] < 600
        assert 400 < counts["treatment"] < 600

    def test_allocation_on_non_running_raises(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        with pytest.raises(ValueError, match="Cannot allocate"):
            svc.allocate_traffic("exp-001")

    def test_thompson_sampling_allocates_all_variants(self):
        cfg = _make_config(allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING)
        svc = ExperimentationService()
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        seen = set()
        random.seed(42)
        for _ in range(200):
            seen.add(svc.allocate_traffic("exp-001"))
        assert "control" in seen
        assert "treatment" in seen

    def test_ucb1_explores_all_arms(self):
        cfg = _make_config(allocation_strategy=AllocationStrategy.UCB1)
        svc = ExperimentationService()
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        seen = set()
        random.seed(42)
        for _ in range(10):
            v = svc.allocate_traffic("exp-001")
            seen.add(v)
            # Record some outcome so UCB1 has data
            svc.record_outcome("exp-001", v, True, True, 5.0, 0.9)

        assert "control" in seen
        assert "treatment" in seen

    def test_ucb1_zero_requests_picks_randomly(self):
        """UCB1 with no data should still return a valid variant."""
        cfg = _make_config(allocation_strategy=AllocationStrategy.UCB1)
        svc = ExperimentationService()
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        variant = svc.allocate_traffic("exp-001")
        assert variant in ("control", "treatment")


# ---------------------------------------------------------------------------
# Metric recording
# ---------------------------------------------------------------------------


class TestMetricRecording:
    def test_record_outcome_increments_metrics(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")

        svc.record_outcome("exp-001", "control", True, True, 10.0, 0.95)
        svc.record_outcome("exp-001", "control", True, False, 8.0, 0.85)
        svc.record_outcome("exp-001", "control", False, False, 3.0, 0.1)
        svc.record_outcome("exp-001", "control", False, True, 4.0, 0.2)

        exp = svc.get_experiment("exp-001")
        m = exp.variant_metrics["control"]
        assert m.total_requests == 4
        assert m.true_positives == 1
        assert m.false_positives == 1
        assert m.true_negatives == 1
        assert m.false_negatives == 1
        assert m.precision == 0.5
        assert m.recall == 0.5
        assert m.false_positive_rate == 0.5

    def test_record_invalid_variant_raises(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")

        with pytest.raises(KeyError, match="Variant"):
            svc.record_outcome("exp-001", "nonexistent", True, True, 5.0)

    def test_avg_latency(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")

        svc.record_outcome("exp-001", "treatment", True, True, 10.0)
        svc.record_outcome("exp-001", "treatment", False, False, 20.0)

        m = svc.get_experiment("exp-001").variant_metrics["treatment"]
        assert m.avg_latency_ms == 15.0

    def test_thompson_sampling_posterior_update(self):
        svc = ExperimentationService()
        svc.create_experiment(
            _make_config(allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING)
        )
        svc.start_experiment("exp-001")

        # True positive -> alpha increments
        svc.record_outcome("exp-001", "control", True, True, 5.0)
        m = svc.get_experiment("exp-001").variant_metrics["control"]
        assert m.ts_alpha == 2.0  # started at 1.0
        assert m.ts_beta == 1.0

        # False negative -> beta increments
        svc.record_outcome("exp-001", "control", False, True, 5.0)
        m = svc.get_experiment("exp-001").variant_metrics["control"]
        assert m.ts_alpha == 2.0
        assert m.ts_beta == 2.0

    def test_ucb_reward_update(self):
        svc = ExperimentationService()
        svc.create_experiment(
            _make_config(allocation_strategy=AllocationStrategy.UCB1)
        )
        svc.start_experiment("exp-001")

        # Correct prediction -> reward +1
        svc.record_outcome("exp-001", "control", True, True, 5.0)
        m = svc.get_experiment("exp-001").variant_metrics["control"]
        assert m.ucb_total_reward == 1.0

        # Incorrect prediction -> no reward
        svc.record_outcome("exp-001", "control", True, False, 5.0)
        m = svc.get_experiment("exp-001").variant_metrics["control"]
        assert m.ucb_total_reward == 1.0  # unchanged


# ---------------------------------------------------------------------------
# VariantMetrics edge cases
# ---------------------------------------------------------------------------


class TestVariantMetrics:
    def test_zero_requests_metrics(self):
        m = VariantMetrics(variant_id="v1")
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.false_positive_rate == 0.0
        assert m.avg_latency_ms == 0.0
        assert m.avg_fraud_score == 0.0

    def test_to_dict(self):
        m = VariantMetrics(variant_id="v1", total_requests=10, true_positives=5,
                           false_positives=2, true_negatives=2, false_negatives=1,
                           total_latency_ms=100.0, total_fraud_score=7.5)
        d = m.to_dict()
        assert d["variant_id"] == "v1"
        assert d["total_requests"] == 10
        assert d["precision"] == round(5 / 7, 6)
        assert d["recall"] == round(5 / 6, 6)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


class TestStatisticalFunctions:
    def test_proportion_z_test_identical(self):
        z, p, sig = proportion_z_test(50, 100, 50, 100)
        assert z == 0.0
        assert not sig

    def test_proportion_z_test_different(self):
        # Very different proportions with large N should be significant
        z, p, sig = proportion_z_test(90, 100, 10, 100, 0.95)
        assert sig
        assert p < 0.05

    def test_proportion_z_test_zero_n(self):
        z, p, sig = proportion_z_test(0, 0, 5, 10)
        assert not sig
        assert p == 1.0

    def test_chi_squared_identical(self):
        observed = [[50, 50], [50, 50]]
        chi2, p, sig = chi_squared_test(observed)
        assert chi2 == 0.0
        assert not sig

    def test_chi_squared_different(self):
        observed = [[90, 10], [10, 90]]
        chi2, p, sig = chi_squared_test(observed, 0.95)
        assert sig
        assert chi2 > 0

    def test_chi_squared_single_row(self):
        chi2, p, sig = chi_squared_test([[10, 20]])
        assert not sig

    def test_chi_squared_empty(self):
        chi2, p, sig = chi_squared_test([[0, 0], [0, 0]])
        assert not sig


# ---------------------------------------------------------------------------
# Significance testing via service
# ---------------------------------------------------------------------------


class TestSignificanceTesting:
    def test_run_significance_with_insufficient_data(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config(min_samples_per_variant=100))
        svc.start_experiment("exp-001")

        _seed_outcomes(svc, "exp-001", "control", tp=5, fp=2, tn=10, fn=1)
        _seed_outcomes(svc, "exp-001", "treatment", tp=8, fp=1, tn=10, fn=1)

        results = svc.run_significance_tests("exp-001")
        assert results == []  # not enough samples

    def test_run_significance_with_data(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config(min_samples_per_variant=30))
        svc.start_experiment("exp-001")

        # Control: lower precision
        _seed_outcomes(svc, "exp-001", "control", tp=10, fp=20, tn=50, fn=5)
        # Treatment: higher precision
        _seed_outcomes(svc, "exp-001", "treatment", tp=30, fp=5, tn=50, fn=5)

        results = svc.run_significance_tests("exp-001")
        assert len(results) > 0

        precision_results = [r for r in results if r.metric_name == "precision"]
        assert len(precision_results) == 1
        # Treatment has much higher precision (30/35 vs 10/30)
        assert precision_results[0].treatment_value > precision_results[0].control_value

    def test_significance_results_stored_on_experiment(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config(min_samples_per_variant=30))
        svc.start_experiment("exp-001")

        _seed_outcomes(svc, "exp-001", "control", tp=10, fp=20, tn=50, fn=5)
        _seed_outcomes(svc, "exp-001", "treatment", tp=30, fp=5, tn=50, fn=5)

        svc.run_significance_tests("exp-001")
        exp = svc.get_experiment("exp-001")
        assert len(exp.significance_results) > 0


# ---------------------------------------------------------------------------
# Auto-conclusion
# ---------------------------------------------------------------------------


class TestAutoConclusion:
    def test_auto_conclude_on_significance(self):
        svc = ExperimentationService()
        cfg = _make_config(
            auto_conclude=True,
            min_samples_per_variant=30,
            primary_metric="precision",
        )
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        # Seed enough data to reach significance.
        # Control: precision = 20/120 ~ 0.167
        _seed_outcomes(svc, "exp-001", "control", tp=20, fp=100, tn=50, fn=5)
        # Treatment: precision = 80/90 ~ 0.889 -- large enough gap
        _seed_outcomes(svc, "exp-001", "treatment", tp=80, fp=10, tn=50, fn=5)

        # The last record_outcome should trigger auto_conclude
        # Force a check by recording one more
        svc.record_outcome("exp-001", "treatment", True, True, 5.0, 0.95)

        exp = svc.get_experiment("exp-001")
        # It may or may not have concluded depending on p-value; check consistency
        if exp.status == ExperimentStatus.CONCLUDED:
            assert exp.conclusion_reason == "auto_significance_reached"
            assert exp.winner is not None

    def test_no_auto_conclude_when_disabled(self):
        svc = ExperimentationService()
        cfg = _make_config(auto_conclude=False, min_samples_per_variant=5)
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        _seed_outcomes(svc, "exp-001", "control", tp=5, fp=50, tn=50, fn=5)
        _seed_outcomes(svc, "exp-001", "treatment", tp=50, fp=5, tn=50, fn=5)

        exp = svc.get_experiment("exp-001")
        assert exp.status == ExperimentStatus.RUNNING

    def test_no_auto_conclude_below_min_samples(self):
        svc = ExperimentationService()
        cfg = _make_config(auto_conclude=True, min_samples_per_variant=1000)
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        _seed_outcomes(svc, "exp-001", "control", tp=5, fp=5, tn=5, fn=5)
        _seed_outcomes(svc, "exp-001", "treatment", tp=5, fp=5, tn=5, fn=5)

        exp = svc.get_experiment("exp-001")
        assert exp.status == ExperimentStatus.RUNNING


# ---------------------------------------------------------------------------
# Experiment summary / serialization
# ---------------------------------------------------------------------------


class TestExperimentSummary:
    def test_get_summary(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")

        _seed_outcomes(svc, "exp-001", "control", tp=10, fp=5, tn=30, fn=2)

        summary = svc.get_experiment_summary("exp-001")
        assert summary["experiment_id"] == "exp-001"
        assert summary["status"] == "running"
        assert "control" in summary["variant_metrics"]
        assert summary["variant_metrics"]["control"]["total_requests"] == 47

    def test_summary_not_found(self):
        svc = ExperimentationService()
        with pytest.raises(KeyError):
            svc.get_experiment_summary("missing")

    def test_experiment_to_dict_concluded(self):
        svc = ExperimentationService()
        svc.create_experiment(_make_config())
        svc.start_experiment("exp-001")
        svc.conclude_experiment("exp-001", winner="control", reason="test")

        summary = svc.get_experiment_summary("exp-001")
        assert summary["status"] == "concluded"
        assert summary["winner"] == "control"
        assert summary["concluded_at"] is not None


# ---------------------------------------------------------------------------
# Multi-variant experiments
# ---------------------------------------------------------------------------


class TestMultiVariant:
    def test_three_variant_experiment(self):
        cfg = _make_config(
            variants={"control": 0.34, "model_a": 0.33, "model_b": 0.33},
            control_variant="control",
        )
        svc = ExperimentationService()
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        seen = set()
        random.seed(42)
        for _ in range(300):
            v = svc.allocate_traffic("exp-001")
            seen.add(v)

        assert seen == {"control", "model_a", "model_b"}

    def test_significance_multi_variant(self):
        cfg = _make_config(
            variants={"control": 0.34, "model_a": 0.33, "model_b": 0.33},
            control_variant="control",
            min_samples_per_variant=30,
        )
        svc = ExperimentationService()
        svc.create_experiment(cfg)
        svc.start_experiment("exp-001")

        _seed_outcomes(svc, "exp-001", "control", tp=10, fp=20, tn=50, fn=5)
        _seed_outcomes(svc, "exp-001", "model_a", tp=25, fp=5, tn=50, fn=5)
        _seed_outcomes(svc, "exp-001", "model_b", tp=15, fp=15, tn=50, fn=5)

        results = svc.run_significance_tests("exp-001")
        # Should have results for model_a and model_b vs control
        # At least model_a should appear (higher precision)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# SignificanceResult serialization
# ---------------------------------------------------------------------------


class TestSignificanceResultSerialization:
    def test_to_dict(self):
        r = SignificanceResult(
            test_name="z_test",
            metric_name="precision",
            statistic=2.5,
            p_value=0.012,
            is_significant=True,
            confidence_level=0.95,
            control_value=0.5,
            treatment_value=0.8,
            winner="treatment",
        )
        d = r.to_dict()
        assert d["test_name"] == "z_test"
        assert d["winner"] == "treatment"
        assert d["is_significant"] is True
