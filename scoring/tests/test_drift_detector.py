"""Tests for the DriftDetector service."""

import random

from scoring.services.drift_detector import DriftDetector


class TestDriftDetector:
    def test_summary_with_no_data(self):
        dd = DriftDetector()
        summary = dd.summary()
        assert summary["samples_added"] == 0
        assert summary["reports_generated"] == 0
        assert summary["latest_drift_detected"] is None

    def test_summary_after_samples(self):
        dd = DriftDetector(reference_window=200, current_window=100)
        # Add reference samples
        for _ in range(200):
            dd.add_sample({"fraud_score": random.uniform(0, 0.3), "amount_zscore": random.gauss(0, 1),
                           "txn_count_1h": random.randint(0, 5), "txn_count_24h": random.randint(0, 50),
                           "distance_from_last_txn_km": random.uniform(0, 100),
                           "amount_to_avg_ratio": random.uniform(0.5, 2),
                           "unique_merchants_24h": random.randint(1, 10),
                           "time_since_last_txn_seconds": random.uniform(60, 3600)})
        # Add current samples
        for _ in range(100):
            dd.add_sample({"fraud_score": random.uniform(0, 0.3), "amount_zscore": random.gauss(0, 1),
                           "txn_count_1h": random.randint(0, 5), "txn_count_24h": random.randint(0, 50),
                           "distance_from_last_txn_km": random.uniform(0, 100),
                           "amount_to_avg_ratio": random.uniform(0.5, 2),
                           "unique_merchants_24h": random.randint(1, 10),
                           "time_since_last_txn_seconds": random.uniform(60, 3600)})

        summary = dd.summary()
        assert summary["samples_added"] == 300
        assert summary["monitored_features"] == DriftDetector.DEFAULT_FEATURES

    def test_check_drift_returns_none_with_insufficient_data(self):
        dd = DriftDetector()
        assert dd.check_drift() is None

    def test_psi_identical_distributions(self):
        data = [float(i) for i in range(100)]
        psi = DriftDetector._compute_psi(data, data)
        assert psi < 0.01  # Identical distributions should have near-zero PSI

    def test_ks_identical_distributions(self):
        data = [float(i) for i in range(100)]
        ks = DriftDetector._compute_ks(data, data)
        assert ks == 0.0

    def test_ks_different_distributions(self):
        ref = [float(i) for i in range(100)]
        cur = [float(i + 50) for i in range(100)]
        ks = DriftDetector._compute_ks(ref, cur)
        assert ks > 0.3  # Shifted distribution should show significant KS
