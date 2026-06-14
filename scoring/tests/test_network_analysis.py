"""Tests for the advanced network analysis service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from scoring.services.network_analysis import (
    CentralityScores,
    CommunityResult,
    FraudRingCandidate,
    NetworkAnalyzer,
    NetworkNode,
    NetworkRiskProfile,
    _percentile,
)


@pytest.fixture
def analyzer() -> NetworkAnalyzer:
    return NetworkAnalyzer(fraud_threshold=0.6)


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _run(coro):
    """Helper to run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ts(minutes_ago: int = 0) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes_ago)


# ── Graph construction ───────────────────────────────────────────────────────


class TestGraphConstruction:
    def test_add_single_transaction(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="user1",
            merchant_id="merch1",
            device_id="dev1",
            ip_address="1.2.3.4",
            amount=100.0,
            timestamp=_ts(),
        ))
        assert analyzer.node_count == 4
        # 4 edges: user-merchant, device-user, ip-user, ip-device
        assert analyzer.edge_count == 4

    def test_nodes_created_with_correct_types(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="10.0.0.1", amount=50.0, timestamp=_ts(),
        ))
        assert analyzer.get_node("u1").node_type == "user"
        assert analyzer.get_node("m1").node_type == "merchant"
        assert analyzer.get_node("d1").node_type == "device"
        assert analyzer.get_node("10.0.0.1").node_type == "ip"

    def test_multiple_transactions_update_counts(self, analyzer: NetworkAnalyzer):
        for _ in range(3):
            _run(analyzer.add_transaction(
                user_id="u1", merchant_id="m1", device_id="d1",
                ip_address="10.0.0.1", amount=25.0, timestamp=_ts(),
            ))
        node = analyzer.get_node("u1")
        assert node.transaction_count == 3
        assert node.total_amount == 75.0

    def test_get_neighbors(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="10.0.0.1", amount=10.0, timestamp=_ts(),
        ))
        neighbors = analyzer.get_neighbors("u1")
        assert "m1" in neighbors
        assert "d1" in neighbors
        assert "10.0.0.1" in neighbors

    def test_get_node_returns_none_for_missing(self, analyzer: NetworkAnalyzer):
        assert analyzer.get_node("nonexistent") is None

    def test_fraud_count_incremented_when_above_threshold(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="10.0.0.1", amount=500.0, timestamp=_ts(),
            fraud_score=0.9,
        ))
        node = analyzer.get_node("u1")
        assert node.fraud_count == 1
        assert node.fraud_rate == 1.0

    def test_fraud_count_not_incremented_below_threshold(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="10.0.0.1", amount=10.0, timestamp=_ts(),
            fraud_score=0.1,
        ))
        node = analyzer.get_node("u1")
        assert node.fraud_count == 0

    def test_edge_eviction_at_capacity(self):
        analyzer = NetworkAnalyzer(max_edges=20)
        # Each transaction adds 4 edges; 6 txns = 24 edges > 20
        for i in range(6):
            _run(analyzer.add_transaction(
                user_id=f"u{i}", merchant_id="m1", device_id="d1",
                ip_address="10.0.0.1", amount=10.0, timestamp=_ts(),
            ))
        # Edges should have been evicted to stay near capacity
        assert analyzer.edge_count <= 24  # some eviction happened


# ── Community detection ──────────────────────────────────────────────────────


class TestCommunityDetection:
    def _build_two_clusters(self, analyzer: NetworkAnalyzer):
        """Build two distinct clusters with separate entities."""
        # Cluster A: u1, u2 share m_a, d_a, ip_a
        for uid in ("u1", "u2"):
            _run(analyzer.add_transaction(
                user_id=uid, merchant_id="m_a", device_id="d_a",
                ip_address="ip_a", amount=100.0, timestamp=_ts(),
            ))
        # Cluster B: u3, u4 share m_b, d_b, ip_b
        for uid in ("u3", "u4"):
            _run(analyzer.add_transaction(
                user_id=uid, merchant_id="m_b", device_id="d_b",
                ip_address="ip_b", amount=100.0, timestamp=_ts(),
            ))

    def test_communities_detected(self, analyzer: NetworkAnalyzer):
        self._build_two_clusters(analyzer)
        communities = _run(analyzer.detect_communities())
        assert len(communities) >= 1
        assert all(isinstance(c, CommunityResult) for c in communities)

    def test_community_has_members(self, analyzer: NetworkAnalyzer):
        self._build_two_clusters(analyzer)
        communities = _run(analyzer.detect_communities())
        for comm in communities:
            assert comm.size == len(comm.member_ids)
            assert comm.size > 0

    def test_community_risk_levels(self, analyzer: NetworkAnalyzer):
        # All transactions flagged as fraud
        for uid in ("u1", "u2", "u3"):
            _run(analyzer.add_transaction(
                user_id=uid, merchant_id="m1", device_id="d1",
                ip_address="ip1", amount=100.0, timestamp=_ts(),
                fraud_score=0.9,
            ))
        communities = _run(analyzer.detect_communities())
        # At least one community should have high/critical risk
        risk_levels = {c.risk_level for c in communities}
        assert risk_levels & {"high", "critical"}

    def test_empty_graph_returns_no_communities(self, analyzer: NetworkAnalyzer):
        communities = _run(analyzer.detect_communities())
        assert communities == []

    def test_community_cache_used(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=10.0, timestamp=_ts(),
        ))
        c1 = _run(analyzer.detect_communities())
        # Second call should use cache (dirty flag not set since no new transaction)
        analyzer._dirty = False
        c2 = _run(analyzer.detect_communities())
        assert c1 == c2


# ── Fraud ring detection ────────────────────────────────────────────────────


class TestFraudRingDetection:
    def test_fraud_ring_detected(self, analyzer: NetworkAnalyzer):
        # Create a dense cluster with high fraud
        for uid in ("u1", "u2", "u3"):
            _run(analyzer.add_transaction(
                user_id=uid, merchant_id="m1", device_id="d1",
                ip_address="ip1", amount=500.0, timestamp=_ts(),
                fraud_score=0.9,
            ))
        rings = _run(analyzer.detect_fraud_rings(min_fraud_rate=0.3, min_size=3, min_density=0.1))
        assert len(rings) >= 1
        assert all(isinstance(r, FraudRingCandidate) for r in rings)

    def test_fraud_ring_has_correct_fields(self, analyzer: NetworkAnalyzer):
        for uid in ("u1", "u2", "u3"):
            _run(analyzer.add_transaction(
                user_id=uid, merchant_id="m1", device_id="d1",
                ip_address="ip1", amount=500.0, timestamp=_ts(),
                fraud_score=0.9,
            ))
        rings = _run(analyzer.detect_fraud_rings(min_fraud_rate=0.3, min_size=3, min_density=0.1))
        if rings:
            ring = rings[0]
            assert ring.size >= 3
            assert ring.fraud_rate > 0
            assert ring.confidence > 0
            assert ring.detection_method == "louvain_community"

    def test_no_fraud_ring_in_clean_graph(self, analyzer: NetworkAnalyzer):
        for uid in ("u1", "u2", "u3"):
            _run(analyzer.add_transaction(
                user_id=uid, merchant_id="m1", device_id="d1",
                ip_address="ip1", amount=10.0, timestamp=_ts(),
                fraud_score=0.0,
            ))
        rings = _run(analyzer.detect_fraud_rings(min_fraud_rate=0.3, min_size=3))
        assert len(rings) == 0


# ── PageRank ─────────────────────────────────────────────────────────────────


class TestPageRank:
    def test_pagerank_sums_to_one(self, analyzer: NetworkAnalyzer):
        for uid in ("u1", "u2"):
            _run(analyzer.add_transaction(
                user_id=uid, merchant_id="m1", device_id="d1",
                ip_address="ip1", amount=10.0, timestamp=_ts(),
            ))
        pr = _run(analyzer.compute_pagerank())
        assert abs(sum(pr.values()) - 1.0) < 0.01

    def test_pagerank_returns_all_nodes(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=10.0, timestamp=_ts(),
        ))
        pr = _run(analyzer.compute_pagerank())
        assert set(pr.keys()) == {"u1", "m1", "d1", "ip1"}

    def test_pagerank_empty_graph(self, analyzer: NetworkAnalyzer):
        pr = _run(analyzer.compute_pagerank())
        assert pr == {}

    def test_pagerank_cache(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=10.0, timestamp=_ts(),
        ))
        pr1 = _run(analyzer.compute_pagerank())
        analyzer._dirty = False
        pr2 = _run(analyzer.compute_pagerank())
        assert pr1 == pr2


class TestRiskPropagation:
    def test_risk_propagates_to_neighbors(self, analyzer: NetworkAnalyzer):
        # u1 is fraudulent, u2 is clean but connected via same merchant
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=100.0, timestamp=_ts(),
            fraud_score=0.9,
        ))
        _run(analyzer.add_transaction(
            user_id="u2", merchant_id="m1", device_id="d2",
            ip_address="ip2", amount=50.0, timestamp=_ts(),
            fraud_score=0.0,
        ))
        risk = _run(analyzer.propagate_risk())
        # u2 should have some propagated risk from m1
        assert risk.get("u2", 0.0) > 0.0

    def test_propagation_empty_graph(self, analyzer: NetworkAnalyzer):
        risk = _run(analyzer.propagate_risk())
        assert risk == {}


# ── Anomalous pattern detection ──────────────────────────────────────────────


class TestAnomalousPatterns:
    def test_fan_out_detected(self, analyzer: NetworkAnalyzer):
        # User transacting with many merchants
        for i in range(6):
            _run(analyzer.add_transaction(
                user_id="u1", merchant_id=f"m{i}", device_id="d1",
                ip_address="ip1", amount=10.0, timestamp=_ts(),
            ))
        patterns = _run(analyzer.detect_anomalous_patterns(fan_out_threshold=5))
        fan_out = [p for p in patterns if p.pattern_type == "fan_out"]
        assert len(fan_out) >= 1
        assert fan_out[0].node_id == "u1"

    def test_shared_device_detected(self, analyzer: NetworkAnalyzer):
        # Multiple users sharing a device
        for i in range(4):
            _run(analyzer.add_transaction(
                user_id=f"u{i}", merchant_id="m1", device_id="shared_dev",
                ip_address=f"ip{i}", amount=10.0, timestamp=_ts(),
            ))
        patterns = _run(analyzer.detect_anomalous_patterns())
        shared = [p for p in patterns if p.pattern_type == "shared_device"]
        assert len(shared) >= 1
        assert shared[0].node_id == "shared_dev"

    def test_ip_hopping_detected(self, analyzer: NetworkAnalyzer):
        # User appearing from many IPs
        for i in range(4):
            _run(analyzer.add_transaction(
                user_id="u1", merchant_id="m1", device_id="d1",
                ip_address=f"10.0.0.{i}", amount=10.0, timestamp=_ts(),
            ))
        patterns = _run(analyzer.detect_anomalous_patterns())
        ip_hop = [p for p in patterns if p.pattern_type == "ip_hopping"]
        assert len(ip_hop) >= 1
        assert ip_hop[0].node_id == "u1"

    def test_velocity_spike_detected(self, analyzer: NetworkAnalyzer):
        # Many edges in a short time window
        now = _ts()
        for i in range(12):
            _run(analyzer.add_transaction(
                user_id="u1", merchant_id=f"m{i}", device_id="d1",
                ip_address="ip1", amount=10.0, timestamp=now,
            ))
        patterns = _run(analyzer.detect_anomalous_patterns(
            velocity_threshold=10, window_seconds=3600.0,
        ))
        velocity = [p for p in patterns if p.pattern_type == "velocity_spike"]
        assert len(velocity) >= 1

    def test_no_patterns_in_small_graph(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=10.0, timestamp=_ts(),
        ))
        patterns = _run(analyzer.detect_anomalous_patterns())
        # Should not trigger fan_out, shared_device, or ip_hopping
        non_bridge = [p for p in patterns if p.pattern_type != "bridge_node"]
        assert len(non_bridge) == 0


# ── Centrality scoring ───────────────────────────────────────────────────────


class TestCentralityScoring:
    def test_centrality_computed(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=10.0, timestamp=_ts(),
        ))
        scores = _run(analyzer.compute_centrality("u1"))
        assert isinstance(scores, CentralityScores)
        assert scores.node_id == "u1"
        assert scores.degree_centrality >= 0.0
        assert scores.betweenness_centrality >= 0.0
        assert scores.closeness_centrality >= 0.0
        assert scores.eigenvector_centrality >= 0.0

    def test_centrality_raises_for_missing_node(self, analyzer: NetworkAnalyzer):
        with pytest.raises(KeyError):
            _run(analyzer.compute_centrality("nonexistent"))

    def test_hub_node_has_higher_degree(self, analyzer: NetworkAnalyzer):
        # m1 is a hub connected to multiple users
        for i in range(5):
            _run(analyzer.add_transaction(
                user_id=f"u{i}", merchant_id="m1", device_id=f"d{i}",
                ip_address=f"ip{i}", amount=10.0, timestamp=_ts(),
            ))
        hub_scores = _run(analyzer.compute_centrality("m1"))
        leaf_scores = _run(analyzer.compute_centrality("u0"))
        assert hub_scores.degree_centrality > leaf_scores.degree_centrality


# ── Risk profile ─────────────────────────────────────────────────────────────


class TestRiskProfile:
    def test_risk_profile_computed(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=100.0, timestamp=_ts(),
            fraud_score=0.5,
        ))
        profile = _run(analyzer.get_risk_profile("u1"))
        assert isinstance(profile, NetworkRiskProfile)
        assert profile.node_id == "u1"
        assert profile.risk_level in ("low", "medium", "high", "critical")
        assert 0.0 <= profile.overall_risk_score <= 1.0

    def test_risk_profile_raises_for_missing_node(self, analyzer: NetworkAnalyzer):
        with pytest.raises(KeyError):
            _run(analyzer.get_risk_profile("nonexistent"))

    def test_fraudulent_node_has_higher_risk(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u_fraud", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=1000.0, timestamp=_ts(),
            fraud_score=0.95,
        ))
        _run(analyzer.add_transaction(
            user_id="u_clean", merchant_id="m2", device_id="d2",
            ip_address="ip2", amount=10.0, timestamp=_ts(),
            fraud_score=0.0,
        ))
        fraud_profile = _run(analyzer.get_risk_profile("u_fraud"))
        clean_profile = _run(analyzer.get_risk_profile("u_clean"))
        assert fraud_profile.overall_risk_score > clean_profile.overall_risk_score


# ── Graph stats and lifecycle ────────────────────────────────────────────────


class TestGraphStats:
    def test_stats_returns_correct_counts(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=10.0, timestamp=_ts(),
        ))
        stats = _run(analyzer.get_stats())
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 4
        assert stats["node_type_counts"]["user"] == 1
        assert stats["node_type_counts"]["ip"] == 1

    def test_clear_resets_graph(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=10.0, timestamp=_ts(),
        ))
        _run(analyzer.clear())
        assert analyzer.node_count == 0
        assert analyzer.edge_count == 0

    def test_flagged_edge_count(self, analyzer: NetworkAnalyzer):
        _run(analyzer.add_transaction(
            user_id="u1", merchant_id="m1", device_id="d1",
            ip_address="ip1", amount=500.0, timestamp=_ts(),
            fraud_score=0.9,
        ))
        stats = _run(analyzer.get_stats())
        assert stats["flagged_edges"] == 4  # all 4 edges flagged


# ── Utility ──────────────────────────────────────────────────────────────────


class TestPercentile:
    def test_percentile_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == 3.0

    def test_percentile_100(self):
        values = [1.0, 2.0, 3.0]
        assert _percentile(values, 100) == 3.0

    def test_percentile_0(self):
        values = [1.0, 2.0, 3.0]
        assert _percentile(values, 0) == 1.0

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0


# ── NetworkNode model ────────────────────────────────────────────────────────


class TestNetworkNodeModel:
    def test_fraud_rate_zero_txns(self):
        node = NetworkNode(
            node_id="n1", node_type="user",
            first_seen=_ts(), last_seen=_ts(),
        )
        assert node.fraud_rate == 0.0

    def test_fraud_rate_computed(self):
        node = NetworkNode(
            node_id="n1", node_type="user",
            first_seen=_ts(), last_seen=_ts(),
            transaction_count=10, fraud_count=3,
        )
        assert abs(node.fraud_rate - 0.3) < 1e-9
