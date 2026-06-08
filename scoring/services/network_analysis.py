"""Advanced network / graph analysis for fraud ring detection.

Extends the basic graph analysis with:

  - Multi-entity transaction networks (users <-> merchants <-> devices <-> IPs)
  - Community detection via Louvain-style modularity optimisation
  - PageRank-based risk propagation across the network
  - Anomalous connection pattern detection (velocity, fan-out, shared entities)
  - Network centrality scoring (degree, betweenness, closeness, eigenvector-approx)

All algorithms are implemented without external graph libraries so the service
has zero heavy dependencies beyond the standard library and pydantic.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ── Pydantic models ─────────────────────────────────────────────────────────


class NetworkNode(BaseModel):
    """A vertex in the multi-entity transaction network."""

    node_id: str
    node_type: str  # "user" | "merchant" | "device" | "ip"
    first_seen: datetime
    last_seen: datetime
    transaction_count: int = 0
    total_amount: float = 0.0
    fraud_count: int = 0
    avg_fraud_score: float = 0.0
    risk_score: float = 0.0

    @property
    def fraud_rate(self) -> float:
        if self.transaction_count == 0:
            return 0.0
        return self.fraud_count / self.transaction_count


class NetworkEdge(BaseModel):
    """An edge connecting two entities in the network."""

    source_id: str
    target_id: str
    weight: float = 1.0
    amount: float = 0.0
    timestamp: datetime | None = None
    fraud_score: float = 0.0
    is_flagged: bool = False


class CommunityResult(BaseModel):
    """Result of community detection for a group of nodes."""

    community_id: int
    member_ids: list[str]
    size: int
    internal_edges: int
    density: float
    avg_fraud_score: float
    fraud_rate: float
    risk_level: str  # "low" | "medium" | "high" | "critical"


class FraudRingCandidate(BaseModel):
    """A candidate fraud ring identified by network analysis."""

    ring_id: str
    member_ids: list[str]
    size: int
    internal_edge_count: int
    fraud_rate: float
    avg_fraud_score: float
    total_amount: float
    confidence: float
    detection_method: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AnomalousPattern(BaseModel):
    """An anomalous connection pattern detected in the network."""

    pattern_type: str  # "velocity_spike" | "fan_out" | "shared_device" | "ip_hopping" | "bridge_node"
    node_id: str
    severity: str  # "low" | "medium" | "high" | "critical"
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CentralityScores(BaseModel):
    """Centrality metrics for a single node."""

    node_id: str
    node_type: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float
    composite_score: float
    risk_factors: list[str] = Field(default_factory=list)


class NetworkRiskProfile(BaseModel):
    """Full risk profile combining all network analysis signals."""

    node_id: str
    node_type: str
    pagerank_score: float
    centrality: CentralityScores
    community_risk: float
    neighbor_risk: float
    anomalous_patterns: list[AnomalousPattern]
    overall_risk_score: float
    risk_level: str


# ── Internal lightweight edge ────────────────────────────────────────────────


class _Edge:
    """Lightweight edge for internal graph operations."""

    __slots__ = (
        "source_id", "target_id", "weight", "amount",
        "timestamp_ts", "fraud_score", "is_flagged",
    )

    def __init__(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        amount: float,
        timestamp_ts: float,
        fraud_score: float,
        is_flagged: bool,
    ) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.amount = amount
        self.timestamp_ts = timestamp_ts
        self.fraud_score = fraud_score
        self.is_flagged = is_flagged


# ── Main service ─────────────────────────────────────────────────────────────


class NetworkAnalyzer:
    """Advanced network analysis engine for fraud detection.

    Builds a multi-entity graph connecting users, merchants, devices, and IPs.
    Provides community detection, PageRank risk propagation, anomalous pattern
    detection, and multi-metric centrality scoring.

    Thread-safety: public async methods acquire ``_lock`` before mutating
    shared graph state.
    """

    _PAGERANK_EPSILON: float = 1e-6
    _PAGERANK_MAX_ITER: int = 100
    _LOUVAIN_MAX_ITER: int = 20

    def __init__(
        self,
        fraud_threshold: float = 0.6,
        max_edges: int = 500_000,
    ) -> None:
        self._fraud_threshold = fraud_threshold
        self._max_edges = max_edges

        # node_id -> NetworkNode
        self._nodes: dict[str, NetworkNode] = {}

        # Undirected adjacency: node_id -> set of neighbour node_ids
        self._adj: dict[str, set[str]] = defaultdict(set)

        # Directed adjacency: node_id -> list of target node_ids
        self._out_adj: dict[str, list[str]] = defaultdict(list)

        # Edge storage
        self._edges: list[_Edge] = []

        # node_id -> list of edges touching this node
        self._node_edges: dict[str, list[_Edge]] = defaultdict(list)

        # Weighted adjacency for modularity: (u, v) -> total weight
        self._edge_weights: dict[tuple[str, str], float] = defaultdict(float)

        # Caches
        self._pagerank_cache: dict[str, float] | None = None
        self._community_cache: list[CommunityResult] | None = None
        self._centrality_cache: dict[str, CentralityScores] | None = None
        self._dirty: bool = True

        self._lock = asyncio.Lock()
        logger.info("network_analyzer.initialized", max_edges=max_edges)

    # ── Graph construction ───────────────────────────────────────────────────

    async def add_transaction(
        self,
        user_id: str,
        merchant_id: str,
        device_id: str,
        ip_address: str,
        amount: float,
        timestamp: datetime,
        fraud_score: float = 0.0,
    ) -> None:
        """Add a transaction creating edges: user-merchant, device-user, ip-user, ip-device."""
        is_flagged = fraud_score >= self._fraud_threshold
        ts = timestamp.timestamp()

        async with self._lock:
            # Evict oldest edges when at capacity
            if len(self._edges) >= self._max_edges:
                cutoff = int(self._max_edges * 0.1)
                self._edges = self._edges[cutoff:]
                logger.warning("network_analyzer.edge_eviction", evicted=cutoff)

            # Upsert nodes
            for nid, ntype in (
                (user_id, "user"),
                (merchant_id, "merchant"),
                (device_id, "device"),
                (ip_address, "ip"),
            ):
                if nid not in self._nodes:
                    self._nodes[nid] = NetworkNode(
                        node_id=nid,
                        node_type=ntype,
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )
                node = self._nodes[nid]
                node.transaction_count += 1
                node.total_amount += amount
                node.last_seen = max(node.last_seen, timestamp)
                node.avg_fraud_score = (
                    node.avg_fraud_score * (node.transaction_count - 1) + fraud_score
                ) / node.transaction_count
                if is_flagged:
                    node.fraud_count += 1

            # Create edges for all entity relationships
            pairs = [
                (user_id, merchant_id),
                (device_id, user_id),
                (ip_address, user_id),
                (ip_address, device_id),
            ]
            for src, tgt in pairs:
                edge = _Edge(
                    source_id=src,
                    target_id=tgt,
                    weight=1.0,
                    amount=amount,
                    timestamp_ts=ts,
                    fraud_score=fraud_score,
                    is_flagged=is_flagged,
                )
                self._edges.append(edge)
                self._node_edges[src].append(edge)
                self._node_edges[tgt].append(edge)
                self._adj[src].add(tgt)
                self._adj[tgt].add(src)
                self._out_adj[src].append(tgt)
                # Canonical key for weighted adjacency
                key = (min(src, tgt), max(src, tgt))
                self._edge_weights[key] += 1.0

            self._dirty = True

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def get_node(self, node_id: str) -> NetworkNode | None:
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> set[str]:
        return set(self._adj.get(node_id, set()))

    # ── Community detection (Louvain-style modularity optimisation) ───────────

    async def detect_communities(self) -> list[CommunityResult]:
        """Detect communities using Louvain-style greedy modularity optimisation.

        Returns communities sorted by descending fraud rate.
        """
        if not self._dirty and self._community_cache is not None:
            return self._community_cache

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._louvain_communities,
        )
        self._community_cache = result
        return result

    def _louvain_communities(self) -> list[CommunityResult]:
        """Single-level Louvain modularity optimisation."""
        nodes = list(self._nodes.keys())
        if not nodes:
            return []

        # Total edge weight (sum of all edge weights, each undirected edge counted once)
        m = sum(self._edge_weights.values())
        if m == 0:
            # No edges: each node is its own community
            return [
                self._build_community(cid, [nid])
                for cid, nid in enumerate(nodes)
            ]

        # Initialise: each node in its own community
        node_to_comm: dict[str, int] = {nid: i for i, nid in enumerate(nodes)}
        # Weighted degree for each node
        k: dict[str, float] = defaultdict(float)
        for (u, v), w in self._edge_weights.items():
            k[u] += w
            k[v] += w

        for _ in range(self._LOUVAIN_MAX_ITER):
            improved = False
            random.shuffle(nodes)
            for nid in nodes:
                current_comm = node_to_comm[nid]
                best_comm = current_comm
                best_gain = 0.0

                # Gather neighbouring communities
                neighbor_comms: dict[int, float] = defaultdict(float)
                for nb in self._adj.get(nid, set()):
                    key = (min(nid, nb), max(nid, nb))
                    w = self._edge_weights.get(key, 0.0)
                    neighbor_comms[node_to_comm[nb]] += w

                # Sum of weights for current community (excluding nid)
                ki = k.get(nid, 0.0)

                for comm, w_ic in neighbor_comms.items():
                    if comm == current_comm:
                        continue
                    # Modularity gain for moving nid to comm
                    # Sum of weights inside target community
                    sigma_tot = sum(
                        k.get(n2, 0.0)
                        for n2, c in node_to_comm.items()
                        if c == comm
                    )
                    gain = w_ic - (sigma_tot * ki) / (2.0 * m)
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm:
                    node_to_comm[nid] = best_comm
                    improved = True

            if not improved:
                break

        # Group nodes by community
        groups: dict[int, list[str]] = defaultdict(list)
        for nid, cid in node_to_comm.items():
            groups[cid].append(nid)

        communities = [
            self._build_community(idx, members)
            for idx, (_, members) in enumerate(groups.items())
        ]
        communities.sort(key=lambda c: c.fraud_rate, reverse=True)
        return communities

    def _build_community(self, cid: int, members: list[str]) -> CommunityResult:
        member_set = set(members)
        internal_edges = 0
        fraud_count = 0
        txn_count = 0
        total_fraud_score = 0.0

        for nid in members:
            node = self._nodes[nid]
            fraud_count += node.fraud_count
            txn_count += node.transaction_count
            total_fraud_score += node.avg_fraud_score
            for nb in self._adj.get(nid, set()):
                if nb in member_set and nb > nid:
                    internal_edges += 1

        n = len(members)
        max_possible = n * (n - 1) / 2 if n > 1 else 1
        density = internal_edges / max_possible if max_possible > 0 else 0.0
        avg_fraud = total_fraud_score / n if n else 0.0
        fraud_rate = fraud_count / txn_count if txn_count else 0.0

        if fraud_rate >= 0.5:
            risk_level = "critical"
        elif fraud_rate >= 0.3:
            risk_level = "high"
        elif fraud_rate >= 0.1:
            risk_level = "medium"
        else:
            risk_level = "low"

        return CommunityResult(
            community_id=cid,
            member_ids=members,
            size=n,
            internal_edges=internal_edges,
            density=round(density, 4),
            avg_fraud_score=round(avg_fraud, 4),
            fraud_rate=round(fraud_rate, 4),
            risk_level=risk_level,
        )

    # ── Fraud ring identification ────────────────────────────────────────────

    async def detect_fraud_rings(
        self,
        min_fraud_rate: float = 0.3,
        min_size: int = 3,
        min_density: float = 0.3,
    ) -> list[FraudRingCandidate]:
        """Identify fraud rings from communities with high fraud rates and density."""
        communities = await self.detect_communities()
        rings: list[FraudRingCandidate] = []

        for comm in communities:
            if (
                comm.fraud_rate < min_fraud_rate
                or comm.size < min_size
                or comm.density < min_density
            ):
                continue

            member_set = set(comm.member_ids)
            total_amount = 0.0
            fraud_scores: list[float] = []
            for edge in self._edges:
                if edge.source_id in member_set and edge.target_id in member_set:
                    total_amount += edge.amount
                    if edge.is_flagged:
                        fraud_scores.append(edge.fraud_score)

            avg_score = sum(fraud_scores) / len(fraud_scores) if fraud_scores else 0.0
            confidence = min(
                1.0,
                (comm.fraud_rate * 0.4) + (comm.density * 0.3) + (avg_score * 0.3),
            )

            ring_id = f"ring:{comm.community_id}:{sorted(comm.member_ids)[0]}"
            rings.append(FraudRingCandidate(
                ring_id=ring_id,
                member_ids=comm.member_ids,
                size=comm.size,
                internal_edge_count=comm.internal_edges,
                fraud_rate=comm.fraud_rate,
                avg_fraud_score=round(avg_score, 4),
                total_amount=round(total_amount, 2),
                confidence=round(confidence, 4),
                detection_method="louvain_community",
            ))

        rings.sort(key=lambda r: r.confidence, reverse=True)
        return rings

    # ── PageRank-based risk propagation ──────────────────────────────────────

    async def compute_pagerank(self, damping: float = 0.85) -> dict[str, float]:
        """Compute PageRank scores using power iteration.

        Returns node_id -> score mapping (scores sum to 1.0).
        """
        if not self._dirty and self._pagerank_cache is not None:
            return self._pagerank_cache

        pr = await asyncio.get_event_loop().run_in_executor(
            None, self._pagerank_sync, damping,
        )
        self._pagerank_cache = pr
        return pr

    def _pagerank_sync(self, damping: float) -> dict[str, float]:
        nodes = list(self._nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}

        idx = {nid: i for i, nid in enumerate(nodes)}
        teleport = (1.0 - damping) / n
        pr = [1.0 / n] * n
        out_degree = [len(self._out_adj.get(nid, [])) for nid in nodes]

        for _ in range(self._PAGERANK_MAX_ITER):
            new_pr = [teleport] * n
            for nid in nodes:
                i = idx[nid]
                od = out_degree[i]
                if od == 0:
                    contrib = damping * pr[i] / n
                    for j in range(n):
                        new_pr[j] += contrib
                else:
                    contrib = damping * pr[i] / od
                    for nb in self._out_adj.get(nid, []):
                        new_pr[idx[nb]] += contrib

            delta = sum(abs(new_pr[i] - pr[i]) for i in range(n))
            pr = new_pr
            if delta < self._PAGERANK_EPSILON:
                break

        return {nodes[i]: round(pr[i], 8) for i in range(n)}

    async def propagate_risk(self, iterations: int = 3, decay: float = 0.5) -> dict[str, float]:
        """Propagate risk scores through the network.

        Starting from each node's direct fraud rate, iteratively spread risk
        to neighbours with exponential decay. Returns node_id -> propagated risk.
        """
        # Initialise with direct fraud rates
        risk: dict[str, float] = {
            nid: node.fraud_rate for nid, node in self._nodes.items()
        }

        for _ in range(iterations):
            new_risk: dict[str, float] = {}
            for nid in self._nodes:
                neighbors = self._adj.get(nid, set())
                if not neighbors:
                    new_risk[nid] = risk.get(nid, 0.0)
                    continue
                neighbor_risk = sum(risk.get(nb, 0.0) for nb in neighbors) / len(neighbors)
                new_risk[nid] = risk[nid] + decay * neighbor_risk
            # Normalise to [0, 1]
            max_risk = max(new_risk.values()) if new_risk else 1.0
            if max_risk > 0:
                risk = {nid: min(v / max_risk, 1.0) for nid, v in new_risk.items()}
            else:
                risk = new_risk

        return {nid: round(v, 4) for nid, v in risk.items()}

    # ── Anomalous connection pattern detection ───────────────────────────────

    async def detect_anomalous_patterns(
        self,
        velocity_threshold: int = 10,
        fan_out_threshold: int = 5,
        window_seconds: float = 3600.0,
    ) -> list[AnomalousPattern]:
        """Detect anomalous connection patterns in the network.

        Checks for:
          - Velocity spikes: nodes with unusually many edges in a time window
          - Fan-out: users transacting with many merchants rapidly
          - Shared device: multiple users sharing a device
          - IP hopping: single user appearing from many IPs
          - Bridge nodes: high betweenness nodes connecting otherwise separate clusters
        """
        patterns: list[AnomalousPattern] = []
        now_ts = time.time()
        cutoff_ts = now_ts - window_seconds

        # Velocity spike detection
        for nid, edges in self._node_edges.items():
            recent = [e for e in edges if e.timestamp_ts >= cutoff_ts]
            if len(recent) >= velocity_threshold:
                node = self._nodes.get(nid)
                if node is None:
                    continue
                patterns.append(AnomalousPattern(
                    pattern_type="velocity_spike",
                    node_id=nid,
                    severity="high" if len(recent) >= velocity_threshold * 2 else "medium",
                    description=f"Node {nid} has {len(recent)} edges in the last {window_seconds}s",
                    evidence={"edge_count": len(recent), "window_seconds": window_seconds},
                ))

        # Fan-out: users connected to many merchants
        for nid, node in self._nodes.items():
            if node.node_type != "user":
                continue
            merchant_neighbors = {
                nb for nb in self._adj.get(nid, set())
                if self._nodes.get(nb) and self._nodes[nb].node_type == "merchant"
            }
            if len(merchant_neighbors) >= fan_out_threshold:
                patterns.append(AnomalousPattern(
                    pattern_type="fan_out",
                    node_id=nid,
                    severity="high" if len(merchant_neighbors) >= fan_out_threshold * 2 else "medium",
                    description=f"User {nid} transacted with {len(merchant_neighbors)} merchants",
                    evidence={"merchant_count": len(merchant_neighbors)},
                ))

        # Shared device: devices connected to multiple users
        for nid, node in self._nodes.items():
            if node.node_type != "device":
                continue
            user_neighbors = {
                nb for nb in self._adj.get(nid, set())
                if self._nodes.get(nb) and self._nodes[nb].node_type == "user"
            }
            if len(user_neighbors) >= 3:
                patterns.append(AnomalousPattern(
                    pattern_type="shared_device",
                    node_id=nid,
                    severity="high" if len(user_neighbors) >= 5 else "medium",
                    description=f"Device {nid} shared by {len(user_neighbors)} users",
                    evidence={"user_count": len(user_neighbors)},
                ))

        # IP hopping: users connected to many IPs
        for nid, node in self._nodes.items():
            if node.node_type != "user":
                continue
            ip_neighbors = {
                nb for nb in self._adj.get(nid, set())
                if self._nodes.get(nb) and self._nodes[nb].node_type == "ip"
            }
            if len(ip_neighbors) >= 3:
                patterns.append(AnomalousPattern(
                    pattern_type="ip_hopping",
                    node_id=nid,
                    severity="high" if len(ip_neighbors) >= 5 else "medium",
                    description=f"User {nid} seen from {len(ip_neighbors)} IPs",
                    evidence={"ip_count": len(ip_neighbors)},
                ))

        # Bridge node detection via approximate betweenness
        betweenness = self._compute_betweenness()
        if betweenness:
            threshold = _percentile(list(betweenness.values()), 95)
            for nid, score in betweenness.items():
                if score > threshold and threshold > 0:
                    patterns.append(AnomalousPattern(
                        pattern_type="bridge_node",
                        node_id=nid,
                        severity="medium",
                        description=f"Node {nid} is a high-betweenness bridge (score={score:.4f})",
                        evidence={"betweenness": round(score, 6), "threshold": round(threshold, 6)},
                    ))

        return patterns

    # ── Network centrality scoring ───────────────────────────────────────────

    async def compute_centrality(self, node_id: str) -> CentralityScores:
        """Compute multi-metric centrality for a specific node."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not in graph")

        node = self._nodes[node_id]
        degree = self._degree_centrality(node_id)
        betweenness = self._compute_betweenness().get(node_id, 0.0)
        closeness = self._closeness_centrality(node_id)
        eigenvector = self._eigenvector_centrality().get(node_id, 0.0)

        composite = (
            degree * 0.25
            + betweenness * 0.30
            + closeness * 0.20
            + eigenvector * 0.25
        )

        risk_factors: list[str] = []
        if degree > 0.3:
            risk_factors.append("high_degree")
        if betweenness > 0.1:
            risk_factors.append("high_betweenness")
        if closeness > 0.5:
            risk_factors.append("high_closeness")
        if eigenvector > 0.3:
            risk_factors.append("high_eigenvector")

        return CentralityScores(
            node_id=node_id,
            node_type=node.node_type,
            degree_centrality=round(degree, 6),
            betweenness_centrality=round(betweenness, 6),
            closeness_centrality=round(closeness, 6),
            eigenvector_centrality=round(eigenvector, 6),
            composite_score=round(composite, 6),
            risk_factors=risk_factors,
        )

    def _degree_centrality(self, node_id: str) -> float:
        n = len(self._nodes)
        if n <= 1:
            return 0.0
        return len(self._adj.get(node_id, set())) / (n - 1)

    def _closeness_centrality(self, node_id: str) -> float:
        """BFS-based closeness centrality for a single node."""
        if node_id not in self._nodes:
            return 0.0
        n = len(self._nodes)
        if n <= 1:
            return 0.0

        dist: dict[str, int] = {node_id: 0}
        queue: deque[str] = deque([node_id])
        while queue:
            v = queue.popleft()
            for nb in self._adj.get(v, set()):
                if nb not in dist:
                    dist[nb] = dist[v] + 1
                    queue.append(nb)

        reachable = len(dist) - 1  # exclude self
        if reachable == 0:
            return 0.0
        total_dist = sum(dist.values())
        if total_dist == 0:
            return 0.0
        return reachable / total_dist

    def _compute_betweenness(self) -> dict[str, float]:
        """Approximate betweenness centrality using Brandes' algorithm with sampling."""
        nodes = list(self._nodes.keys())
        n = len(nodes)
        if n <= 2:
            return {nid: 0.0 for nid in nodes}

        sample_size = min(n, 100)
        sources = random.sample(nodes, sample_size)
        scale = n / sample_size

        betweenness: dict[str, float] = defaultdict(float)

        for s in sources:
            stack: list[str] = []
            pred: dict[str, list[str]] = {nid: [] for nid in nodes}
            sigma: dict[str, float] = defaultdict(float)
            sigma[s] = 1.0
            dist: dict[str, int] = {nid: -1 for nid in nodes}
            dist[s] = 0
            queue: deque[str] = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in self._adj.get(v, set()):
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta: dict[str, float] = defaultdict(float)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        norm = (n - 1) * (n - 2) if n > 2 else 1.0
        return {
            nid: round(betweenness.get(nid, 0.0) * scale / norm, 8)
            for nid in nodes
        }

    def _eigenvector_centrality(self, max_iter: int = 50, tol: float = 1e-6) -> dict[str, float]:
        """Approximate eigenvector centrality via power iteration."""
        nodes = list(self._nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}

        idx = {nid: i for i, nid in enumerate(nodes)}
        x = [1.0 / n] * n

        for _ in range(max_iter):
            new_x = [0.0] * n
            for nid in nodes:
                i = idx[nid]
                for nb in self._adj.get(nid, set()):
                    new_x[i] += x[idx[nb]]

            norm = math.sqrt(sum(v * v for v in new_x)) or 1.0
            new_x = [v / norm for v in new_x]

            delta = sum(abs(new_x[i] - x[i]) for i in range(n))
            x = new_x
            if delta < tol:
                break

        return {nodes[i]: round(x[i], 8) for i in range(n)}

    # ── Full risk profile ────────────────────────────────────────────────────

    async def get_risk_profile(self, node_id: str) -> NetworkRiskProfile:
        """Build a comprehensive risk profile for a node."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not in graph")

        node = self._nodes[node_id]

        pr_map = await self.compute_pagerank()
        pagerank = pr_map.get(node_id, 0.0)

        centrality = await self.compute_centrality(node_id)

        communities = await self.detect_communities()
        community_risk = 0.0
        for comm in communities:
            if node_id in comm.member_ids:
                community_risk = comm.fraud_rate
                break

        neighbors = self._adj.get(node_id, set())
        if neighbors:
            neighbor_risk = sum(
                self._nodes[nb].fraud_rate
                for nb in neighbors
                if nb in self._nodes
            ) / len(neighbors)
        else:
            neighbor_risk = 0.0

        patterns = await self.detect_anomalous_patterns()
        node_patterns = [p for p in patterns if p.node_id == node_id]

        overall = min(
            1.0,
            (node.fraud_rate * 0.30)
            + (community_risk * 0.20)
            + (neighbor_risk * 0.15)
            + (min(pagerank * 500, 1.0) * 0.10)
            + (centrality.composite_score * 0.15)
            + (min(len(node_patterns) * 0.1, 0.3) * 0.10),
        )

        if overall >= 0.7:
            risk_level = "critical"
        elif overall >= 0.5:
            risk_level = "high"
        elif overall >= 0.2:
            risk_level = "medium"
        else:
            risk_level = "low"

        return NetworkRiskProfile(
            node_id=node_id,
            node_type=node.node_type,
            pagerank_score=round(pagerank, 8),
            centrality=centrality,
            community_risk=round(community_risk, 4),
            neighbor_risk=round(neighbor_risk, 4),
            anomalous_patterns=node_patterns,
            overall_risk_score=round(overall, 4),
            risk_level=risk_level,
        )

    # ── Graph summary ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Return summary statistics of the network."""
        type_counts: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            type_counts[node.node_type] += 1

        total_nodes = len(self._nodes)
        total_edges = len(self._edges)
        flagged = sum(1 for e in self._edges if e.is_flagged)

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "flagged_edges": flagged,
            "node_type_counts": dict(type_counts),
            "avg_degree": round(
                sum(len(nb) for nb in self._adj.values()) / total_nodes, 2
            ) if total_nodes else 0.0,
        }

    async def clear(self) -> None:
        """Reset the graph."""
        async with self._lock:
            self._nodes.clear()
            self._adj.clear()
            self._out_adj.clear()
            self._edges.clear()
            self._node_edges.clear()
            self._edge_weights.clear()
            self._pagerank_cache = None
            self._community_cache = None
            self._centrality_cache = None
            self._dirty = True
        logger.info("network_analyzer.cleared")


# ── Utility ──────────────────────────────────────────────────────────────────


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of a sorted list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])
