"""Graph-based fraud network analysis service.

Constructs a transaction graph where users, merchants, and devices are nodes
and transactions are weighted, time-stamped edges. Provides:

  - Community detection via label propagation
  - PageRank for node importance scoring
  - Fraud ring detection (dense subgraphs with elevated fraud rates)
  - Degree / betweenness centrality
  - Bipartite user-merchant analysis
  - Temporal graph analysis (velocity, burst detection)

All graph structures are held in-memory using plain adjacency lists so there
is no dependency on networkx or any other graph library.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

GRAPH_NODES_TOTAL = Gauge(
    "fraud_graph_nodes_total",
    "Total number of nodes currently in the transaction graph",
    ["node_type"],  # user | merchant | device
)

GRAPH_EDGES_TOTAL = Gauge(
    "fraud_graph_edges_total",
    "Total number of edges (transactions) in the transaction graph",
)

FRAUD_RINGS_DETECTED = Counter(
    "fraud_rings_detected_total",
    "Cumulative number of fraud rings identified",
)

COMMUNITY_DETECTION_SECONDS = Histogram(
    "community_detection_duration_seconds",
    "Time spent running label-propagation community detection",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

PAGERANK_COMPUTATION_SECONDS = Histogram(
    "pagerank_computation_duration_seconds",
    "Time spent computing PageRank across the graph",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

HIGH_RISK_NODES = Gauge(
    "fraud_graph_high_risk_nodes_total",
    "Number of nodes whose network risk score exceeds 0.7",
)

GRAPH_ANALYSIS_ERRORS = Counter(
    "fraud_graph_analysis_errors_total",
    "Errors raised during graph analysis operations",
    ["operation"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────


class GraphNode(BaseModel):
    """A vertex in the transaction graph."""

    node_id: str
    node_type: str  # "user" | "merchant" | "device"
    first_seen: datetime
    last_seen: datetime
    transaction_count: int = 0
    total_amount: float = 0.0
    fraud_count: int = 0
    # Running average fraud score for transactions touching this node
    avg_fraud_score: float = 0.0

    @property
    def fraud_rate(self) -> float:
        if self.transaction_count == 0:
            return 0.0
        return self.fraud_count / self.transaction_count


class GraphEdge(BaseModel):
    """A directed edge representing a transaction between two nodes."""

    edge_id: str
    source_id: str
    target_id: str
    amount: float
    timestamp: datetime
    fraud_score: float = 0.0
    is_flagged: bool = False


class CommunityInfo(BaseModel):
    """A detected community (cluster) within the graph."""

    community_id: int
    member_ids: list[str]
    size: int
    avg_fraud_score: float
    fraud_rate: float
    dominant_node_type: str
    # Ratio of internal edges to maximum possible internal edges
    density: float
    risk_level: str  # "low" | "medium" | "high" | "critical"


class FraudRing(BaseModel):
    """A dense subgraph with an elevated fraud rate."""

    ring_id: str
    member_ids: list[str]
    size: int
    internal_edge_count: int
    fraud_rate: float
    avg_fraud_score: float
    total_fraudulent_amount: float
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float  # 0-1 confidence that this is a genuine fraud ring


class NetworkRiskScore(BaseModel):
    """Aggregated network-level risk indicators for a single node."""

    node_id: str
    node_type: str
    direct_fraud_rate: float
    pagerank_score: float
    degree_centrality: float
    betweenness_centrality: float
    community_fraud_rate: float
    neighbor_avg_fraud_rate: float
    # Composite score combining all signals
    network_risk_score: float
    risk_factors: list[str]


# ── Internal data structures (not exposed via API) ────────────────────────────


class _EdgeRecord:
    """Lightweight in-memory edge, avoids Pydantic overhead for hot path."""

    __slots__ = (
        "edge_id", "source_id", "target_id", "amount",
        "timestamp_ts", "fraud_score", "is_flagged",
    )

    def __init__(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        amount: float,
        timestamp_ts: float,
        fraud_score: float,
        is_flagged: bool,
    ) -> None:
        self.edge_id = edge_id
        self.source_id = source_id
        self.target_id = target_id
        self.amount = amount
        self.timestamp_ts = timestamp_ts
        self.fraud_score = fraud_score
        self.is_flagged = is_flagged


# ── Main service class ────────────────────────────────────────────────────────


class TransactionGraphAnalyzer:
    """In-memory graph analyzer for fraud ring and network risk detection.

    Thread-safety: public methods acquire ``_lock`` (asyncio.Lock) before
    mutating shared state.  Read-only operations on stable snapshots do not
    require the lock.
    """

    # PageRank convergence threshold
    _PAGERANK_EPSILON: float = 1e-6
    _PAGERANK_MAX_ITER: int = 100

    # Label propagation convergence
    _LP_MAX_ITER: int = 30

    def __init__(
        self,
        fraud_score_threshold: float = 0.6,
        max_edges: int = 500_000,
    ) -> None:
        self._fraud_score_threshold = fraud_score_threshold
        self._max_edges = max_edges

        # node_id -> GraphNode
        self._nodes: dict[str, GraphNode] = {}

        # adjacency list: node_id -> set of neighbour node_ids (undirected view)
        self._adj: dict[str, set[str]] = defaultdict(set)

        # directed adjacency for PageRank: node_id -> list of out-neighbour ids
        self._out_adj: dict[str, list[str]] = defaultdict(list)

        # edge list (ordered by insertion for temporal queries)
        self._edges: list[_EdgeRecord] = []

        # node_id -> list[_EdgeRecord] for fast neighbour-edge lookup
        self._node_edges: dict[str, list[_EdgeRecord]] = defaultdict(list)

        # cached results (invalidated on graph mutation)
        self._pagerank_cache: dict[str, float] | None = None
        self._community_cache: list[CommunityInfo] | None = None
        self._betweenness_cache: dict[str, float] | None = None
        self._cache_dirty: bool = True

        self._lock = asyncio.Lock()

        logger.info("graph_analyzer.initialized", max_edges=max_edges)

    # ── Graph construction ────────────────────────────────────────────────────

    async def add_transaction(
        self,
        user_id: str,
        merchant_id: str,
        device_id: str,
        amount: float,
        timestamp: datetime,
        fraud_score: float = 0.0,
    ) -> str:
        """Add a transaction to the graph, updating nodes and edges.

        Returns the generated edge_id for the primary user->merchant edge.
        Creates three directed edges per call:
          user -> merchant, device -> user, device -> merchant
        """
        is_flagged = fraud_score >= self._fraud_score_threshold
        ts = timestamp.timestamp()
        edge_id = f"{user_id}:{merchant_id}:{ts:.3f}"

        async with self._lock:
            if len(self._edges) >= self._max_edges:
                # Evict oldest 10 % to stay within memory budget
                cutoff = int(self._max_edges * 0.1)
                evicted = self._edges[:cutoff]
                self._edges = self._edges[cutoff:]
                for e in evicted:
                    self._node_edges.get(e.source_id, [])  # touched but not pruned
                logger.warning(
                    "graph_analyzer.edge_eviction",
                    evicted=cutoff,
                    remaining=len(self._edges),
                )

            now = datetime.now(timezone.utc)

            # Ensure nodes exist
            for nid, ntype in (
                (user_id, "user"),
                (merchant_id, "merchant"),
                (device_id, "device"),
            ):
                if nid not in self._nodes:
                    self._nodes[nid] = GraphNode(
                        node_id=nid,
                        node_type=ntype,
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )
                    GRAPH_NODES_TOTAL.labels(node_type=ntype).inc()
                node = self._nodes[nid]
                node.transaction_count += 1
                node.total_amount += amount
                node.last_seen = max(node.last_seen, timestamp)
                # Rolling average of fraud score
                node.avg_fraud_score = (
                    node.avg_fraud_score * (node.transaction_count - 1) + fraud_score
                ) / node.transaction_count
                if is_flagged:
                    node.fraud_count += 1

            # Add primary edge: user -> merchant
            primary = _EdgeRecord(
                edge_id=edge_id,
                source_id=user_id,
                target_id=merchant_id,
                amount=amount,
                timestamp_ts=ts,
                fraud_score=fraud_score,
                is_flagged=is_flagged,
            )
            self._edges.append(primary)
            self._node_edges[user_id].append(primary)
            self._node_edges[merchant_id].append(primary)
            self._adj[user_id].add(merchant_id)
            self._adj[merchant_id].add(user_id)
            self._out_adj[user_id].append(merchant_id)

            # Device -> user link (undirected for community, directed for PR)
            dev_user_id = f"{device_id}:{user_id}:{ts:.3f}"
            du = _EdgeRecord(
                edge_id=dev_user_id,
                source_id=device_id,
                target_id=user_id,
                amount=amount,
                timestamp_ts=ts,
                fraud_score=fraud_score,
                is_flagged=is_flagged,
            )
            self._edges.append(du)
            self._node_edges[device_id].append(du)
            self._node_edges[user_id].append(du)
            self._adj[device_id].add(user_id)
            self._adj[user_id].add(device_id)
            self._out_adj[device_id].append(user_id)

            # Device -> merchant link
            dev_merch_id = f"{device_id}:{merchant_id}:{ts:.3f}"
            dm = _EdgeRecord(
                edge_id=dev_merch_id,
                source_id=device_id,
                target_id=merchant_id,
                amount=amount,
                timestamp_ts=ts,
                fraud_score=fraud_score,
                is_flagged=is_flagged,
            )
            self._edges.append(dm)
            self._node_edges[device_id].append(dm)
            self._node_edges[merchant_id].append(dm)
            self._adj[device_id].add(merchant_id)
            self._adj[merchant_id].add(device_id)
            self._out_adj[device_id].append(merchant_id)

            GRAPH_EDGES_TOTAL.set(len(self._edges))
            self._cache_dirty = True

        return edge_id

    # ── Community detection (label propagation) ───────────────────────────────

    async def detect_communities(self) -> list[CommunityInfo]:
        """Run label propagation to find communities.

        Label propagation is O(E) per iteration and converges quickly in
        practice (typically < 20 iterations for social-like graphs).
        """
        if not self._cache_dirty and self._community_cache is not None:
            return self._community_cache

        start = time.perf_counter()
        try:
            communities = await asyncio.get_event_loop().run_in_executor(
                None, self._label_propagation
            )
        except Exception as exc:
            GRAPH_ANALYSIS_ERRORS.labels(operation="detect_communities").inc()
            logger.error("graph_analyzer.community_detection_failed", error=str(exc))
            raise
        finally:
            COMMUNITY_DETECTION_SECONDS.observe(time.perf_counter() - start)

        self._community_cache = communities
        return communities

    def _label_propagation(self) -> list[CommunityInfo]:
        """Pure-Python label propagation community detection."""
        nodes = list(self._nodes.keys())
        if not nodes:
            return []

        # Initialise each node with its own label
        labels: dict[str, str] = {nid: nid for nid in nodes}

        for iteration in range(self._LP_MAX_ITER):
            changed = False
            # Randomise traversal order each iteration to avoid bias
            random.shuffle(nodes)
            for nid in nodes:
                neighbours = list(self._adj.get(nid, set()))
                if not neighbours:
                    continue
                # Count neighbour labels
                label_counts: dict[str, int] = defaultdict(int)
                for nb in neighbours:
                    label_counts[labels[nb]] += 1
                # Pick most frequent; ties broken by lexicographic order
                best = max(label_counts, key=lambda lbl: (label_counts[lbl], lbl))
                if best != labels[nid]:
                    labels[nid] = best
                    changed = True
            if not changed:
                logger.debug(
                    "graph_analyzer.label_propagation_converged",
                    iteration=iteration,
                )
                break

        # Group nodes by label
        groups: dict[str, list[str]] = defaultdict(list)
        for nid, lbl in labels.items():
            groups[lbl].append(nid)

        communities: list[CommunityInfo] = []
        for cid, (lbl, members) in enumerate(groups.items()):
            community = self._build_community_info(cid, members)
            communities.append(community)

        # Sort by descending fraud rate so the riskiest appear first
        communities.sort(key=lambda c: c.fraud_rate, reverse=True)

        # Update HIGH_RISK_NODES gauge
        high_risk = sum(
            1 for n in self._nodes.values() if n.fraud_rate > 0.3
        )
        HIGH_RISK_NODES.set(high_risk)

        return communities

    def _build_community_info(self, cid: int, members: list[str]) -> CommunityInfo:
        member_set = set(members)
        total_fraud_score = 0.0
        fraud_count = 0
        txn_count = 0
        type_counts: dict[str, int] = defaultdict(int)
        internal_edges = 0

        for nid in members:
            node = self._nodes[nid]
            type_counts[node.node_type] += 1
            fraud_count += node.fraud_count
            txn_count += node.transaction_count
            total_fraud_score += node.avg_fraud_score
            # Count internal edges (each edge counted once)
            for nb in self._adj.get(nid, set()):
                if nb in member_set and nb > nid:
                    internal_edges += 1

        n = len(members)
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        density = internal_edges / max_edges if max_edges > 0 else 0.0
        avg_fraud_score = total_fraud_score / n if n else 0.0
        fraud_rate = fraud_count / txn_count if txn_count else 0.0
        dominant = max(type_counts, key=lambda t: type_counts[t])

        if fraud_rate >= 0.5:
            risk_level = "critical"
        elif fraud_rate >= 0.3:
            risk_level = "high"
        elif fraud_rate >= 0.1:
            risk_level = "medium"
        else:
            risk_level = "low"

        return CommunityInfo(
            community_id=cid,
            member_ids=members,
            size=n,
            avg_fraud_score=round(avg_fraud_score, 4),
            fraud_rate=round(fraud_rate, 4),
            dominant_node_type=dominant,
            density=round(density, 4),
            risk_level=risk_level,
        )

    # ── Fraud ring detection ──────────────────────────────────────────────────

    async def detect_fraud_rings(
        self,
        min_fraud_rate: float = 0.3,
        min_size: int = 3,
        min_density: float = 0.4,
    ) -> list[FraudRing]:
        """Identify dense subgraphs whose fraud rate exceeds *min_fraud_rate*.

        Algorithm:
          1. Run community detection to obtain candidate clusters.
          2. Filter candidates by fraud rate, size, and internal density.
          3. For each surviving candidate, compute a confidence score.
        """
        try:
            communities = await self.detect_communities()
        except Exception as exc:
            GRAPH_ANALYSIS_ERRORS.labels(operation="detect_fraud_rings").inc()
            logger.error("graph_analyzer.fraud_ring_detection_failed", error=str(exc))
            raise

        rings: list[FraudRing] = []
        for comm in communities:
            if (
                comm.fraud_rate < min_fraud_rate
                or comm.size < min_size
                or comm.density < min_density
            ):
                continue

            ring = self._build_fraud_ring(comm)
            rings.append(ring)
            FRAUD_RINGS_DETECTED.inc()
            logger.warning(
                "graph_analyzer.fraud_ring_detected",
                ring_id=ring.ring_id,
                size=ring.size,
                fraud_rate=ring.fraud_rate,
                confidence=ring.confidence,
            )

        rings.sort(key=lambda r: r.confidence, reverse=True)
        return rings

    def _build_fraud_ring(self, comm: CommunityInfo) -> FraudRing:
        member_set = set(comm.member_ids)
        internal_edges = 0
        fraudulent_amount = 0.0
        fraud_scores: list[float] = []

        for edge in self._edges:
            if edge.source_id in member_set and edge.target_id in member_set:
                internal_edges += 1
                if edge.is_flagged:
                    fraudulent_amount += edge.amount
                    fraud_scores.append(edge.fraud_score)

        avg_score = sum(fraud_scores) / len(fraud_scores) if fraud_scores else 0.0

        # Confidence: blend of fraud_rate, density, and avg_fraud_score
        confidence = min(
            1.0,
            (comm.fraud_rate * 0.5) + (comm.density * 0.3) + (avg_score * 0.2),
        )

        ring_id = f"ring:{sorted(comm.member_ids)[0]}:{comm.community_id}"
        return FraudRing(
            ring_id=ring_id,
            member_ids=comm.member_ids,
            size=comm.size,
            internal_edge_count=internal_edges,
            fraud_rate=comm.fraud_rate,
            avg_fraud_score=round(avg_score, 4),
            total_fraudulent_amount=round(fraudulent_amount, 2),
            confidence=round(confidence, 4),
        )

    # ── PageRank ──────────────────────────────────────────────────────────────

    async def compute_pagerank(self, damping: float = 0.85) -> dict[str, float]:
        """Compute personalised PageRank using the power-iteration method.

        Returns a mapping of node_id -> PageRank score (values sum to 1).
        Results are cached until the graph is mutated.
        """
        if not self._cache_dirty and self._pagerank_cache is not None:
            return self._pagerank_cache

        start = time.perf_counter()
        try:
            pr = await asyncio.get_event_loop().run_in_executor(
                None, self._compute_pagerank_sync, damping
            )
        except Exception as exc:
            GRAPH_ANALYSIS_ERRORS.labels(operation="compute_pagerank").inc()
            logger.error("graph_analyzer.pagerank_failed", error=str(exc))
            raise
        finally:
            PAGERANK_COMPUTATION_SECONDS.observe(time.perf_counter() - start)

        self._pagerank_cache = pr
        # Only clear dirty flag when all caches are populated
        if self._community_cache is not None:
            self._cache_dirty = False
        return pr

    def _compute_pagerank_sync(self, damping: float) -> dict[str, float]:
        nodes = list(self._nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}

        idx = {nid: i for i, nid in enumerate(nodes)}
        teleport = (1.0 - damping) / n
        pr = [1.0 / n] * n

        # Build out-degree lookup
        out_degree = [len(self._out_adj.get(nid, [])) for nid in nodes]

        for _ in range(self._PAGERANK_MAX_ITER):
            new_pr = [teleport] * n
            for nid in nodes:
                i = idx[nid]
                od = out_degree[i]
                if od == 0:
                    # Dangling node: distribute rank evenly
                    contrib = damping * pr[i] / n
                    for j in range(n):
                        new_pr[j] += contrib
                else:
                    contrib = damping * pr[i] / od
                    for nb in self._out_adj.get(nid, []):
                        new_pr[idx[nb]] += contrib

            # Check convergence (L1 norm)
            delta = sum(abs(new_pr[i] - pr[i]) for i in range(n))
            pr = new_pr
            if delta < self._PAGERANK_EPSILON:
                break

        return {nodes[i]: round(pr[i], 8) for i in range(n)}

    # ── Centrality measures ───────────────────────────────────────────────────

    def _compute_degree_centrality(self) -> dict[str, float]:
        """Normalised degree centrality: degree / (N-1)."""
        n = len(self._nodes)
        if n <= 1:
            return {nid: 0.0 for nid in self._nodes}
        return {
            nid: len(self._adj.get(nid, set())) / (n - 1)
            for nid in self._nodes
        }

    def _compute_betweenness_centrality(self) -> dict[str, float]:
        """Approximate betweenness centrality using Brandes' algorithm (BFS).

        Exact computation is O(VE); for large graphs this runs on a sample
        of source nodes for performance.
        """
        nodes = list(self._nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}

        # Sample up to 200 source nodes for scalability
        sample_size = min(n, 200)
        sources = random.sample(nodes, sample_size)
        scale = n / sample_size  # extrapolation factor

        betweenness: dict[str, float] = defaultdict(float)

        for s in sources:
            # BFS from s
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

            # Accumulate dependencies
            delta: dict[str, float] = defaultdict(float)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        # Normalise and apply sample scaling
        norm = (n - 1) * (n - 2) if n > 2 else 1.0
        return {
            nid: round(betweenness.get(nid, 0.0) * scale / norm, 8)
            for nid in nodes
        }

    # ── Bipartite analysis ────────────────────────────────────────────────────

    def get_bipartite_stats(self) -> dict[str, Any]:
        """Return statistics for the bipartite user-merchant subgraph."""
        users = [n for n in self._nodes.values() if n.node_type == "user"]
        merchants = [n for n in self._nodes.values() if n.node_type == "merchant"]

        user_ids = {n.node_id for n in users}
        merchant_ids = {n.node_id for n in merchants}

        bip_edges = [
            e for e in self._edges
            if (e.source_id in user_ids and e.target_id in merchant_ids)
            or (e.source_id in merchant_ids and e.target_id in user_ids)
        ]

        shared_merchants: dict[str, set[str]] = defaultdict(set)
        for e in bip_edges:
            uid = e.source_id if e.source_id in user_ids else e.target_id
            mid = e.source_id if e.source_id in merchant_ids else e.target_id
            shared_merchants[mid].add(uid)

        # Merchants shared by many users are high-risk hubs
        high_fan_in = {
            mid: len(uids)
            for mid, uids in shared_merchants.items()
            if len(uids) >= 5
        }

        return {
            "user_count": len(users),
            "merchant_count": len(merchants),
            "bipartite_edge_count": len(bip_edges),
            "avg_users_per_merchant": (
                sum(len(u) for u in shared_merchants.values()) / len(shared_merchants)
                if shared_merchants else 0.0
            ),
            "high_fan_in_merchants": high_fan_in,
        }

    # ── Temporal analysis ─────────────────────────────────────────────────────

    def get_temporal_stats(
        self,
        node_id: str,
        window_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        """Return velocity and burst metrics for a node within the time window."""
        node_edge_list = self._node_edges.get(node_id, [])
        if not node_edge_list:
            return {"node_id": node_id, "edge_count": 0, "burst_detected": False}

        now_ts = time.time()
        cutoff = now_ts - window_seconds
        recent = [e for e in node_edge_list if e.timestamp_ts >= cutoff]

        if not recent:
            return {
                "node_id": node_id,
                "edge_count": 0,
                "burst_detected": False,
                "window_seconds": window_seconds,
            }

        amounts = [e.amount for e in recent]
        fraud_scores = [e.fraud_score for e in recent]

        # Burst: more than 10 edges within any 60-second sub-window
        recent_sorted = sorted(recent, key=lambda e: e.timestamp_ts)
        burst_detected = False
        for i, edge in enumerate(recent_sorted):
            sub_window_end = edge.timestamp_ts + 60.0
            count_in_minute = sum(
                1 for e2 in recent_sorted[i:]
                if e2.timestamp_ts <= sub_window_end
            )
            if count_in_minute > 10:
                burst_detected = True
                break

        return {
            "node_id": node_id,
            "edge_count": len(recent),
            "total_amount": round(sum(amounts), 2),
            "avg_amount": round(sum(amounts) / len(amounts), 2),
            "avg_fraud_score": round(sum(fraud_scores) / len(fraud_scores), 4),
            "burst_detected": burst_detected,
            "window_seconds": window_seconds,
        }

    # ── Node risk scoring ─────────────────────────────────────────────────────

    async def get_node_risk(self, node_id: str) -> NetworkRiskScore:
        """Compute a composite network risk score for a single node.

        Combines PageRank, degree centrality, betweenness centrality,
        community-level fraud rate, and neighbour fraud rates.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not found in graph")

        node = self._nodes[node_id]

        # Ensure caches are warm
        pr_map = await self.compute_pagerank()
        communities = await self.detect_communities()

        # Betweenness is expensive; use cached value when available
        if self._betweenness_cache is None:
            self._betweenness_cache = await asyncio.get_event_loop().run_in_executor(
                None, self._compute_betweenness_centrality
            )
        betweenness_map = self._betweenness_cache
        degree_map = self._compute_degree_centrality()

        # Find this node's community
        community_fraud_rate = 0.0
        for comm in communities:
            if node_id in comm.member_ids:
                community_fraud_rate = comm.fraud_rate
                break

        # Neighbour average fraud rate
        neighbours = list(self._adj.get(node_id, set()))
        if neighbours:
            nb_fraud_rates = [
                self._nodes[nb].fraud_rate
                for nb in neighbours
                if nb in self._nodes
            ]
            neighbor_avg = sum(nb_fraud_rates) / len(nb_fraud_rates) if nb_fraud_rates else 0.0
        else:
            neighbor_avg = 0.0

        pagerank = pr_map.get(node_id, 0.0)
        degree_c = degree_map.get(node_id, 0.0)
        betweenness_c = betweenness_map.get(node_id, 0.0)
        direct_fraud = node.fraud_rate

        # Composite: weighted blend
        network_risk = min(
            1.0,
            (direct_fraud * 0.35)
            + (community_fraud_rate * 0.25)
            + (neighbor_avg * 0.20)
            + (min(pagerank * 500, 1.0) * 0.10)  # scale PR from ~0.002 to [0,1]
            + (min(betweenness_c * 10, 1.0) * 0.10),
        )

        risk_factors: list[str] = []
        if direct_fraud > 0.3:
            risk_factors.append("high_direct_fraud_rate")
        if community_fraud_rate > 0.3:
            risk_factors.append("high_community_fraud_rate")
        if neighbor_avg > 0.25:
            risk_factors.append("high_neighbor_fraud_rate")
        if degree_c > 0.1:
            risk_factors.append("high_degree_centrality")
        if betweenness_c > 0.05:
            risk_factors.append("high_betweenness_centrality")
        if pagerank > 0.01:
            risk_factors.append("high_pagerank")

        return NetworkRiskScore(
            node_id=node_id,
            node_type=node.node_type,
            direct_fraud_rate=round(direct_fraud, 4),
            pagerank_score=round(pagerank, 8),
            degree_centrality=round(degree_c, 6),
            betweenness_centrality=round(betweenness_c, 8),
            community_fraud_rate=round(community_fraud_rate, 4),
            neighbor_avg_fraud_rate=round(neighbor_avg, 4),
            network_risk_score=round(network_risk, 4),
            risk_factors=risk_factors,
        )

    async def get_user_network_score(self, user_id: str) -> float:
        """Return the composite network risk score for a user node (0-1)."""
        try:
            risk = await self.get_node_risk(user_id)
            return risk.network_risk_score
        except KeyError:
            logger.debug("graph_analyzer.user_not_in_graph", user_id=user_id)
            return 0.0
        except Exception as exc:
            GRAPH_ANALYSIS_ERRORS.labels(operation="get_user_network_score").inc()
            logger.error(
                "graph_analyzer.user_score_failed",
                user_id=user_id,
                error=str(exc),
            )
            return 0.0

    # ── Graph statistics ──────────────────────────────────────────────────────

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return a summary of the current graph state."""
        node_type_counts: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            node_type_counts[node.node_type] += 1

        total_edges = len(self._edges)
        total_nodes = len(self._nodes)

        flagged_edges = sum(1 for e in self._edges if e.is_flagged)
        avg_degree = (
            sum(len(nbrs) for nbrs in self._adj.values()) / total_nodes
            if total_nodes else 0.0
        )

        high_risk_nodes = sum(
            1 for n in self._nodes.values() if n.fraud_rate > 0.3
        )
        fraud_rate_overall = (
            sum(n.fraud_count for n in self._nodes.values())
            / max(sum(n.transaction_count for n in self._nodes.values()), 1)
        )

        communities = self._community_cache
        community_count = len(communities) if communities else None

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "flagged_edges": flagged_edges,
            "flagged_edge_ratio": round(flagged_edges / total_edges, 4) if total_edges else 0.0,
            "node_type_counts": dict(node_type_counts),
            "avg_degree": round(avg_degree, 2),
            "high_risk_nodes": high_risk_nodes,
            "overall_fraud_rate": round(fraud_rate_overall, 4),
            "community_count": community_count,
            "cache_dirty": self._cache_dirty,
            "max_edge_capacity": self._max_edges,
        }

    # ── Lifecycle helpers ─────────────────────────────────────────────────────

    async def clear(self) -> None:
        """Reset the graph to an empty state (useful for testing)."""
        async with self._lock:
            self._nodes.clear()
            self._adj.clear()
            self._out_adj.clear()
            self._edges.clear()
            self._node_edges.clear()
            self._pagerank_cache = None
            self._community_cache = None
            self._betweenness_cache = None
            self._cache_dirty = True
        logger.info("graph_analyzer.cleared")
