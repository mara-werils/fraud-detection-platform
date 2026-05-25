"""Session linking and velocity analysis service.

Tracks user sessions end-to-end, builds per-session risk profiles, and
runs multi-dimensional velocity checks across users, IPs, and devices.

Key capabilities:
  - Per-session event ingestion and profile building
  - Velocity rules: transactions/hour, unique merchants/hour, amount/day,
    concurrent sessions, and failed attempts per IP
  - Anomaly detection within and across sessions (rapid country switch,
    impossible travel, sudden amount spike, concurrent session abuse)
  - Session clustering / linking across shared device or IP signals
  - Prometheus metrics for all hot-path operations
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────

SESSION_EVENTS_TOTAL = Counter(
    "session_events_total",
    "Total session events ingested",
    ["event_type"],
)

SESSION_RISK_SCORE_HISTOGRAM = Histogram(
    "session_risk_score",
    "Distribution of session risk scores at analysis time",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

VELOCITY_CHECKS_TOTAL = Counter(
    "session_velocity_checks_total",
    "Total velocity threshold evaluations",
    ["entity_type", "exceeded"],
)

ACTIVE_SESSIONS_GAUGE = Gauge(
    "session_active_sessions_total",
    "Number of currently active sessions (no logout event received)",
)

SESSION_ANOMALIES_TOTAL = Counter(
    "session_anomalies_total",
    "Anomalies detected during session analysis",
    ["anomaly_type"],
)

SESSION_CLUSTERS_TOTAL = Counter(
    "session_clusters_created_total",
    "Total session clusters created by link_sessions",
)

SESSION_ANALYSIS_LATENCY = Histogram(
    "session_analysis_latency_seconds",
    "Time spent in analyze_session_risk",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)


# ── Enumerations ───────────────────────────────────────────────────────────────


class EventType(StrEnum):
    LOGIN = "login"
    TRANSACTION = "transaction"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PROFILE_UPDATE = "profile_update"


# ── Pydantic models ────────────────────────────────────────────────────────────


class SessionEvent(BaseModel):
    """A single event recorded within a user session."""

    session_id: str = Field(..., description="Stable session identifier")
    user_id: str
    device_id: str
    ip_address: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionProfile(BaseModel):
    """Aggregated profile for a single session."""

    session_id: str
    user_id: str
    device_id: str
    ip_address: str
    start_time: datetime
    last_activity: datetime
    duration_seconds: float = 0.0
    event_count: int = 0
    transaction_count: int = 0
    total_amount: float = 0.0
    risk_score: float = 0.0
    risk_factors: list[str] = Field(default_factory=list)


class VelocityCheck(BaseModel):
    """Result of a single velocity threshold evaluation."""

    entity_type: str  # "user" | "ip" | "device"
    entity_value: str
    window_seconds: int
    count: int
    amount_sum: float
    unique_counterparties: int
    exceeds_threshold: bool
    threshold_value: float


class VelocityCheckConfig(BaseModel):
    """Configuration for a single velocity check request."""

    entity_type: str
    entity_value: str
    window_seconds: int
    threshold: float
    check_amount: bool = False
    check_unique_counterparties: bool = False


class SessionCluster(BaseModel):
    """A group of sessions linked by shared signals (device, IP, or user)."""

    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_ids: list[str]
    link_reason: str  # e.g. "shared_device" | "shared_ip" | "same_user"
    risk_score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Internal lightweight event record ─────────────────────────────────────────


class _EventRecord:
    """Lightweight in-memory event; avoids Pydantic overhead on the hot path."""

    __slots__ = (
        "session_id", "user_id", "device_id", "ip_address",
        "event_type", "timestamp_ts", "amount", "merchant_id",
    )

    def __init__(
        self,
        session_id: str,
        user_id: str,
        device_id: str,
        ip_address: str,
        event_type: str,
        timestamp_ts: float,
        amount: float,
        merchant_id: str,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.device_id = device_id
        self.ip_address = ip_address
        self.event_type = event_type
        self.timestamp_ts = timestamp_ts
        self.amount = amount
        self.merchant_id = merchant_id


# ── Built-in velocity rule definitions ────────────────────────────────────────

# (entity_type, window_seconds, threshold, label)
_BUILTIN_RULES: list[tuple[str, int, float, str]] = [
    # Max transactions per user per hour
    ("user_txn_count", 3_600, 20.0, "max_txn_per_user_per_hour"),
    # Max transactions per user per day
    ("user_txn_count", 86_400, 100.0, "max_txn_per_user_per_day"),
    # Max unique merchants per user per hour
    ("user_merchant_unique", 3_600, 10.0, "max_merchants_per_user_per_hour"),
    # Max amount per user per day
    ("user_amount", 86_400, 50_000.0, "max_amount_per_user_per_day"),
    # Max concurrent sessions per user  (checked differently; window is irrelevant)
    ("user_sessions_concurrent", 0, 5.0, "max_concurrent_sessions_per_user"),
    # Max failed login attempts per IP per hour
    ("ip_failed_login", 3_600, 10.0, "max_failed_logins_per_ip_per_hour"),
]


# ── SessionAnalyzer ────────────────────────────────────────────────────────────


class SessionAnalyzer:
    """In-memory session linking and velocity analysis engine.

    All mutable state is protected by ``_lock`` (asyncio.Lock) to allow safe
    concurrent access from multiple coroutines.

    Args:
        max_events_per_session: Ring-buffer cap for events per session.
        max_global_events: Global hard cap; oldest events evicted when reached.
        session_idle_timeout_seconds: Sessions with no activity beyond this are
            considered closed for concurrent-session counting purposes.
    """

    def __init__(
        self,
        max_events_per_session: int = 500,
        max_global_events: int = 1_000_000,
        session_idle_timeout_seconds: int = 1_800,
    ) -> None:
        self._max_events_per_session = max_events_per_session
        self._max_global_events = max_global_events
        self._idle_timeout = session_idle_timeout_seconds

        # session_id -> SessionProfile
        self._sessions: dict[str, SessionProfile] = {}

        # session_id -> deque[_EventRecord]  (capped ring buffer)
        self._session_events: dict[str, deque[_EventRecord]] = defaultdict(
            lambda: deque(maxlen=max_events_per_session)
        )

        # user_id -> list[session_id]
        self._user_sessions: dict[str, list[str]] = defaultdict(list)

        # device_id -> set[session_id]
        self._device_sessions: dict[str, set[str]] = defaultdict(set)

        # ip_address -> set[session_id]
        self._ip_sessions: dict[str, set[str]] = defaultdict(set)

        # Global ordered event log for velocity window queries
        # Stored as _EventRecord in arrival order
        self._global_events: deque[_EventRecord] = deque()

        # Velocity counters — velocity stat totals (all-time)
        self._velocity_stats: dict[str, int] = defaultdict(int)

        self._lock = asyncio.Lock()
        logger.info(
            "session_analyzer.initialized",
            max_events_per_session=max_events_per_session,
            max_global_events=max_global_events,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def record_event(self, event: SessionEvent) -> None:
        """Ingest a SessionEvent and update the relevant session profile.

        Creates the session profile on first event.  Updates last_activity,
        event counts, and running totals.  Thread-safe.
        """
        amount: float = float(event.metadata.get("amount", 0.0))
        merchant_id: str = str(event.metadata.get("merchant_id", ""))
        ts = event.timestamp.timestamp()

        rec = _EventRecord(
            session_id=event.session_id,
            user_id=event.user_id,
            device_id=event.device_id,
            ip_address=event.ip_address,
            event_type=event.event_type,
            timestamp_ts=ts,
            amount=amount,
            merchant_id=merchant_id,
        )

        async with self._lock:
            # Evict oldest global events if cap reached
            if len(self._global_events) >= self._max_global_events:
                evict_count = self._max_global_events // 10
                for _ in range(evict_count):
                    if self._global_events:
                        self._global_events.popleft()
                logger.warning(
                    "session_analyzer.global_event_eviction",
                    evicted=evict_count,
                )

            self._global_events.append(rec)
            self._session_events[event.session_id].append(rec)

            # Create or update session profile
            if event.session_id not in self._sessions:
                profile = SessionProfile(
                    session_id=event.session_id,
                    user_id=event.user_id,
                    device_id=event.device_id,
                    ip_address=event.ip_address,
                    start_time=event.timestamp,
                    last_activity=event.timestamp,
                )
                self._sessions[event.session_id] = profile
                self._user_sessions[event.user_id].append(event.session_id)
                self._device_sessions[event.device_id].add(event.session_id)
                self._ip_sessions[event.ip_address].add(event.session_id)
                ACTIVE_SESSIONS_GAUGE.inc()
            else:
                profile = self._sessions[event.session_id]

            # Mutate profile in-place
            profile.last_activity = event.timestamp
            profile.duration_seconds = ts - profile.start_time.timestamp()
            profile.event_count += 1

            if event.event_type == EventType.TRANSACTION:
                profile.transaction_count += 1
                profile.total_amount += amount

            if event.event_type == EventType.LOGOUT:
                ACTIVE_SESSIONS_GAUGE.dec()

        SESSION_EVENTS_TOTAL.labels(event_type=event.event_type.value).inc()
        logger.debug(
            "session_analyzer.event_recorded",
            session_id=event.session_id,
            event_type=event.event_type,
            user_id=event.user_id,
        )

    async def get_session(self, session_id: str) -> SessionProfile | None:
        """Return the SessionProfile for *session_id*, or None if not found."""
        return self._sessions.get(session_id)

    async def get_user_sessions(
        self, user_id: str, limit: int = 10
    ) -> list[SessionProfile]:
        """Return up to *limit* most-recent sessions for *user_id*."""
        session_ids = self._user_sessions.get(user_id, [])
        profiles: list[SessionProfile] = []
        for sid in reversed(session_ids):
            profile = self._sessions.get(sid)
            if profile is not None:
                profiles.append(profile)
            if len(profiles) >= limit:
                break
        return profiles

    async def analyze_session_risk(self, session_id: str) -> float:
        """Compute a composite risk score (0.0–1.0) for *session_id*.

        Combines transaction velocity, session duration, amount concentration,
        and anomaly signals into a single score.  Updates the cached
        ``risk_score`` on the SessionProfile.

        Returns:
            Risk score between 0.0 (no risk) and 1.0 (maximum risk).
        """
        start = time.perf_counter()
        profile = self._sessions.get(session_id)
        if profile is None:
            logger.warning("session_analyzer.analyze_unknown_session", session_id=session_id)
            return 0.0

        score = 0.0
        factors: list[str] = []

        # 1. High transaction frequency within session
        if profile.transaction_count > 15:
            score += 0.25
            factors.append("high_intra_session_transaction_count")
        elif profile.transaction_count > 8:
            score += 0.10

        # 2. Large total amount
        if profile.total_amount > 20_000:
            score += 0.25
            factors.append("high_intra_session_amount")
        elif profile.total_amount > 5_000:
            score += 0.10

        # 3. Very short session with transactions (smash-and-grab pattern)
        if profile.transaction_count > 0 and profile.duration_seconds < 30:
            score += 0.20
            factors.append("rapid_transactions_short_session")

        # 4. Velocity breaches (user-level, 1-hour window)
        txn_check = await self.check_velocity(
            entity_type="user",
            entity_value=profile.user_id,
            window_seconds=3_600,
            threshold=20.0,
        )
        if txn_check.exceeds_threshold:
            score += 0.20
            factors.append("velocity_txn_count_1h_exceeded")

        amount_check = await self.check_velocity(
            entity_type="user",
            entity_value=profile.user_id,
            window_seconds=86_400,
            threshold=50_000.0,
        )
        if amount_check.exceeds_threshold:
            score += 0.15
            factors.append("velocity_amount_24h_exceeded")

        # 5. Concurrent session abuse
        concurrent = await self.detect_concurrent_sessions(profile.user_id)
        if len(concurrent) > 3:
            score += 0.20
            factors.append("excessive_concurrent_sessions")
        elif len(concurrent) > 1:
            score += 0.10
            factors.append("multiple_concurrent_sessions")

        # 6. Anomaly signals
        anomalies = await self.detect_session_anomalies(session_id)
        score += min(len(anomalies) * 0.05, 0.20)
        factors.extend(anomalies)

        risk_score = round(min(score, 1.0), 4)

        async with self._lock:
            profile.risk_score = risk_score
            profile.risk_factors = factors

        SESSION_RISK_SCORE_HISTOGRAM.observe(risk_score)
        SESSION_ANALYSIS_LATENCY.observe(time.perf_counter() - start)
        logger.info(
            "session_analyzer.risk_analyzed",
            session_id=session_id,
            risk_score=risk_score,
            factor_count=len(factors),
        )
        return risk_score

    async def check_velocity(
        self,
        entity_type: str,
        entity_value: str,
        window_seconds: int,
        threshold: float,
    ) -> VelocityCheck:
        """Evaluate a single velocity threshold for an entity.

        Scans the global event log within *window_seconds* and aggregates
        count, amount_sum, and unique counterparties for the given entity.

        Args:
            entity_type: One of ``"user"``, ``"ip"``, or ``"device"``.
            entity_value: The specific user_id, ip_address, or device_id.
            window_seconds: Look-back window in seconds.
            threshold: The threshold count (or amount, depending on context)
                above which ``exceeds_threshold`` is set to True.

        Returns:
            A fully populated VelocityCheck result.
        """
        cutoff = time.time() - window_seconds
        count = 0
        amount_sum = 0.0
        counterparties: set[str] = set()

        for rec in self._global_events:
            if rec.timestamp_ts < cutoff:
                continue
            match = (
                (entity_type == "user" and rec.user_id == entity_value)
                or (entity_type == "ip" and rec.ip_address == entity_value)
                or (entity_type == "device" and rec.device_id == entity_value)
            )
            if not match:
                continue
            if rec.event_type == EventType.TRANSACTION:
                count += 1
                amount_sum += rec.amount
                if rec.merchant_id:
                    counterparties.add(rec.merchant_id)

        exceeds = count > threshold
        key = f"{entity_type}_velocity"
        self._velocity_stats[key] += 1
        if exceeds:
            self._velocity_stats[f"{key}_exceeded"] += 1

        VELOCITY_CHECKS_TOTAL.labels(
            entity_type=entity_type,
            exceeded=str(exceeds).lower(),
        ).inc()

        return VelocityCheck(
            entity_type=entity_type,
            entity_value=entity_value,
            window_seconds=window_seconds,
            count=count,
            amount_sum=round(amount_sum, 2),
            unique_counterparties=len(counterparties),
            exceeds_threshold=exceeds,
            threshold_value=threshold,
        )

    async def check_multi_velocity(
        self, checks: list[VelocityCheckConfig]
    ) -> list[VelocityCheck]:
        """Evaluate multiple velocity rules concurrently.

        Args:
            checks: List of VelocityCheckConfig descriptors.

        Returns:
            Ordered list of VelocityCheck results matching *checks*.
        """
        tasks = [
            self.check_velocity(
                entity_type=cfg.entity_type,
                entity_value=cfg.entity_value,
                window_seconds=cfg.window_seconds,
                threshold=cfg.threshold,
            )
            for cfg in checks
        ]
        return list(await asyncio.gather(*tasks))

    async def detect_session_anomalies(self, session_id: str) -> list[str]:
        """Return a list of anomaly labels detected within *session_id*.

        Checks for:
        - IP change mid-session
        - Device change mid-session
        - Unusual event ordering (e.g. transaction before login)
        - Sudden amount spike relative to session average
        - Excessive password-change or profile-update events
        """
        events = list(self._session_events.get(session_id, []))
        if not events:
            return []

        anomalies: list[str] = []
        seen_ips: set[str] = set()
        seen_devices: set[str] = set()
        amounts: list[float] = []
        has_login = False
        password_changes = 0
        profile_updates = 0

        for rec in events:
            seen_ips.add(rec.ip_address)
            seen_devices.add(rec.device_id)

            if rec.event_type == EventType.LOGIN:
                has_login = True
            if rec.event_type == EventType.TRANSACTION:
                if not has_login:
                    anomalies.append("transaction_before_login")
                if rec.amount > 0:
                    amounts.append(rec.amount)
            if rec.event_type == EventType.PASSWORD_CHANGE:
                password_changes += 1
            if rec.event_type == EventType.PROFILE_UPDATE:
                profile_updates += 1

        # Mid-session IP hop
        if len(seen_ips) > 1:
            anomalies.append("mid_session_ip_change")

        # Mid-session device hop
        if len(seen_devices) > 1:
            anomalies.append("mid_session_device_change")

        # Sudden amount spike: last transaction is > 3x rolling average
        if len(amounts) >= 3:
            avg_prior = sum(amounts[:-1]) / (len(amounts) - 1)
            if avg_prior > 0 and amounts[-1] > avg_prior * 3:
                anomalies.append("sudden_amount_spike")

        # Repeated sensitive actions
        if password_changes > 1:
            anomalies.append("multiple_password_changes")
        if profile_updates > 3:
            anomalies.append("excessive_profile_updates")

        for anomaly in anomalies:
            SESSION_ANOMALIES_TOTAL.labels(anomaly_type=anomaly).inc()

        return anomalies

    async def detect_concurrent_sessions(self, user_id: str) -> list[SessionProfile]:
        """Return sessions for *user_id* that are currently active.

        A session is considered active when its last_activity is within
        ``session_idle_timeout_seconds`` of now and no logout event was seen.
        """
        now = time.time()
        active: list[SessionProfile] = []
        for sid in self._user_sessions.get(user_id, []):
            profile = self._sessions.get(sid)
            if profile is None:
                continue
            age = now - profile.last_activity.timestamp()
            if age <= self._idle_timeout:
                # Check there is no logout event in this session
                has_logout = any(
                    r.event_type == EventType.LOGOUT
                    for r in self._session_events.get(sid, [])
                )
                if not has_logout:
                    active.append(profile)
        return active

    async def link_sessions(self, session_ids: list[str]) -> SessionCluster:
        """Create a SessionCluster grouping the given sessions.

        Determines the dominant link reason by inspecting shared signals
        (device_id, ip_address, or user_id) across the provided sessions.

        Args:
            session_ids: List of session IDs to cluster together.

        Returns:
            A SessionCluster with a computed aggregate risk_score.
        """
        profiles = [
            self._sessions[sid]
            for sid in session_ids
            if sid in self._sessions
        ]

        if not profiles:
            return SessionCluster(session_ids=session_ids, link_reason="unknown")

        devices = {p.device_id for p in profiles}
        ips = {p.ip_address for p in profiles}
        users = {p.user_id for p in profiles}

        if len(devices) == 1:
            link_reason = "shared_device"
        elif len(ips) == 1:
            link_reason = "shared_ip"
        elif len(users) == 1:
            link_reason = "same_user"
        else:
            link_reason = "mixed_signals"

        # Aggregate risk: mean of per-session scores, boosted for multi-user
        avg_risk = sum(p.risk_score for p in profiles) / len(profiles)
        if len(users) > 1:
            avg_risk = min(avg_risk + 0.15, 1.0)

        cluster = SessionCluster(
            session_ids=[p.session_id for p in profiles],
            link_reason=link_reason,
            risk_score=round(avg_risk, 4),
        )

        SESSION_CLUSTERS_TOTAL.inc()
        logger.info(
            "session_analyzer.sessions_linked",
            cluster_id=cluster.cluster_id,
            session_count=len(profiles),
            link_reason=link_reason,
            risk_score=cluster.risk_score,
        )
        return cluster

    async def get_velocity_stats(self) -> dict[str, Any]:
        """Return all-time velocity check counters and session summary stats.

        Useful for admin dashboards and SLO monitoring.
        """
        active_count = 0
        now = time.time()
        for sid, profile in self._sessions.items():
            age = now - profile.last_activity.timestamp()
            has_logout = any(
                r.event_type == EventType.LOGOUT
                for r in self._session_events.get(sid, [])
            )
            if age <= self._idle_timeout and not has_logout:
                active_count += 1

        return {
            "total_sessions": len(self._sessions),
            "active_sessions_estimate": active_count,
            "total_global_events": len(self._global_events),
            "unique_users": len(self._user_sessions),
            "unique_devices": len(self._device_sessions),
            "unique_ips": len(self._ip_sessions),
            "velocity_checks_performed": dict(self._velocity_stats),
            "builtin_rules": [
                {
                    "entity_type": r[0],
                    "window_seconds": r[1],
                    "threshold": r[2],
                    "label": r[3],
                }
                for r in _BUILTIN_RULES
            ],
        }

    async def run_builtin_velocity_checks(
        self,
        user_id: str,
        ip_address: str,
    ) -> list[VelocityCheck]:
        """Run all built-in velocity rules for the given user and IP.

        Returns the full list of VelocityCheck results.  Callers can inspect
        ``exceeds_threshold`` to decide on actions.
        """
        configs: list[VelocityCheckConfig] = []

        for entity_type, window_seconds, threshold, _ in _BUILTIN_RULES:
            # Concurrent sessions use a different check path; skip here
            if entity_type == "user_sessions_concurrent":
                continue

            if entity_type.startswith("ip_"):
                entity_value = ip_address
                etype = "ip"
            else:
                entity_value = user_id
                etype = "user"

            configs.append(
                VelocityCheckConfig(
                    entity_type=etype,
                    entity_value=entity_value,
                    window_seconds=window_seconds if window_seconds > 0 else 3_600,
                    threshold=threshold,
                )
            )

        results = await self.check_multi_velocity(configs)

        # Concurrent session check (separate code path)
        concurrent = await self.detect_concurrent_sessions(user_id)
        concurrent_exceeds = len(concurrent) > 5
        results.append(
            VelocityCheck(
                entity_type="user",
                entity_value=user_id,
                window_seconds=0,
                count=len(concurrent),
                amount_sum=0.0,
                unique_counterparties=0,
                exceeds_threshold=concurrent_exceeds,
                threshold_value=5.0,
            )
        )

        exceeded_count = sum(1 for r in results if r.exceeds_threshold)
        logger.info(
            "session_analyzer.builtin_velocity_checked",
            user_id=user_id,
            ip_address=ip_address,
            rules_checked=len(results),
            rules_exceeded=exceeded_count,
        )
        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _events_in_window(
        self, entity_type: str, entity_value: str, cutoff_ts: float
    ) -> list[_EventRecord]:
        """Return all global events for an entity at or after *cutoff_ts*."""
        result: list[_EventRecord] = []
        for rec in self._global_events:
            if rec.timestamp_ts < cutoff_ts:
                continue
            match = (
                (entity_type == "user" and rec.user_id == entity_value)
                or (entity_type == "ip" and rec.ip_address == entity_value)
                or (entity_type == "device" and rec.device_id == entity_value)
            )
            if match:
                result.append(rec)
        return result
