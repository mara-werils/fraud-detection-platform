# ADR-003: GNN for Graph-Based Fraud Detection

## Status

Accepted

## Date

2026-05-01

## Context

The XGBoost model scores each transaction independently using tabular features. While effective for most fraud patterns (card testing, velocity abuse, geo anomalies), it cannot detect **coordinated fraud** involving multiple accounts and merchants.

Real-world fraud rings operate by:
- Creating multiple accounts and transacting between them
- Using shared devices, IPs, or phone numbers across accounts
- Routing money through complicit merchants in circular patterns
- Gradually building trust on each account before executing high-value fraud

These patterns are inherently **relational** — the fraud signal lives in the graph structure, not in individual transaction features.

## Decision

We chose a **Graph Neural Network (GraphSAGE)** as a complementary model for detecting graph-based fraud patterns.

## Alternatives Considered

### 1. Rule-Based Graph Analysis

Define explicit rules like "flag users sharing > 2 devices" or "detect cycles of length 3."

- **Pro**: Interpretable, fast, no training needed
- **Con**: Brittle, requires constant manual tuning, misses novel patterns
- **Verdict**: Too rigid for evolving fraud patterns

### 2. Traditional Graph Algorithms (PageRank, Community Detection)

Use established graph algorithms to compute risk scores.

- **Pro**: Well-understood, deterministic, no training data needed
- **Con**: Not differentiable, cannot incorporate node/edge features, limited expressiveness
- **Verdict**: Good as features for XGBoost but insufficient as a standalone model

### 3. Graph Neural Network (GNN)

Learn node representations from graph structure and features.

- **Pro**: Learns patterns automatically, incorporates features, generalizes to unseen fraud rings
- **Con**: Higher latency, requires graph construction, more complex deployment
- **Verdict**: Best balance of expressiveness and automation

## Rationale

### Why GraphSAGE Specifically

| GNN Variant | Pros | Cons | Fit |
|------------|------|------|-----|
| **GCN** | Simple, effective | Full-graph computation, no inductive learning | Poor — cannot score new nodes without retraining |
| **GAT** | Attention-weighted neighbors | Higher compute cost | Medium — attention is useful but slower |
| **GraphSAGE** | Inductive, samples neighbors, scalable | Sampling introduces variance | Best — handles dynamic graphs with new users/merchants |

GraphSAGE was chosen because:

1. **Inductive learning**: Can generate embeddings for nodes not seen during training (new users, new merchants). This is critical — the fraud detection system processes new users continuously.

2. **Neighbor sampling**: Instead of computing over the entire graph, GraphSAGE samples a fixed number of neighbors per layer. This bounds computation cost and enables consistent latency.

3. **Scalability**: With 2 layers and mean aggregation, inference scales linearly with the number of sampled neighbors, not the total graph size.

### Graph Construction

```
Nodes:
  - Users (features: profile attributes, behavioral metrics)
  - Merchants (features: category, fraud rate, transaction volume)

Edges:
  - User → Merchant (transaction)
  - User → User (shared device, shared IP — inferred)

Edge Features:
  - Transaction amount
  - Transaction frequency
  - Time pattern similarity
```

### Fraud Patterns Detected by GNN

| Pattern | XGBoost | GNN |
|---------|---------|-----|
| High-amount single transaction | Yes | Partial |
| Velocity abuse (single user) | Yes | Partial |
| Geo anomaly | Yes | No |
| New device + high amount | Yes | Partial |
| **Fraud ring (multiple accounts)** | No | **Yes** |
| **Shared device cluster** | No | **Yes** |
| **Circular money flow** | No | **Yes** |
| **Merchant collusion** | No | **Yes** |

## Consequences

### Positive
- Detects coordinated fraud that tabular models miss entirely
- Learns representations automatically — no manual feature engineering for graph patterns
- Inductive — handles new users and merchants without retraining
- Embedding vectors can be cached in Redis for fast lookup
- Complements XGBoost in the ensemble (different error profiles)

### Negative
- Higher latency than XGBoost (< 50 ms vs. < 10 ms)
- Requires graph construction and maintenance infrastructure
- Batch inference (every 5 minutes) introduces staleness
- More complex training pipeline (graph sampling, mini-batching)
- Harder to interpret than tree-based models (addressed by LLM explainer)

### Mitigations
- Cache GNN embeddings in Redis (TTL: 6 hours) to avoid real-time graph computation
- Run batch inference every 5 minutes — acceptable for fraud ring detection (not latency-critical)
- Ensemble weighting (30% GNN) limits impact of stale embeddings
- Fallback: if embeddings are unavailable, ensemble uses XGBoost only
- Use LLM explainer to generate human-readable explanations for GNN-driven decisions
