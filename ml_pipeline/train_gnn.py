"""GraphSAGE GNN training for fraud detection (PyTorch Geometric).

Builds a heterogeneous transaction graph with User and Merchant nodes,
trains a GraphSAGE model for edge-level fraud classification, and
exports learned embeddings.

CLI::

    python -m ml_pipeline.train_gnn --data data/training.parquet --output models/gnn_v1.pt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for optional torch dependencies
# ---------------------------------------------------------------------------


def _check_torch() -> None:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for GNN training. "
            "Install with: pip install 'fraud-detection-platform[gnn]'"
        ) from exc


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_transaction_graph(
    df: pd.DataFrame,
) -> Any:
    """Build a PyTorch Geometric Data object from transaction data.

    Nodes:
        - User nodes: indexed by user_id
        - Merchant nodes: indexed by merchant_id

    Edges:
        - Each transaction creates a bidirectional edge between user and
          merchant, with edge features (amount, time_delta, is_fraud).

    Returns:
        ``torch_geometric.data.Data`` with node features, edge_index,
        edge_attr, and labels.
    """
    import torch
    from torch_geometric.data import Data

    # --- Build node indices ---
    user_ids = df["user_id"].unique()
    merchant_ids = df["merchant_id"].unique()

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    merchant_to_idx = {mid: i + len(user_ids) for i, mid in enumerate(merchant_ids)}

    n_users = len(user_ids)
    n_merchants = len(merchant_ids)
    n_nodes = n_users + n_merchants

    log.info("graph_nodes", n_users=n_users, n_merchants=n_merchants, n_total=n_nodes)

    # --- User node features ---
    user_features_list: list[np.ndarray] = []
    for uid in user_ids:
        user_df = df[df["user_id"] == uid]
        feats = np.array([
            user_df["amount"].astype(float).mean(),
            user_df["amount"].astype(float).std() if len(user_df) > 1 else 0.0,
            float(len(user_df)),
            user_df["is_fraud"].mean(),
            user_df["f_txn_count_24h"].mean() if "f_txn_count_24h" in df.columns else 0.0,
            user_df["f_distance_from_home_km"].mean() if "f_distance_from_home_km" in df.columns else 0.0,
            user_df["f_is_new_device"].mean() if "f_is_new_device" in df.columns else 0.0,
            user_df["f_amount_deviation"].mean() if "f_amount_deviation" in df.columns else 0.0,
        ], dtype=np.float32)
        user_features_list.append(feats)

    user_features = np.stack(user_features_list) if user_features_list else np.zeros((0, 8), dtype=np.float32)

    # --- Merchant node features ---
    merchant_features_list: list[np.ndarray] = []
    for mid in merchant_ids:
        merch_df = df[df["merchant_id"] == mid]
        feats = np.array([
            merch_df["amount"].astype(float).mean(),
            merch_df["amount"].astype(float).std() if len(merch_df) > 1 else 0.0,
            float(len(merch_df)),
            merch_df["is_fraud"].mean(),
            merch_df["f_merchant_category_risk"].mean() if "f_merchant_category_risk" in df.columns else 0.01,
            0.0, 0.0, 0.0,  # padding to match user feature dim
        ], dtype=np.float32)
        merchant_features_list.append(feats)

    merchant_features = np.stack(merchant_features_list) if merchant_features_list else np.zeros((0, 8), dtype=np.float32)

    # --- Combine node features ---
    x = torch.tensor(np.vstack([user_features, merchant_features]), dtype=torch.float)

    # --- Node type indicator (0=user, 1=merchant) ---
    node_type = torch.cat([
        torch.zeros(n_users, dtype=torch.long),
        torch.ones(n_merchants, dtype=torch.long),
    ])

    # --- Build edges (bidirectional) ---
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_features_list: list[list[float]] = []
    edge_labels: list[int] = []

    timestamps = pd.to_datetime(df["timestamp"])
    min_ts = timestamps.min()
    time_deltas = (timestamps - min_ts).dt.total_seconds().values

    for i, row in df.iterrows():
        u_idx = user_to_idx[row["user_id"]]
        m_idx = merchant_to_idx[row["merchant_id"]]
        amount = float(row["amount"])
        td = float(time_deltas[i]) if i < len(time_deltas) else 0.0
        is_fraud = int(row["is_fraud"])

        # User -> Merchant
        src_list.append(u_idx)
        dst_list.append(m_idx)
        edge_features_list.append([amount, td, float(is_fraud)])
        edge_labels.append(is_fraud)

        # Merchant -> User (bidirectional)
        src_list.append(m_idx)
        dst_list.append(u_idx)
        edge_features_list.append([amount, td, float(is_fraud)])
        edge_labels.append(is_fraud)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    edge_y = torch.tensor(edge_labels, dtype=torch.long)

    # Node-level labels: a node is fraudulent if any of its edges are
    node_labels = torch.zeros(n_nodes, dtype=torch.long)
    for i, row in df[df["is_fraud"] == 1].iterrows():
        u_idx = user_to_idx[row["user_id"]]
        m_idx = merchant_to_idx[row["merchant_id"]]
        node_labels[u_idx] = 1
        node_labels[m_idx] = 1

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_labels,
        edge_y=edge_y,
        node_type=node_type,
    )

    log.info(
        "graph_built",
        n_nodes=data.num_nodes,
        n_edges=data.num_edges,
        n_fraud_nodes=int(node_labels.sum()),
    )
    return data


# ---------------------------------------------------------------------------
# GraphSAGE model
# ---------------------------------------------------------------------------


def _build_model(
    in_channels: int,
    hidden_channels: int = 128,
    out_channels: int = 64,
    n_classes: int = 2,
    dropout: float = 0.3,
) -> Any:
    """Build a 2-layer GraphSAGE model with MLP classification head."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GraphSAGEFraud(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)

            # MLP classification head
            self.classifier = nn.Sequential(
                nn.Linear(out_channels, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes),
            )

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)

            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)

            return self.classifier(x)

        def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            return x

    return GraphSAGEFraud()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_gnn(
    data: Any,
    *,
    epochs: int = 200,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 20,
    hidden_channels: int = 128,
    out_channels: int = 64,
    dropout: float = 0.3,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[Any, dict[str, Any]]:
    """Train the GraphSAGE model.

    Uses NeighborLoader for mini-batch training with class-weighted
    cross-entropy loss.  Early stopping on validation AUC.

    Args:
        data: PyTorch Geometric Data object.
        epochs: Max training epochs.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        patience: Early stopping patience.
        hidden_channels: Hidden layer size.
        out_channels: Embedding dimension.
        dropout: Dropout rate.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from torch_geometric.loader import NeighborLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("training_device", device=str(device))

    # --- Train/val/test masks ---
    n_nodes = data.num_nodes
    perm = torch.randperm(n_nodes)
    n_test = int(n_nodes * test_ratio)
    n_val = int(n_nodes * val_ratio)

    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)

    test_mask[perm[:n_test]] = True
    val_mask[perm[n_test : n_test + n_val]] = True
    train_mask[perm[n_test + n_val :]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # --- Class weights ---
    y_train = data.y[train_mask]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0:
        weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
    else:
        weight = torch.tensor([1.0, 1.0], dtype=torch.float, device=device)

    log.info("class_distribution", train_pos=n_pos, train_neg=n_neg)

    # --- Model ---
    model = _build_model(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- NeighborLoader for mini-batch ---
    train_loader = NeighborLoader(
        data,
        num_neighbors=[25, 10],
        batch_size=512,
        input_nodes=train_mask,
        shuffle=True,
    )

    # --- Training ---
    best_val_auc = 0.0
    best_epoch = 0
    best_state: dict[str, Any] = {}
    no_improve = 0

    data = data.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y, weight=weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_probs = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
            val_labels = data.y[data.val_mask].cpu().numpy()

            if len(np.unique(val_labels)) > 1:
                val_auc = roc_auc_score(val_labels, val_probs)
            else:
                val_auc = 0.0

        if epoch % 10 == 0 or epoch == 1:
            log.info("epoch", epoch=epoch, loss=round(avg_loss, 5), val_auc=round(val_auc, 5))

        # --- Early stopping ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info("early_stopping", epoch=epoch, best_epoch=best_epoch, best_val_auc=round(best_val_auc, 5))
            break

    # --- Restore best model ---
    if best_state:
        model.load_state_dict(best_state)

    # --- Test evaluation ---
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_probs = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()
        test_labels = data.y[data.test_mask].cpu().numpy()

        if len(np.unique(test_labels)) > 1:
            test_auc = roc_auc_score(test_labels, test_probs)
        else:
            test_auc = 0.0

    metrics = {
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 5),
        "test_auc": round(test_auc, 5),
        "n_nodes": n_nodes,
        "n_edges": data.num_edges,
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    log.info("training_complete", **metrics)
    return model, metrics


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def save_embeddings(model: Any, data: Any, output_path: str | Path) -> None:
    """Extract and save node embeddings from the trained model."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), embeddings)
    log.info("embeddings_saved", path=str(output_path), shape=embeddings.shape)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def run_training(
    data_path: str,
    output_path: str,
    *,
    epochs: int = 200,
    lr: float = 0.001,
    hidden_channels: int = 128,
    out_channels: int = 64,
) -> dict[str, Any]:
    """Full GNN training pipeline.

    Args:
        data_path: Path to training Parquet.
        output_path: Path to save the model state dict.
        epochs: Max training epochs.
        lr: Learning rate.
        hidden_channels: Hidden layer size.
        out_channels: Embedding dimension.

    Returns:
        Summary dict with metrics and paths.
    """
    import torch

    _check_torch()

    from ml_pipeline.feature_engineering import ALL_FEATURE_NAMES, engineer_features

    log.info("gnn_pipeline_start", data=data_path, output=output_path)
    start_time = time.time()

    # --- Load and engineer features ---
    raw_df = pd.read_parquet(data_path)
    df = engineer_features(raw_df, feature_names=ALL_FEATURE_NAMES)

    # --- Build graph ---
    data = build_transaction_graph(df)

    # --- Train ---
    model, metrics = train_gnn(
        data,
        epochs=epochs,
        lr=lr,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    # --- Save model ---
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    log.info("model_saved", path=output_path)

    # --- Save embeddings ---
    emb_path = output_dir / "embeddings.npy"
    save_embeddings(model, data, emb_path)

    elapsed = time.time() - start_time
    metrics["model_path"] = output_path
    metrics["embeddings_path"] = str(emb_path)
    metrics["training_time_s"] = round(elapsed, 1)

    log.info("gnn_pipeline_complete", **metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraphSAGE GNN for fraud detection")
    parser.add_argument("--data", required=True, help="Path to training Parquet file")
    parser.add_argument("--output", default="models/gnn_v1.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden channels (default: 128)")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dim (default: 64)")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
        hidden_channels=args.hidden,
        out_channels=args.embed_dim,
    )


if __name__ == "__main__":
    main()
