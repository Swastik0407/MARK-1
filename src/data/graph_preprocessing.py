
import torch
from torch_geometric.utils import (
    remove_self_loops,
    to_undirected
)


def preprocess_graph(data):

    # Validate dataset
    if not hasattr(data, "x"):
        raise ValueError("Dataset has no node features.")

    if not hasattr(data, "edge_index"):
        raise ValueError("Dataset has no edge_index.")

    if not hasattr(data, "y"):
        raise ValueError("Dataset has no anomaly labels.")

    # ---------------------------------
    # 1. Prepare node features
    # ---------------------------------
    x = data.x.float()

    if not torch.isfinite(x).all():
        raise ValueError("Node features contain NaN or Inf.")

    # Standardize each feature independently
    feature_mean = x.mean(dim=0, keepdim=True)
    feature_std = x.std(dim=0, keepdim=True)

    # Avoid division by zero for constant features
    feature_std = torch.where(
        feature_std < 1e-8,
        torch.ones_like(feature_std),
        feature_std
    )

    x = (x - feature_mean) / feature_std

    # ---------------------------------
    # 2. Prepare undirected graph
    # ---------------------------------
    edge_index = data.edge_index.long()

    edge_index, _ = remove_self_loops(edge_index)

    edge_index = to_undirected(
        edge_index,
        num_nodes=x.size(0)
    )

    edge_index = torch.unique(
        edge_index,
        dim=1
    )

    # ---------------------------------
    # 3. Prepare anomaly labels
    # ---------------------------------
    y = data.y.bool()

    # ---------------------------------
    # 4. Validate output
    # ---------------------------------
    if not torch.isfinite(x).all():
        raise ValueError("Normalized features contain NaN or Inf.")

    print("\nGraph preprocessing completed!")
    print("Nodes:", x.size(0))
    print("Features:", x.size(1))
    print("Undirected edge_index shape:", edge_index.shape)
    print("Anomalies:", y.sum().item())
    print("Normal nodes:", (~y).sum().item())

    print("\nNormalized feature statistics:")
    print("Minimum:", x.min().item())
    print("Maximum:", x.max().item())
    print("Mean:", x.mean().item())
    print("Standard deviation:", x.std().item())

    return x, edge_index, y
