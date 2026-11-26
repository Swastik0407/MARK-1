
from pathlib import Path
import torch


def load_books_dataset():
    """
    Load the Books attributed graph dataset.

    Returns:
        data: PyTorch Geometric Data object
    """

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = project_root / "data" / "raw" / "Books.pt"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Books dataset not found: {dataset_path}"
        )

    # Load trusted local PyTorch Geometric dataset
    data = torch.load(
        dataset_path,
        map_location="cpu",
        weights_only=False
    )

    # Validate required attributes
    required_fields = ["x", "edge_index", "y"]

    for field in required_fields:
        if not hasattr(data, field):
            raise ValueError(
                f"Missing required dataset field: {field}"
            )

    # Validate tensor dimensions
    if data.x.dim() != 2:
        raise ValueError("Node features must be a 2D tensor.")

    if data.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    if data.y.dim() != 1:
        raise ValueError("Node labels must be a 1D tensor.")

    if data.x.shape[0] != data.y.shape[0]:
        raise ValueError("Node features and labels count mismatch.")

    print("Books dataset loaded successfully!")
    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)
    print("Node features:", data.num_node_features)

    return data
