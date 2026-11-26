
import torch
import torch.nn.functional as F


def calculate_anomaly_scores(x, x_hat, adjacency, a_logits):
    """
    Calculate per-node attribute and structural reconstruction errors.

    Returns:
        attribute_error: [N]
        structural_error: [N]
    """

    num_nodes = adjacency.size(0)

    if x.shape != x_hat.shape:
        raise ValueError("Original and reconstructed features must match.")

    if adjacency.shape != a_logits.shape:
        raise ValueError("Adjacency and reconstructed logits must match.")

    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("Adjacency must be a square matrix.")

    # -----------------------------------------
    # 1. Attribute reconstruction error
    # -----------------------------------------
    attribute_error = torch.mean(
        (x - x_hat) ** 2,
        dim=1
    )

    # -----------------------------------------
    # 2. Create mask excluding self-loops
    # -----------------------------------------
    mask = ~torch.eye(
        num_nodes,
        dtype=torch.bool,
        device=adjacency.device
    )

    # Keep adjacency and logits as [N, N]
    target = adjacency
    logits = a_logits

    # -----------------------------------------
    # 3. Calculate positive-class weight
    # -----------------------------------------
    valid_targets = target[mask]

    positive_count = valid_targets.sum()
    negative_count = valid_targets.numel() - positive_count

    if positive_count.item() == 0:
        raise ValueError("Adjacency contains no positive edges.")

    pos_weight = negative_count / positive_count

    # -----------------------------------------
    # 4. Calculate weighted BCE per matrix entry
    # -----------------------------------------
    element_loss = (
        pos_weight * target * F.softplus(-logits)
        + (1.0 - target) * F.softplus(logits)
    )

    # Exclude diagonal entries
    element_loss = element_loss.masked_fill(
        ~mask,
        0.0
    )

    # -----------------------------------------
    # 5. Calculate structural error per node
    # -----------------------------------------
    structural_error = (
        element_loss.sum(dim=1) / (num_nodes - 1)
    )

    return {
        "attribute_error": attribute_error,
        "structural_error": structural_error
    }
