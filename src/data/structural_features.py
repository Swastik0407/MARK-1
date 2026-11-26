
import torch
from torch_geometric.utils import degree


def extract_structural_features(edge_index, num_nodes):
    """
    Extract basic structural features for every node.

    Returns:
        structural_features: [num_nodes, 3]
    """

    # 1. Calculate node degree
    node_degree = degree(
        edge_index[0],
        num_nodes=num_nodes,
        dtype=torch.float32
    )

    # 2. Normalize degree
    max_degree = node_degree.max().clamp(min=1)

    normalized_degree = node_degree / max_degree

    # 3. Compute local clustering coefficient
    neighbors = [set() for _ in range(num_nodes)]

    edges = edge_index.t().tolist()

    for u, v in edges:
        neighbors[u].add(v)

    clustering = torch.zeros(num_nodes)

    for node in range(num_nodes):
        node_neighbors = neighbors[node]
        k = len(node_neighbors)

        if k < 2:
            continue

        links = 0

        for u in node_neighbors:
            for v in node_neighbors:
                if u < v and v in neighbors[u]:
                    links += 1

        clustering[node] = (
            2.0 * links / (k * (k - 1))
        )

    # 4. Combine structural features
    structural_features = torch.stack(
        [
            node_degree,
            normalized_degree,
            clustering
        ],
        dim=1
    )

    print("\nStructural features extracted!")
    print("Structural feature shape:", structural_features.shape)

    return structural_features
