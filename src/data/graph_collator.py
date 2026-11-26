
import torch


def create_adjacency_matrix(edge_index, num_nodes):
    """
    Generate a dense adjacency matrix from edge_index.

    Returns:
        adjacency: [num_nodes, num_nodes]
    """

    adjacency = torch.zeros(
        (num_nodes, num_nodes),
        dtype=torch.float32
    )

    # Insert graph connections
    source = edge_index[0]
    target = edge_index[1]

    adjacency[source, target] = 1.0

    # Ensure undirected connectivity
    adjacency = torch.maximum(
        adjacency,
        adjacency.T
    )

    # Remove self-loops
    adjacency.fill_diagonal_(0)

    # Validate symmetry
    assert torch.equal(adjacency, adjacency.T)

    print("\nAdjacency matrix generated!")
    print("Shape:", adjacency.shape)
    print("Total connections:", int(adjacency.sum().item()))
    print("Symmetric:", torch.equal(adjacency, adjacency.T))

    return adjacency
