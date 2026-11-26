
import torch

files = [
    "data/raw/Books.pt",
    "data/raw/Disney.pt"
]

for file in files:
    print("\n" + "=" * 55)
    print("DATASET:", file)
    print("=" * 55)

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False
    )

    # 1. Check anomaly label distribution
    labels, counts = torch.unique(
        data.y,
        return_counts=True
    )

    print("\nAnomaly label distribution:")
    for label, count in zip(labels, counts):
        print(f"Label {label.item()}: {count.item()} nodes")

    # 2. Check graph symmetry
    edges = data.edge_index.t().tolist()

    edge_set = set(map(tuple, edges))

    reverse_edges = sum(
        1 for u, v in edge_set
        if (v, u) in edge_set
    )

    missing_reverse = sum(
        1 for u, v in edge_set
        if (v, u) not in edge_set
    )

    print("\nGraph symmetry:")
    print("Stored edges:", len(edges))
    print("Unique directed edges:", len(edge_set))
    print("Edges with reverse present:", reverse_edges)
    print("Edges missing reverse:", missing_reverse)

    # 3. Check self-loops
    self_loops = sum(
        1 for u, v in edge_set if u == v
    )

    print("Self-loops:", self_loops)
