
import torch
from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph
from src.data.graph_collator import create_adjacency_matrix

data = load_books_dataset()

x, edge_index, y = preprocess_graph(data)

A = create_adjacency_matrix(
    edge_index,
    num_nodes=x.shape[0]
)

print("\nFinal adjacency matrix:")
print("A shape:", A.shape)
print("A[0, 1]:", A[0, 1].item())
print("A[1, 0]:", A[1, 0].item())
print("Non-zero entries:", torch.count_nonzero(A).item())
