
import torch

from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph
from src.data.structural_features import extract_structural_features
from src.data.graph_collator import create_adjacency_matrix

from src.models.structural_encoder import StructuralEncoder


# Load actual Books dataset
data = load_books_dataset()

x, edge_index, y = preprocess_graph(data)

num_nodes = x.size(0)

# Extract structural features
structural_features = extract_structural_features(
    edge_index,
    num_nodes
)

# Construct adjacency matrix
adjacency = create_adjacency_matrix(
    edge_index,
    num_nodes
)

# Initialize Structural Encoder
model = StructuralEncoder(
    input_dim=3,
    hidden_size=64,
    ffn_size=128,
    num_heads=4,
    num_layers=2
)

# Forward pass
h_s = model(
    structural_features,
    adjacency
)

print("Adjacency shape:", adjacency.shape)
print("Structural features shape:", structural_features.shape)
print("Structural embedding shape:", h_s.shape)

print(
    "Contains NaN:",
    torch.isnan(h_s).any().item()
)

print(
    "Contains Inf:",
    torch.isinf(h_s).any().item()
)
