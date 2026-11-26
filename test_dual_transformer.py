
import torch

from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph
from src.data.structural_features import extract_structural_features
from src.data.graph_collator import create_adjacency_matrix

from src.models.dual_transformer import DualTransformer


# ---------------------------------
# 1. Load Books dataset
# ---------------------------------
data = load_books_dataset()

x, edge_index, y = preprocess_graph(data)

num_nodes = x.size(0)

# ---------------------------------
# 2. Extract structural features
# ---------------------------------
structural_features = extract_structural_features(
    edge_index,
    num_nodes
)

# ---------------------------------
# 3. Generate adjacency matrix
# ---------------------------------
adjacency = create_adjacency_matrix(
    edge_index,
    num_nodes
)

# ---------------------------------
# 4. Initialize Dual Transformer
# ---------------------------------
model = DualTransformer(
    input_dim=21,
    structural_dim=3,
    hidden_size=64,
    ffn_size=128,
    num_heads=4,
    num_layers=2,
    decoder_hidden_size=32
)

model.eval()

# ---------------------------------
# 5. Forward pass
# ---------------------------------
with torch.no_grad():

    outputs = model(
        x,
        structural_features,
        adjacency
    )

# ---------------------------------
# 6. Display output shapes
# ---------------------------------
print("\nDual Transformer Integration Test")

print("Original X:", x.shape)
print("Structural features:", structural_features.shape)
print("Original A:", adjacency.shape)

print("\nModel outputs:")

for name, tensor in outputs.items():
    print(name, ":", tensor.shape)

# ---------------------------------
# 7. Validate outputs
# ---------------------------------
for name, tensor in outputs.items():

    assert torch.isfinite(tensor).all(), (
        f"{name} contains NaN or Inf"
    )

assert outputs["x_hat"].shape == x.shape

assert outputs["a_hat"].shape == adjacency.shape

assert torch.allclose(
    outputs["a_hat"],
    outputs["a_hat"].T,
    atol=1e-6
)

print("\nAll output checks passed!")
