
import torch

from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph
from src.data.graph_collator import create_adjacency_matrix
from src.data.structural_features import extract_structural_features

from src.models.dual_transformer import DualTransformer
from src.losses.reconstruction_loss import DualReconstructionLoss


# ---------------------------------
# 1. Load Books dataset
# ---------------------------------
data = load_books_dataset()

x, edge_index, y = preprocess_graph(data)

num_nodes = x.size(0)

# ---------------------------------
# 2. Prepare graph inputs
# ---------------------------------
structural_features = extract_structural_features(
    edge_index,
    num_nodes
)

adjacency = create_adjacency_matrix(
    edge_index,
    num_nodes
)

# ---------------------------------
# 3. Initialize model
# ---------------------------------
model = DualTransformer()

# ---------------------------------
# 4. Forward pass
# ---------------------------------
outputs = model(
    x,
    structural_features,
    adjacency
)

# ---------------------------------
# 5. Inspect attribute values
# ---------------------------------
print("\n" + "=" * 50)
print("ATTRIBUTE VALUE INSPECTION")
print("=" * 50)

print("\nOriginal X:")
print("Shape:", x.shape)
print("Minimum:", x.min().item())
print("Maximum:", x.max().item())
print("Mean:", x.mean().item())
print("Standard deviation:", x.std().item())

print("\nReconstructed X:")
print("Shape:", outputs["x_hat"].shape)
print("Minimum:", outputs["x_hat"].min().item())
print("Maximum:", outputs["x_hat"].max().item())
print("Mean:", outputs["x_hat"].mean().item())
print("Standard deviation:", outputs["x_hat"].std().item())

# ---------------------------------
# 6. Inspect structural outputs
# ---------------------------------
print("\n" + "=" * 50)
print("STRUCTURAL OUTPUT INSPECTION")
print("=" * 50)

print("\nAdjacency logits:")
print("Shape:", outputs["a_logits"].shape)
print("Minimum:", outputs["a_logits"].min().item())
print("Maximum:", outputs["a_logits"].max().item())
print("Contains NaN:", torch.isnan(outputs["a_logits"]).any().item())

print("\nAdjacency probabilities:")
print("Shape:", outputs["a_hat"].shape)
print("Minimum:", outputs["a_hat"].min().item())
print("Maximum:", outputs["a_hat"].max().item())
print("Symmetric:", torch.allclose(
    outputs["a_hat"],
    outputs["a_hat"].T,
    atol=1e-6
))

# ---------------------------------
# 7. Calculate reconstruction loss
# ---------------------------------
criterion = DualReconstructionLoss(
    lambda_attribute=1.0,
    lambda_structure=1.0
)

losses = criterion(
    x,
    outputs["x_hat"],
    adjacency,
    outputs["a_logits"]
)

# ---------------------------------
# 8. Display losses
# ---------------------------------
print("\n" + "=" * 50)
print("RECONSTRUCTION LOSS RESULTS")
print("=" * 50)

for name, value in losses.items():
    print(f"{name}: {value.item():.6f}")

# ---------------------------------
# 9. Validate gradients
# ---------------------------------
losses["total_loss"].backward()

attribute_gradients = any(
    p.grad is not None
    for p in model.attribute_encoder.parameters()
)

structural_gradients = any(
    p.grad is not None
    for p in model.structural_encoder.parameters()
)

decoder_gradients = any(
    p.grad is not None
    for p in model.structural_decoder.parameters()
)

print("\n" + "=" * 50)
print("GRADIENT VALIDATION")
print("=" * 50)

print("Attribute encoder gradients:", attribute_gradients)
print("Structural encoder gradients:", structural_gradients)
print("Structural decoder gradients:", decoder_gradients)

print("\nReconstruction loss test completed!")
