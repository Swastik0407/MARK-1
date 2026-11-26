
import torch

from src.models.structural_decoder import StructuralDecoder

# Books dataset dimensions
num_nodes = 1418
hidden_size = 64

# Simulated structural embeddings
h_s = torch.randn(
    num_nodes,
    hidden_size
)

# Initialize decoder
decoder = StructuralDecoder()

# Reconstruct adjacency matrix
a_hat = decoder(h_s)

print("Input embedding shape:", h_s.shape)
print("Reconstructed adjacency shape:", a_hat.shape)

print("Minimum value:", a_hat.min().item())
print("Maximum value:", a_hat.max().item())

print("Contains NaN:", torch.isnan(a_hat).any().item())
print("Contains Inf:", torch.isinf(a_hat).any().item())

print(
    "Symmetric:",
    torch.allclose(a_hat, a_hat.T)
)
