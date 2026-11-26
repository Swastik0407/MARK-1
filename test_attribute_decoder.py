
import torch

from src.models.attribute_decoder import AttributeDecoder

# Books dataset dimensions
num_nodes = 1418
hidden_size = 64
original_feature_dim = 21

# Simulated Attribute Transformer embeddings
h_a = torch.randn(
    num_nodes,
    hidden_size
)

# Initialize Attribute Decoder
decoder = AttributeDecoder(
    hidden_size=64,
    decoder_hidden_size=32,
    output_dim=21
)

# Reconstruct node features
x_hat = decoder(h_a)

print("Input embedding shape:", h_a.shape)
print("Reconstructed feature shape:", x_hat.shape)

print(
    "Contains NaN:",
    torch.isnan(x_hat).any().item()
)

print(
    "Contains Inf:",
    torch.isinf(x_hat).any().item()
)
