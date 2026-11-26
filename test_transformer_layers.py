
import torch

from src.models.transformer_layers import (
    MultiHeadAttention,
    FeedForwardNetwork,
    EncoderLayer
)

# Test configuration
batch_size = 1
num_nodes = 1418
hidden_size = 64
ffn_size = 128
num_heads = 4

# Dummy node representations
x = torch.randn(
    batch_size,
    num_nodes,
    hidden_size
)

# Create encoder layer
encoder = EncoderLayer(
    hidden_size=hidden_size,
    ffn_size=ffn_size,
    num_heads=num_heads
)

# Forward pass
output = encoder(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print(
    "Output contains NaN:",
    torch.isnan(output).any().item()
)

print(
    "Output contains Inf:",
    torch.isinf(output).any().item()
)
