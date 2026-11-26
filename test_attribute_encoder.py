
import torch

from src.models.attribute_encoder import AttributeEncoder

# Books dataset dimensions
num_nodes = 1418
input_dim = 21

# Dummy node features
x = torch.randn(num_nodes, input_dim)

# Initialize Attribute Encoder
model = AttributeEncoder(
    input_dim=21,
    hidden_size=64,
    ffn_size=128,
    num_heads=4,
    num_layers=2
)

# Forward pass
h_a = model(x)

print("Input shape:", x.shape)
print("Attribute embedding shape:", h_a.shape)

print("Contains NaN:", torch.isnan(h_a).any().item())
print("Contains Inf:", torch.isinf(h_a).any().item())
