
import torch
import torch.nn as nn

from src.models.transformer_layers import EncoderLayer


class StructuralEncoder(nn.Module):
    """
    Structural Transformer Encoder.

    Learns node representations using:
    1. Structural node features S
    2. Graph adjacency matrix A

    Input:
        structural_features: [num_nodes, 3]
        adjacency: [num_nodes, num_nodes]

    Output:
        H_S: [num_nodes, hidden_size]
    """

    def __init__(
        self,
        input_dim=3,
        hidden_size=64,
        ffn_size=128,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            hidden_size
        )

        self.input_dropout = nn.Dropout(dropout_rate)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, structural_features, adjacency):

        num_nodes = structural_features.size(0)

        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(
                "Adjacency matrix dimensions do not match "
                "the number of nodes."
            )

        # Step 1: Project structural features
        x = self.input_projection(structural_features)

        x = self.input_dropout(x)

        # Step 2: Construct graph attention mask
        # Allow attention between connected nodes.
        # Always allow self-attention.
        attention_mask = adjacency > 0

        attention_mask = attention_mask | torch.eye(
            num_nodes,
            dtype=torch.bool,
            device=adjacency.device
        )

        # Block attention to non-neighbors
        # Shape: [1, 1, N, N]
        attention_bias = torch.zeros(
            num_nodes,
            num_nodes,
            dtype=x.dtype,
            device=x.device
        )

        attention_bias = attention_bias.masked_fill(
            ~attention_mask,
            float("-inf")
        )

        attention_bias = attention_bias.unsqueeze(0).unsqueeze(0)

        # Step 3: Add batch dimension
        x = x.unsqueeze(0)

        # Step 4: Transformer encoding
        for layer in self.encoder_layers:
            x = layer(x, attention_bias)

        # Step 5: Normalize
        x = self.output_norm(x)

        # Step 6: Remove batch dimension
        h_s = x.squeeze(0)

        return h_s
