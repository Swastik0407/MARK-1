
import torch
import torch.nn as nn

from src.models.transformer_layers import EncoderLayer


class AttributeEncoder(nn.Module):

    def __init__(
        self,
        input_dim=21,
        hidden_size=64,
        ffn_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            hidden_size
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_size=ffn_size
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):

        x = x.unsqueeze(0)

        h = self.input_projection(x)

        for layer in self.encoder_layers:
            h = layer(h)

        return h.squeeze(0)
