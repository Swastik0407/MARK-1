
import torch
import torch.nn as nn

from src.models.attribute_encoder import AttributeEncoder
from src.models.structural_encoder import StructuralEncoder
from src.models.attribute_decoder import AttributeDecoder
from src.models.structural_decoder import StructuralDecoder


class DualTransformer(nn.Module):

    def __init__(
        self,
        input_dim=21,
        structural_dim=3,
        hidden_size=64,
        ffn_size=128,
        num_heads=4,
        num_layers=2,
        decoder_hidden_size=32,
        dropout=0.1
    ):
        super().__init__()

        # Attribute Encoder
        self.attribute_encoder = AttributeEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Attribute Decoder
        self.attribute_decoder = AttributeDecoder(
            hidden_size=hidden_size,
            output_dim=input_dim,
            decoder_hidden_size=decoder_hidden_size
        )

        # Structural Encoder
        self.structural_encoder = StructuralEncoder(
            input_dim=structural_dim,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            num_heads=num_heads,
            num_layers=num_layers
        )

        # Structural Decoder
        self.structural_decoder = StructuralDecoder(
            hidden_size=hidden_size
        )

    def forward(self, x, structural_features, adjacency):

        # Attribute branch
        h_a = self.attribute_encoder(x)
        x_hat = self.attribute_decoder(h_a)

        # Structural branch
        h_s = self.structural_encoder(
            structural_features,
            adjacency
        )

        a_logits = self.structural_decoder(h_s)

        a_hat = torch.sigmoid(a_logits)

        return {
            "h_a": h_a,
            "h_s": h_s,
            "x_hat": x_hat,
            "a_hat": a_hat,
            "a_logits": a_logits
        }
