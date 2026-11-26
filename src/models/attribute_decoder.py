
import torch
import torch.nn as nn


class AttributeDecoder(nn.Module):
    """
    Attribute Decoder.

    Reconstructs original node attributes
    from Attribute Transformer embeddings.

    Input:
        h_a: [num_nodes, hidden_size]

    Output:
        x_hat: [num_nodes, output_dim]
    """

    def __init__(
        self,
        hidden_size=64,
        decoder_hidden_size=32,
        output_dim=21,
        dropout_rate=0.1
    ):
        super().__init__()

        # First reconstruction layer
        self.linear1 = nn.Linear(
            hidden_size,
            decoder_hidden_size
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout_rate)

        # Map back to original feature dimensions
        self.linear2 = nn.Linear(
            decoder_hidden_size,
            output_dim
        )

    def forward(self, h_a):

        # Step 1: Transform learned embeddings
        x = self.linear1(h_a)

        # Step 2: Non-linear activation
        x = self.activation(x)

        # Step 3: Dropout
        x = self.dropout(x)

        # Step 4: Reconstruct original node features
        x_hat = self.linear2(x)

        return x_hat
