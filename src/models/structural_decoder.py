
import torch
import torch.nn as nn


class StructuralDecoder(nn.Module):

    def __init__(self, hidden_size=64):
        super().__init__()

        # Learnable symmetric bilinear transformation
        self.weight = nn.Parameter(
            torch.eye(hidden_size) * 0.01
        )

        # Learnable bias, initialized negatively
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h_s):

        # Normalize node embeddings to control logit magnitude
        h_s = nn.functional.normalize(h_s, p=2, dim=1)

        # Ensure symmetric transformation
        symmetric_weight = (
            self.weight + self.weight.transpose(0, 1)
        ) / 2

        # Reconstruct adjacency logits
        a_logits = (
            h_s @ symmetric_weight @ h_s.transpose(0, 1)
        ) + self.bias

        return a_logits
