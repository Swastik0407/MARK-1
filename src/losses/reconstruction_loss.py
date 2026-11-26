
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualReconstructionLoss(nn.Module):

    def __init__(
        self,
        lambda_attribute=1.0,
        lambda_structure=1.0
    ):
        super().__init__()

        self.lambda_attribute = lambda_attribute
        self.lambda_structure = lambda_structure

    def forward(self, x, x_hat, a, a_logits):

        # 1. Attribute reconstruction loss
        attribute_loss = F.mse_loss(x_hat, x)

        # 2. Structural reconstruction loss
        num_nodes = a.size(0)

        # Exclude self-loops
        mask = ~torch.eye(
            num_nodes,
            dtype=torch.bool,
            device=a.device
        )

        target = a[mask]
        logits = a_logits[mask]

        positive_count = target.sum()
        negative_count = target.numel() - positive_count

        if positive_count.item() == 0:
            raise ValueError("No positive edges found.")

        pos_weight = negative_count / positive_count

        structural_loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight
        )

        # 3. Combined loss
        total_loss = (
            self.lambda_attribute * attribute_loss
            +
            self.lambda_structure * structural_loss
        )

        return {
            "total_loss": total_loss,
            "attribute_loss": attribute_loss,
            "structural_loss": structural_loss,
            "positive_weight": pos_weight
        }
