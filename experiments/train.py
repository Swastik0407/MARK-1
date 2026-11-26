
import os
import csv
import torch

from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph
from src.data.structural_features import extract_structural_features
from src.data.graph_collator import create_adjacency_matrix

from src.models.dual_transformer import DualTransformer
from src.losses.reconstruction_loss import DualReconstructionLoss

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger


def train():

    # --------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------
    config = load_config("configs/books.yaml")

    set_seed(config["seed"])

    training_config = config["training"]
    model_config = config["model"]
    loss_config = config["loss"]
    output_config = config["output"]

    logger = get_logger(log_dir=output_config["log_dir"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Starting Dual Transformer training")
    logger.info("Device: %s", device)

    # --------------------------------------------------
    # 2. Load and preprocess Books dataset
    # --------------------------------------------------
    data = load_books_dataset()

    x, edge_index, y = preprocess_graph(data)

    num_nodes = x.size(0)

    structural_features = extract_structural_features(
        edge_index,
        num_nodes
    )

    adjacency = create_adjacency_matrix(
        edge_index,
        num_nodes
    )

    # Labels are deliberately NOT passed into the model
    # or the reconstruction loss.

    x = x.to(device)
    structural_features = structural_features.to(device)
    adjacency = adjacency.to(device)

    logger.info("Dataset preparation completed")
    logger.info("Nodes: %d", num_nodes)
    logger.info("Node features: %d", x.size(1))
    logger.info("Structural features: %d", structural_features.size(1))

    # --------------------------------------------------
    # 3. Initialize model
    # --------------------------------------------------
    model = DualTransformer(**model_config).to(device)

    criterion = DualReconstructionLoss(
        lambda_attribute=loss_config["lambda_attribute"],
        lambda_structure=loss_config["lambda_structure"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )

    logger.info("Model initialized successfully")

    # --------------------------------------------------
    # 4. Prepare output directories
    # --------------------------------------------------
    checkpoint_dir = output_config["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_config["log_dir"], exist_ok=True)

    best_loss = float("inf")
    best_epoch = 0

    history = []

    epochs = training_config["epochs"]
    log_interval = training_config["log_interval"]
    gradient_clip = training_config["gradient_clip"]

    # --------------------------------------------------
    # 5. Training loop
    # --------------------------------------------------
    for epoch in range(1, epochs + 1):

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(
            x,
            structural_features,
            adjacency
        )

        # Calculate reconstruction losses
        losses = criterion(
            x,
            outputs["x_hat"],
            adjacency,
            outputs["a_logits"]
        )

        total_loss = losses["total_loss"]

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss detected at epoch {epoch}"
            )

        # Backpropagation
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip
        )

        # Update model parameters
        optimizer.step()

        # Extract loss values
        total_value = total_loss.item()
        attribute_value = losses["attribute_loss"].item()
        structural_value = losses["structural_loss"].item()

        positive_weight = losses["positive_weight"].item()

        history.append({
            "epoch": epoch,
            "total_loss": total_value,
            "attribute_loss": attribute_value,
            "structural_loss": structural_value,
            "positive_weight": positive_weight
        })

        # Save best checkpoint
        if total_value < best_loss:

            best_loss = total_value
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_value,
                    "config": config
                },
                os.path.join(
                    checkpoint_dir,
                    "best_model.pt"
                )
            )

        # Display progress
        if epoch == 1 or epoch % log_interval == 0:

            logger.info(
                "Epoch [%d/%d] | Total: %.6f | "
                "Attribute: %.6f | Structural: %.6f",
                epoch,
                epochs,
                total_value,
                attribute_value,
                structural_value
            )

    # --------------------------------------------------
    # 6. Save final model
    # --------------------------------------------------
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": history[-1]["total_loss"],
            "config": config
        },
        os.path.join(
            checkpoint_dir,
            "final_model.pt"
        )
    )

    # --------------------------------------------------
    # 7. Save training history
    # --------------------------------------------------
    history_path = output_config["history_file"]

    os.makedirs(
        os.path.dirname(history_path),
        exist_ok=True
    )

    with open(history_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "total_loss",
                "attribute_loss",
                "structural_loss",
                "positive_weight"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    # --------------------------------------------------
    # 8. Training summary
    # --------------------------------------------------
    logger.info("Training completed")
    logger.info("Best epoch: %d", best_epoch)
    logger.info("Best total loss: %.6f", best_loss)

    logger.info(
        "Final total loss: %.6f",
        history[-1]["total_loss"]
    )

    logger.info(
        "Best checkpoint saved: %s",
        os.path.join(checkpoint_dir, "best_model.pt")
    )

    logger.info(
        "Final checkpoint saved: %s",
        os.path.join(checkpoint_dir, "final_model.pt")
    )

    logger.info("Training history saved: %s", history_path)


if __name__ == "__main__":
    train()
