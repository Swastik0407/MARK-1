
import os
import csv
import torch

from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph
from src.data.structural_features import extract_structural_features
from src.data.graph_collator import create_adjacency_matrix

from src.models.dual_transformer import DualTransformer

from src.detection.anomaly_score import calculate_anomaly_scores
from src.detection.threshold import calculate_threshold, predict_anomalies
from src.evaluation.metrics import calculate_metrics

from src.utils.config import load_config
from src.utils.seed import set_seed


def normalize_scores(scores):
    """Min-max normalize a vector of anomaly scores."""

    minimum = scores.min()
    maximum = scores.max()

    if (maximum - minimum).abs().item() < 1e-12:
        return torch.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def evaluate():

    # --------------------------------------------------
    # 1. Load configuration and select device
    # --------------------------------------------------
    config = load_config("configs/books.yaml")
    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nStarting Dual Transformer evaluation")
    print("Device:", device)

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

    x = x.to(device)
    structural_features = structural_features.to(device)
    adjacency = adjacency.to(device)
    y = y.to(device)

    # --------------------------------------------------
    # 3. Load trained model checkpoint
    # --------------------------------------------------
    checkpoint_path = "outputs/checkpoints/best_model.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model_config = checkpoint["config"]["model"]

    model = DualTransformer(**model_config).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("\nCheckpoint loaded successfully!")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint training loss:", checkpoint["loss"])

    # --------------------------------------------------
    # 4. Generate model reconstructions
    # --------------------------------------------------
    with torch.no_grad():

        outputs = model(
            x,
            structural_features,
            adjacency
        )

    x_hat = outputs["x_hat"]
    a_logits = outputs["a_logits"]

    print("\nModel reconstruction completed!")
    print("Original X shape:", x.shape)
    print("Reconstructed X shape:", x_hat.shape)
    print("Original A shape:", adjacency.shape)
    print("Reconstructed A shape:", outputs["a_hat"].shape)

    # --------------------------------------------------
    # 5. Calculate reconstruction errors
    # --------------------------------------------------
    errors = calculate_anomaly_scores(
        x,
        x_hat,
        adjacency,
        a_logits
    )

    attribute_error = errors["attribute_error"]
    structural_error = errors["structural_error"]

    # Normalize both error types before combining
    normalized_attribute = normalize_scores(attribute_error)
    normalized_structural = normalize_scores(structural_error)

    # Equal weighting of both branches
    combined_scores = (
        0.5 * normalized_attribute
        + 0.5 * normalized_structural
    )

    print("\nAnomaly scores calculated!")

    print(
        "Attribute error range:",
        attribute_error.min().item(),
        "to",
        attribute_error.max().item()
    )

    print(
        "Structural error range:",
        structural_error.min().item(),
        "to",
        structural_error.max().item()
    )

    print(
        "Combined score range:",
        combined_scores.min().item(),
        "to",
        combined_scores.max().item()
    )

    # --------------------------------------------------
    # 6. Generate predictions using 95th percentile
    # --------------------------------------------------
    threshold = calculate_threshold(
        combined_scores,
        method="percentile",
        percentile=95.0
    )

    percentile_predictions = predict_anomalies(
        combined_scores,
        threshold
    )

    print("\n95th-percentile threshold:", threshold.item())
    print(
        "Predicted anomalies:",
        percentile_predictions.sum().item()
    )

    # --------------------------------------------------
    # 7. Top-28 diagnostic comparison
    # --------------------------------------------------
    # Uses the known anomaly count only to create a
    # diagnostic ranking comparison.
    # This is NOT a deployment threshold.

    known_anomaly_count = int(y.sum().item())

    top_indices = torch.topk(
        combined_scores,
        k=known_anomaly_count
    ).indices

    top28_predictions = torch.zeros(
        num_nodes,
        dtype=torch.bool,
        device=device
    )

    top28_predictions[top_indices] = True

    print(
        "Top-k diagnostic anomaly count:",
        top28_predictions.sum().item()
    )

    # --------------------------------------------------
    # 8. Calculate evaluation metrics
    # --------------------------------------------------
    y_true = y.cpu().numpy()
    score_array = combined_scores.cpu().numpy()

    percentile_metrics = calculate_metrics(
        y_true,
        score_array,
        percentile_predictions.cpu().numpy()
    )

    top28_metrics = calculate_metrics(
        y_true,
        score_array,
        top28_predictions.cpu().numpy()
    )

    print("\n========== 95th Percentile Evaluation ==========")

    for name, value in percentile_metrics.items():
        print(f"{name}: {value:.6f}" if isinstance(value, float)
              else f"{name}: {value}")

    print("\n========== Top-28 Diagnostic Evaluation ==========")

    for name, value in top28_metrics.items():
        print(f"{name}: {value:.6f}" if isinstance(value, float)
              else f"{name}: {value}")

    # --------------------------------------------------
    # 9. Save node anomaly scores
    # --------------------------------------------------
    output_dir = "outputs/anomaly_scores"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        "books_anomaly_scores.csv"
    )

    sorted_indices = torch.argsort(
        combined_scores,
        descending=True
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "node_id",
            "attribute_error",
            "structural_error",
            "normalized_attribute_error",
            "normalized_structural_error",
            "combined_anomaly_score",
            "predicted_95_percentile",
            "predicted_top28",
            "ground_truth_label"
        ])

        for idx in sorted_indices.tolist():

            writer.writerow([
                idx,
                attribute_error[idx].item(),
                structural_error[idx].item(),
                normalized_attribute[idx].item(),
                normalized_structural[idx].item(),
                combined_scores[idx].item(),
                int(percentile_predictions[idx].item()),
                int(top28_predictions[idx].item()),
                int(y[idx].item())
            ])

    # --------------------------------------------------
    # 10. Save evaluation metrics
    # --------------------------------------------------
    metrics_path = os.path.join(
        output_dir,
        "books_evaluation_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["evaluation_method", "metric", "value"])

        for name, value in percentile_metrics.items():
            writer.writerow(["95th_percentile", name, value])

        for name, value in top28_metrics.items():
            writer.writerow(["top28_diagnostic", name, value])

    print("\nEvaluation completed successfully!")
    print("Anomaly scores saved:", csv_path)
    print("Evaluation metrics saved:", metrics_path)


if __name__ == "__main__":
    evaluate()
