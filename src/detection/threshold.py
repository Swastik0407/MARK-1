
import torch


def calculate_threshold(scores, method="percentile", percentile=95.0):
    """
    Calculate an anomaly detection threshold.

    Args:
        scores: Tensor containing anomaly scores.
        method: "percentile" or "mean_std".
        percentile: Percentile used when method="percentile".

    Returns:
        Scalar threshold.
    """

    if scores.numel() == 0:
        raise ValueError("Anomaly scores cannot be empty.")

    if not torch.isfinite(scores).all():
        raise ValueError("Anomaly scores contain NaN or Inf.")

    if method == "percentile":
        if not 0 < percentile < 100:
            raise ValueError("Percentile must be between 0 and 100.")

        threshold = torch.quantile(
            scores.float(),
            percentile / 100.0
        )

    elif method == "mean_std":
        threshold = scores.mean() + scores.std(unbiased=False)

    else:
        raise ValueError(
            "Unsupported method. Choose 'percentile' or 'mean_std'."
        )

    return threshold


def predict_anomalies(scores, threshold):
    """
    Predict anomaly labels using the calculated threshold.

    Returns:
        Boolean tensor: True = anomaly, False = normal.
    """

    return scores > threshold
