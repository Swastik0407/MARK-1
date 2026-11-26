
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def calculate_metrics(y_true, anomaly_scores, y_pred):
    """
    Evaluate anomaly detection performance.

    Args:
        y_true: Ground-truth labels (True = anomaly).
        anomaly_scores: Continuous anomaly scores.
        y_pred: Predicted labels (True = anomaly).

    Returns:
        Dictionary of evaluation metrics.
    """

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    if not (
        len(y_true) == len(y_pred) == len(anomaly_scores)
    ):
        raise ValueError("All input arrays must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Evaluation arrays cannot be empty.")

    if not np.isfinite(anomaly_scores).all():
        raise ValueError("Anomaly scores contain NaN or Inf.")

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC-AUC requires both classes.")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, anomaly_scores),
        "pr_auc": average_precision_score(
            y_true, anomaly_scores
        ),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }
