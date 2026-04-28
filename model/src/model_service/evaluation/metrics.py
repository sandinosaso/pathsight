"""Sklearn-based metrics from model predictions."""
from __future__ import annotations

import json
import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model_service.config import ModelServiceConfig

def calculate_and_save_metrics(history):
    config = ModelServiceConfig()

    # 1. Create unique timestamp for this specific run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    # 2. Extract best metrics (at the point where EarlyStopping restored weights)
    monitor = config.train.early_stop_monitor
    if config.train.early_stop_mode == 'max':
        best_idx = history.history[monitor].index(max(history.history[monitor]))
    else:
        best_idx = history.history[monitor].index(min(history.history[monitor]))

    final_metrics = {m: float(v[best_idx]) for m, v in history.history.items()}

    # 3. Calculate F1-Score manually
    p = final_metrics.get('val_precision', 0)
    r = final_metrics.get('val_recall', 0)
    final_metrics['val_f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    final_metrics['run_timestamp'] = timestamp

    # 4. Save to artifacts/metrics/
    json_path = config.paths.artifacts_metrics / f"metrics_{timestamp}.json"
    config.paths.artifacts_metrics.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"✅ Training Complete. Metrics saved to {json_path}")

    return final_metrics


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.int32).ravel()
    y_prob = y_prob.astype(np.float32).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    high_recall_target: float = 0.95,
    high_precision_target: float = 0.95,
) -> dict[str, float]:
    """Extended metrics for medical binary classification.

    Includes all metrics from ``compute_binary_metrics`` plus:

    - ``specificity``       True negative rate (TN / (TN + FP))
    - ``fnr``               False negative rate (FN / (FN + TP)); critical in cancer detection
    - ``pr_auc``            Area under the Precision-Recall curve
    - ``best_f1_threshold`` Decision threshold that maximises F1 on this dataset
    - ``high_recall_threshold``
        Smallest threshold where recall >= ``high_recall_target`` (catch-all mode)
    - ``high_precision_threshold``
        Smallest threshold where precision >= ``high_precision_target`` (low-FP mode)

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_prob:
        Predicted probabilities for the positive class.
    threshold:
        Default decision threshold used for accuracy / precision / recall / F1 / specificity / FNR.
    high_recall_target:
        Minimum recall for the high-recall threshold search (default 0.95).
    high_precision_target:
        Minimum precision for the high-precision threshold search (default 0.95).
    """
    y_true = y_true.astype(np.int32).ravel()
    y_prob = y_prob.astype(np.float32).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Threshold search via sklearn's precision_recall_curve
    # Returns arrays of (precision, recall, thresholds) where len(thresholds) = len(precision) - 1
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_f1_threshold = float(thresholds[np.argmax(f1_scores)]) if len(thresholds) > 0 else threshold

    # High-recall threshold: smallest threshold (most inclusive) where recall >= target
    high_recall_mask = recalls[:-1] >= high_recall_target
    high_recall_threshold = float(thresholds[high_recall_mask].min()) if high_recall_mask.any() else float(thresholds.min())

    # High-precision threshold: smallest threshold where precision >= target
    high_precision_mask = precisions[:-1] >= high_precision_target
    high_precision_threshold = float(thresholds[high_precision_mask].min()) if high_precision_mask.any() else float(thresholds.max())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": pr_auc,
        "specificity": specificity,
        "fnr": fnr,
        "best_f1_threshold": best_f1_threshold,
        f"high_recall_{int(high_recall_target * 100)}_threshold": high_recall_threshold,
        f"high_precision_{int(high_precision_target * 100)}_threshold": high_precision_threshold,
    }


def confusion_matrix_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_true = y_true.astype(np.int32).ravel()
    y_pred = (y_prob.astype(np.float32).ravel() >= threshold).astype(np.int32)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
