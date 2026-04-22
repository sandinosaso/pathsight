"""Plotting helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve


def plot_history(history, out_path: Path | None = None) -> None:
    """Plot loss and key metrics from Keras History."""
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h.get("loss", []), label="train")
    axes[0].plot(h.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(h.get("auc", []), label="train auc")
    axes[1].plot(h.get("val_auc", []), label="val auc")
    axes[1].set_title("AUC")
    axes[1].legend()
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: tuple[str, str], out_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="w" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    best_f1_threshold: float | None = None,
    out_path: Path | None = None,
) -> None:
    """Precision-Recall curve with the optimal-F1 operating point marked.

    The PR curve is the primary diagnostic plot for pathology classifiers because
    it directly captures the precision/recall trade-off without being distorted by
    the large number of true negatives in a near-balanced dataset.

    F1 iso-contours (dashed) are drawn at F1 = 0.5, 0.6, 0.7, 0.8, 0.9 so you
    can visually judge how far the model is from each target.
    """
    y_true = y_true.astype(np.int32).ravel()
    y_score = y_score.astype(np.float32).ravel()

    fig, ax = plt.subplots(figsize=(5, 5))

    # Main PR curve with average precision (PR-AUC) in the legend
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_score,
        ax=ax,
        name="Model",
        color="steelblue",
    )

    # F1 iso-contours
    recall_vals = np.linspace(0.01, 1.0, 300)
    for f1_target in (0.5, 0.6, 0.7, 0.8, 0.9):
        # P = F1 * R / (2R - F1)  — only defined where 2R > F1
        denom = 2 * recall_vals - f1_target
        with np.errstate(divide="ignore", invalid="ignore"):
            precision_iso = np.where(denom > 0, f1_target * recall_vals / denom, np.nan)
        mask = (precision_iso >= 0) & (precision_iso <= 1)
        ax.plot(
            recall_vals[mask],
            precision_iso[mask],
            linestyle="--",
            linewidth=0.8,
            color="gray",
            alpha=0.6,
        )
        # Label at low-recall end
        idx = np.argmax(mask) if mask.any() else None
        if idx is not None:
            ax.text(
                recall_vals[idx],
                precision_iso[idx] + 0.01,
                f"F1={f1_target}",
                fontsize=7,
                color="gray",
                va="bottom",
            )

    # Mark the threshold that maximises F1
    if best_f1_threshold is not None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        # Find the index closest to the best_f1_threshold
        idx = int(np.argmin(np.abs(thresholds - best_f1_threshold))) if len(thresholds) > 0 else None
        if idx is not None:
            f1_at_best = (
                2 * precisions[idx] * recalls[idx] / (precisions[idx] + recalls[idx])
                if (precisions[idx] + recalls[idx]) > 0
                else 0.0
            )
            ax.scatter(
                [recalls[idx]],
                [precisions[idx]],
                zorder=5,
                color="crimson",
                s=80,
                label=f"Best-F1 thr={best_f1_threshold:.2f}  F1={f1_at_best:.3f}",
            )
            ax.legend(fontsize=8, loc="lower left")

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
