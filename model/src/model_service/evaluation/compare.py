"""Aggregate and visualise results across benchmark runs.

Usage (notebook / script)
--------------------------
>>> from model_service.evaluation.compare import load_benchmark_summaries, generate_report
>>> df = load_benchmark_summaries(Path("artifacts/benchmarks"))
>>> generate_report(df, Path("artifacts/benchmarks"))   # saves all charts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# Colour palette — one colour per run_id, stable across all charts
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def build_color_map(summaries: "pd.DataFrame") -> dict[str, str]:
    """Return a stable {run_id: hex_color} mapping for every run in *summaries*.

    Colors are assigned by the order run_ids first appear in the dataframe
    (i.e. alphabetical, since ``load_benchmark_summaries`` sorts by path).
    The same run_id always gets the same color regardless of how individual
    charts sort their bars.
    """
    run_ids = list(dict.fromkeys(summaries["run_id"].tolist()))  # stable-unique
    return {rid: _PALETTE[i % len(_PALETTE)] for i, rid in enumerate(run_ids)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_benchmark_summaries(root: Path) -> "pd.DataFrame":
    """Read every ``summary.json`` under *root* into a flat DataFrame.

    Columns
    -------
    run_id, backbone, image_size, params_total, params_trainable_stage1,
    test_accuracy, test_precision, test_recall, test_f1,
    test_roc_auc, test_auc (alias), test_pr_auc,
    test_specificity, test_fnr,
    threshold_best_f1, threshold_high_recall_95, threshold_high_precision_95,
    epoch_time_s, total_train_s, inference_ms, max_train_samples
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas is required. pip install pandas")

    records = []
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        row: dict = {
            "run_id":                   data.get("run_id", summary_path.parent.name),
            "backbone":                 data.get("backbone", ""),
            "image_size":               data.get("image_size", 0),
            "params_total":             data.get("params_total"),
            "params_trainable_stage1":  data.get("params_trainable_stage1"),
            "_npz_path":                summary_path.parent / "test_predictions.npz",
        }

        for k, v in data.get("test", {}).items():
            row[f"test_{k}"] = v

        for k, v in data.get("thresholds", {}).items():
            row[f"threshold_{k}"] = v

        timing = data.get("timing", {})
        row["epoch_time_s"] = timing.get("epoch_time_s")
        row["total_train_s"] = timing.get("total_train_s")
        row["inference_ms"]  = timing.get("inference_ms_per_image")

        cfg = data.get("config", {})
        row["max_train_samples"] = cfg.get("max_train_samples")

        records.append(row)

    if not records:
        raise FileNotFoundError(f"No summary.json files found under {root}")

    df = pd.DataFrame(records)

    # Convenience alias: test_auc → test_roc_auc (both point to the same value)
    if "test_roc_auc" in df.columns and "test_auc" not in df.columns:
        df["test_auc"] = df["test_roc_auc"]

    return df


# ---------------------------------------------------------------------------
# Overlay curve plots  (require test_predictions.npz)
# ---------------------------------------------------------------------------

def _load_predictions(row: "pd.Series") -> tuple[np.ndarray, np.ndarray] | None:
    npz_path = row.get("_npz_path")
    if npz_path is None or not Path(npz_path).exists():
        print(f"  [skip] {row['run_id']}: test_predictions.npz not found")
        return None
    data = np.load(npz_path)
    return data["y_true"], data["y_prob"]


def plot_roc_overlay(
    summaries: "pd.DataFrame",
    *,
    out_path: Path | None = None,
    figsize: tuple[int, int] = (6, 6),
    color_map: dict[str, str] | None = None,
) -> None:
    """Overlay ROC curves for every run that has a ``test_predictions.npz``."""
    from sklearn.metrics import roc_curve

    cmap = color_map or build_color_map(summaries)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance (AUC=0.50)")

    for _, row in summaries.iterrows():
        preds = _load_predictions(row)
        if preds is None:
            continue
        y_true, y_prob = preds
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = row.get("test_roc_auc", row.get("test_auc"))
        label = f"{row['run_id']}  AUC={auc_val:.3f}" if isinstance(auc_val, float) else row["run_id"]
        ax.plot(fpr, tpr, lw=1.8, color=cmap[row["run_id"]], label=label)

    ax.set_xlabel("False Positive Rate (1 – Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — All Benchmark Runs")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _save_and_show(fig, out_path)


def plot_pr_overlay(
    summaries: "pd.DataFrame",
    *,
    out_path: Path | None = None,
    figsize: tuple[int, int] = (6, 6),
    color_map: dict[str, str] | None = None,
) -> None:
    """Overlay Precision-Recall curves for every run that has a ``test_predictions.npz``."""
    from sklearn.metrics import precision_recall_curve

    cmap = color_map or build_color_map(summaries)
    fig, ax = plt.subplots(figsize=figsize)

    for _, row in summaries.iterrows():
        preds = _load_predictions(row)
        if preds is None:
            continue
        y_true, y_prob = preds
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = row.get("test_pr_auc")
        label = f"{row['run_id']}  AP={pr_auc:.3f}" if isinstance(pr_auc, float) else row["run_id"]
        ax.plot(recall, precision, lw=1.8, color=cmap[row["run_id"]], label=label)

    # Baseline — random classifier
    pos_rate = 0.5
    ax.axhline(pos_rate, color="k", linestyle="--", lw=0.8, label=f"Chance (AP={pos_rate:.2f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curves — All Benchmark Runs")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

def plot_metric_bars(
    summaries: "pd.DataFrame",
    metric: str = "test_auc",
    *,
    out_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    lower_is_better: bool = False,
    color_map: dict[str, str] | None = None,
) -> None:
    """Horizontal bar chart comparing *metric* across all runs."""
    if not _HAS_PANDAS:
        raise ImportError("pandas required")
    if metric not in summaries.columns:
        available = [c for c in summaries.columns if not c.startswith("_")]
        raise ValueError(f"Column {metric!r} not found. Available: {available}")

    cmap = color_map or build_color_map(summaries)
    df = summaries[["run_id", metric]].dropna().sort_values(metric, ascending=not lower_is_better)
    n = len(df)
    h = figsize[1] if figsize else max(3, n * 0.5)
    fig, ax = plt.subplots(figsize=(figsize[0] if figsize else 8, h))

    colors = [cmap[rid] for rid in df["run_id"]]
    bars = ax.barh(df["run_id"], df[metric], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} — benchmark comparison")
    plt.tight_layout()
    _save_and_show(fig, out_path)


def plot_comparison_grid(
    summaries: "pd.DataFrame",
    metrics: list[str] | None = None,
    *,
    out_path: Path | None = None,
    color_map: dict[str, str] | None = None,
) -> None:
    """2-column grid of bar charts for a set of metrics."""
    if not _HAS_PANDAS:
        raise ImportError("pandas required")

    cmap = color_map or build_color_map(summaries)
    _lower_better = {"test_fnr", "inference_ms", "total_train_s"}

    if metrics is None:
        candidates = [
            "test_auc", "test_pr_auc", "test_f1",
            "test_recall", "test_specificity", "test_fnr",
        ]
        metrics = [c for c in candidates if c in summaries.columns]

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).ravel()

    for ax, metric in zip(axes, metrics):
        lower_better = metric in _lower_better
        df = summaries[["run_id", metric]].dropna().sort_values(
            metric, ascending=not lower_better
        )
        colors = [cmap[rid] for rid in df["run_id"]]
        bars = ax.barh(df["run_id"], df[metric], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        suffix = " ↓" if lower_better else " ↑"
        ax.set_title(metric + suffix, fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    plt.suptitle("PCam Benchmark — Metric Comparison Grid", fontsize=13, y=1.01)
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def plot_speed_scatter(
    summaries: "pd.DataFrame",
    x: str = "inference_ms",
    y: str = "test_auc",
    *,
    out_path: Path | None = None,
    figsize: tuple[int, int] = (7, 5),
    color_map: dict[str, str] | None = None,
) -> None:
    """Scatter plot showing the speed–accuracy trade-off across runs.

    Bubble size encodes model parameter count (larger = more params).
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas required")

    cmap = color_map or build_color_map(summaries)
    df = summaries[[c for c in [x, y, "run_id", "params_total"] if c in summaries.columns]].dropna(subset=[x, y])

    fig, ax = plt.subplots(figsize=figsize)

    max_params = df["params_total"].max() if "params_total" in df.columns else 1
    for _, row in df.iterrows():
        size = 200 * (row.get("params_total", max_params) / max_params) + 60
        ax.scatter(row[x], row[y], s=size, color=cmap[row["run_id"]],
                   alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(
            row["run_id"],
            (row[x], row[y]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#333333",
        )

    ax.set_xlabel(x + ("  ← faster" if "ms" in x or "time" in x else ""))
    ax.set_ylabel(y + ("  ↑ better" if "auc" in y or "f1" in y or "recall" in y else ""))
    ax.set_title("Speed vs. Performance Trade-off\n(bubble size = parameter count)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ---------------------------------------------------------------------------
# Full report generator
# ---------------------------------------------------------------------------

def generate_report(
    summaries: "pd.DataFrame",
    out_dir: Path,
    *,
    show: bool = False,
) -> None:
    """Generate and save every comparison chart to *out_dir*.

    Parameters
    ----------
    summaries:
        DataFrame from :func:`load_benchmark_summaries`.
    out_dir:
        Directory where charts are saved (created if needed).
    show:
        If True, display each chart inline (useful in notebooks).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    global _SHOW_PLOTS
    _SHOW_PLOTS = show

    # Build one stable color map here and pass it to every chart so that
    # the same run_id always renders in the same color across all plots.
    cmap = build_color_map(summaries)

    print(f"Saving comparison charts to {out_dir}/")

    plot_roc_overlay(summaries, out_path=out_dir / "roc_all.png", color_map=cmap)
    plot_pr_overlay(summaries, out_path=out_dir / "pr_all.png", color_map=cmap)
    plot_comparison_grid(summaries, out_path=out_dir / "metric_grid.png", color_map=cmap)
    plot_speed_scatter(
        summaries, x="inference_ms", y="test_auc",
        out_path=out_dir / "speed_vs_auc.png", color_map=cmap,
    )
    plot_speed_scatter(
        summaries, x="inference_ms", y="test_f1",
        out_path=out_dir / "speed_vs_f1.png", color_map=cmap,
    )
    plot_metric_bars(
        summaries, metric="inference_ms",
        title="Inference latency per image (ms)  ↓ lower is better",
        lower_is_better=True,
        out_path=out_dir / "inference_latency.png", color_map=cmap,
    )

    print("Done.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SHOW_PLOTS = False


def _save_and_show(fig: plt.Figure, out_path: Path | None) -> None:
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    if _SHOW_PLOTS:
        plt.show()
    plt.close(fig)
