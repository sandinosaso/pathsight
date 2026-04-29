#!/usr/bin/env python3
"""Generate a full comparison report from completed benchmark artifacts.

Usage
-----
# Default — reads artifacts/benchmarks/, saves charts to artifacts/benchmarks/comparison/
python model/scripts/compare_benchmarks.py

# Specify paths explicitly:
python model/scripts/compare_benchmarks.py \\
    --benchmarks-dir artifacts/benchmarks \\
    --out-dir artifacts/benchmarks/comparison

# Print leaderboard only (no charts):
python model/scripts/compare_benchmarks.py --table-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODEL_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(_MODEL_SRC))


# ─────────────────────────────────────────────────────────────────
# Leaderboard table
# ─────────────────────────────────────────────────────────────────

_DISPLAY_COLS = [
    ("run_id",          "Run"),
    ("backbone",        "Backbone"),
    ("image_size",      "Img"),
    ("test_auc",        "AUC ↑"),
    ("test_pr_auc",     "PR-AUC ↑"),
    ("test_f1",         "F1 ↑"),
    ("test_recall",     "Recall ↑"),
    ("test_specificity","Specificity ↑"),
    ("test_fnr",        "FNR ↓"),
    ("inference_ms",    "ms/img ↓"),
    ("total_train_s",   "Train(s)"),
    ("max_train_samples", "Samples"),
]


def _print_leaderboard(df: "pd.DataFrame") -> None:
    available = [(col, label) for col, label in _DISPLAY_COLS if col in df.columns]
    cols   = [c for c, _ in available]
    labels = {c: l for c, l in available}

    display = df[cols].copy().sort_values("test_auc", ascending=False)
    display = display.rename(columns=labels)

    # Format floats
    float_fmt = {l: "{:.4f}" for c, l in available if df[c].dtype == float and c not in ("image_size",)}
    for col, fmt in float_fmt.items():
        if col in display.columns:
            try:
                display[col] = display[col].map(lambda v: fmt.format(v) if v == v else "—")
            except Exception:
                pass

    sep = "─" * 130
    print(f"\n{'PCam Benchmark Leaderboard':^130}")
    print(sep)
    print(display.to_string(index=False))
    print(sep)


def _print_recommendation(df: "pd.DataFrame") -> None:
    print("\n🏆  RECOMMENDATIONS")
    print("─" * 70)

    if "test_auc" in df.columns:
        best_auc = df.loc[df["test_auc"].idxmax()]
        print(f"\n  Best overall (AUC):   {best_auc['run_id']}")
        print(f"    AUC={best_auc['test_auc']:.4f}  PR-AUC={best_auc.get('test_pr_auc', float('nan')):.4f}"
              f"  F1={best_auc.get('test_f1', float('nan')):.4f}"
              f"  Recall={best_auc.get('test_recall', float('nan')):.4f}"
              f"  FNR={best_auc.get('test_fnr', float('nan')):.4f}")

    if "test_recall" in df.columns:
        best_recall = df.loc[df["test_recall"].idxmax()]
        print(f"\n  Best sensitivity (Recall):  {best_recall['run_id']}")
        print(f"    Recall={best_recall['test_recall']:.4f}  FNR={best_recall.get('test_fnr', float('nan')):.4f}"
              f"  AUC={best_recall.get('test_auc', float('nan')):.4f}")

    if "inference_ms" in df.columns:
        fastest = df.loc[df["inference_ms"].idxmin()]
        print(f"\n  Fastest inference:          {fastest['run_id']}")
        print(f"    {fastest['inference_ms']:.1f} ms/image"
              f"  AUC={fastest.get('test_auc', float('nan')):.4f}")

    print()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PCam benchmark comparison report")
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("artifacts/benchmarks"),
        help="Directory containing per-run subdirectories with summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for charts (default: <benchmarks-dir>/comparison/)",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Print the leaderboard table only, skip chart generation",
    )
    args = parser.parse_args()

    benchmarks_dir = args.benchmarks_dir
    out_dir = args.out_dir or (benchmarks_dir / "comparison")

    from model_service.evaluation.compare import (
        generate_report,
        load_benchmark_summaries,
    )

    df = load_benchmark_summaries(benchmarks_dir)
    print(f"Loaded {len(df)} completed run(s) from {benchmarks_dir}")

    _print_leaderboard(df)
    _print_recommendation(df)

    if not args.table_only:
        generate_report(df, out_dir, show=False)
        print(f"\nAll charts saved to: {out_dir}/")


if __name__ == "__main__":
    main()
