#!/usr/bin/env python3
"""Run one or all PCam benchmark experiments defined in a YAML config file.

Usage
-----
# Run the full matrix (default: 20 000 balanced training samples):
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml

# Run a single experiment by run_id:
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --only mobilenetv3small_96

# Quick smoke-test with 4 000 samples:
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --max-samples 4000

# Full dataset (no limit):
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --max-samples 0

# Dry-run — print the configs that would be executed without training:
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --dry-run

The script continues executing remaining runs even if one fails, writing a
``run_benchmark_report.json`` at the end that lists successes and failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Ensure the model package is importable when invoked from the repo root.
_MODEL_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(_MODEL_SRC))

import yaml  # pyyaml — listed in model requirements


def _setup_gpu() -> None:
    """Detect available accelerators, enable memory growth, and log the result.

    On macOS with Apple Silicon, GPU access requires the ``tensorflow-metal``
    plug-in (``pip install tensorflow-metal``).  If no GPU is found the script
    still runs on CPU; this function just prints a clear diagnostic.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            # Memory growth prevents TF from allocating all unified memory up front,
            # which is important on M-series Macs where GPU and CPU share RAM.
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # device already initialised — growth must be set earlier
        print(f"GPU detected ({len(gpus)} device(s)): {[g.name for g in gpus]}")
        print("  Memory growth enabled — TF will allocate GPU memory incrementally.\n")
    else:
        print("WARNING: No GPU detected. Training will run on CPU only.")
        print("  On macOS / Apple Silicon install the Metal plug-in:")
        print("    pip install tensorflow-metal")
        print("  Then restart your terminal / kernel and rerun.\n")


def _load_configs(config_path: Path, only: str | None = None) -> list[dict]:
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list in {config_path}, got {type(raw)}")
    if only:
        raw = [r for r in raw if r.get("run_id") == only]
        if not raw:
            raise ValueError(f"No config found with run_id={only!r}.")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCam benchmark sweep runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to benchmarks.yaml")
    parser.add_argument("--only", type=str, default=None, help="Run only the experiment with this run_id")
    parser.add_argument("--data-dir", type=str, default=None, help="Override TFDS data directory")
    parser.add_argument("--no-download", action="store_true", help="Disable automatic dataset download")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without training")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit training to N balanced samples (50/50 per class). "
            "Val and test are set to N//5 each. "
            "Pass 0 to use the full dataset. "
            "Defaults to the RunConfig value (20 000)."
        ),
    )
    args = parser.parse_args()

    _setup_gpu()

    from model_service.training.experiments import RunConfig, run_benchmark

    raw_configs = _load_configs(args.config, only=args.only)

    # Resolve the CLI --max-samples override:
    #   None  → keep each RunConfig's own default (20 000)
    #   0     → use full dataset (set max_train_samples=None in RunConfig)
    #   N > 0 → override all runs with N
    cli_max_samples: int | None | str
    if args.max_samples is None:
        cli_max_samples = "keep"  # sentinel: don't override
    elif args.max_samples == 0:
        cli_max_samples = None    # use full dataset
    else:
        cli_max_samples = args.max_samples

    print(f"Loaded {len(raw_configs)} experiment(s) from {args.config}")
    if cli_max_samples == "keep":
        print(f"  max_train_samples: per-config default (RunConfig.max_train_samples)")
    elif cli_max_samples is None:
        print(f"  max_train_samples: FULL DATASET (no limit)")
    else:
        print(f"  max_train_samples: {cli_max_samples} (balanced, CLI override)")
    print()

    if args.dry_run:
        for rc in raw_configs:
            print(f"  {rc['run_id']}: backbone={rc['backbone']} image_size={rc['image_size']}")
        return

    results: list[dict] = []
    sweep_start = time.perf_counter()

    for raw in raw_configs:
        run_id = raw.get("run_id", "unknown")
        cfg = RunConfig(**{k: v for k, v in raw.items() if not k.startswith("#")})

        # Apply CLI override when requested
        if cli_max_samples != "keep":
            cfg.max_train_samples = cli_max_samples  # type: ignore[assignment]

        t0 = time.perf_counter()
        try:
            summary = run_benchmark(cfg, data_dir=args.data_dir, download=not args.no_download)
            elapsed = time.perf_counter() - t0
            results.append({
                "run_id": run_id,
                "status": "success",
                "elapsed_s": round(elapsed, 1),
                "max_train_samples": cfg.max_train_samples,
                "test_auc": summary.get("test", {}).get("roc_auc"),
                "test_recall": summary.get("test", {}).get("recall"),
            })
            print(f"\n✓ {run_id} completed in {elapsed/60:.1f} min")
        except Exception:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            print(f"\n✗ {run_id} FAILED after {elapsed:.1f}s:\n{tb}")
            results.append({
                "run_id": run_id,
                "status": "failed",
                "elapsed_s": round(elapsed, 1),
                "error": tb,
            })

    sweep_elapsed = time.perf_counter() - sweep_start
    print(f"\n{'='*60}")
    print(f"Sweep complete in {sweep_elapsed/60:.1f} min")
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]
    print(f"  {len(successes)}/{len(results)} succeeded")
    if failures:
        print(f"  Failed runs: {[r['run_id'] for r in failures]}")

    report_path = args.config.parent / "run_benchmark_report.json"
    report_path.write_text(json.dumps({"results": results, "elapsed_s": round(sweep_elapsed, 1)}, indent=2))
    print(f"  Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
