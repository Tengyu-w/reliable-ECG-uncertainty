from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def _read_latest(results_dir: Path) -> Path:
    latest = results_dir / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"Expected latest run pointer at {latest}")
    return Path(latest.read_text(encoding="utf-8").strip())


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "seed",
        "run_dir",
        "epochs",
        "batch_size",
        "lr",
        "analysis_status",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one or more ECG models across multiple random seeds and save a manifest for "
            "mean/std aggregation. This script intentionally keeps the training logic in src.train."
        )
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn", "inception_time", "reliability_gated_fusion"],
        choices=[
            "cnn",
            "tcn",
            "resnet1d",
            "inception_time",
            "bigru",
            "regularity_fusion",
            "reliability_gated_fusion",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/multiseed"))
    parser.add_argument("--run-suffix", type=str, default="multiseed")
    parser.add_argument("--run-analysis", action="store_true", help="Run uncertainty/embedding analysis after each training run.")
    parser.add_argument("--skip-corruption", action="store_true", help="Skip slow corruption severity analysis if --run-analysis is used.")
    parser.add_argument("--skip-boundary", action="store_true", help="Skip boundary waveform extraction if --run-analysis is used.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / f"multiseed_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []

    for model in args.models:
        for seed in args.seeds:
            suffix = f"{args.run_suffix}_seed{seed}"
            train_cmd = [
                py,
                "-m",
                "src.train",
                "--mat",
                str(args.mat),
                "--model",
                model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--seed",
                str(seed),
                "--out",
                str(args.out),
                "--run-suffix",
                suffix,
            ]
            if args.max_windows_per_record is not None:
                train_cmd.extend(["--max-windows-per-record", str(args.max_windows_per_record)])
            _run(train_cmd, cwd=root, dry_run=args.dry_run)
            run_dir = args.out / f"DRY_RUN_{model}_{seed}" if args.dry_run else _read_latest(args.out)
            analysis_status = "not_requested"

            if args.run_analysis:
                analysis_cmd = [
                    py,
                    "-m",
                    "src.run_analysis_suite",
                    "--mat",
                    str(args.mat),
                    "--run-dir",
                    str(run_dir),
                    "--model",
                    model,
                ]
                if args.skip_corruption:
                    analysis_cmd.append("--skip-corruption")
                if args.skip_boundary:
                    analysis_cmd.append("--skip-boundary")
                _run(analysis_cmd, cwd=root, dry_run=args.dry_run)
                analysis_status = "dry_run" if args.dry_run else "completed"

            rows.append(
                {
                    "model": model,
                    "seed": str(seed),
                    "run_dir": str(run_dir),
                    "epochs": str(args.epochs),
                    "batch_size": str(args.batch_size),
                    "lr": str(args.lr),
                    "analysis_status": analysis_status,
                }
            )
            _write_manifest(manifest_path, rows)

    config = {
        "mat": str(args.mat),
        "models": args.models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_windows_per_record": args.max_windows_per_record,
        "run_analysis": args.run_analysis,
        "skip_corruption": args.skip_corruption,
        "skip_boundary": args.skip_boundary,
        "manifest": str(manifest_path),
    }
    (args.out / "multiseed_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
