from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "model",
                "run_dir",
                "severity_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run corruption-severity robustness validation on selected existing training runs."
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument(
        "--source-run-level",
        type=Path,
        default=Path("results/mitigation_v3_key_ablation_summary_full_analysis/mitigation_run_level_metrics.csv"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "prototype_separation"],
        help="Variants to select from the source run-level metrics CSV.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "tcn",
            "resnet1d",
            "inception_time",
            "bigru",
            "cnn_lstm",
            "regularity_fusion",
            "reliability_gated_fusion",
        ],
        default="reliability_gated_fusion",
    )
    parser.add_argument("--out", type=Path, default=Path("results/severity_validation"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    df = pd.read_csv(args.source_run_level)
    selected = df[df["variant"].isin(args.variants)].copy()
    if selected.empty:
        raise ValueError(f"No rows matched variants {args.variants} in {args.source_run_level}")
    selected = selected.sort_values(["variant", "seed"])

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"severity_validation_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []

    for item in selected.to_dict(orient="records"):
        run_dir = Path(str(item["run_dir"]))
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        seed = int(item["seed"])
        if args.skip_existing and (run_dir / "severity_monotonicity.csv").exists():
            status = "already_exists"
        else:
            _run(
                [
                    py,
                    "-m",
                    "src.evaluate_corruption_severity",
                    "--mat",
                    str(args.mat),
                    "--run-dir",
                    str(run_dir),
                    "--model",
                    args.model,
                    "--seed",
                    str(seed),
                ],
                root,
                args.dry_run,
            )
            _run([py, "-m", "src.monotonicity_analysis", "--run-dir", str(run_dir)], root, args.dry_run)
            status = "dry_run" if args.dry_run else "completed"
        rows.append(
            {
                "variant": str(item["variant"]),
                "seed": str(seed),
                "model": args.model,
                "run_dir": str(run_dir),
                "severity_status": status,
                "notes": "Corruption-severity robustness validation on an existing trained model.",
            }
        )
        _write_manifest(manifest, rows)

    print(f"\nWrote manifest: {manifest}")


if __name__ == "__main__":
    main()
