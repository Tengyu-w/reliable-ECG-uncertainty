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


def _latest(out: Path) -> Path:
    latest = out / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"Missing latest pointer: {latest}")
    return Path(latest.read_text(encoding="utf-8").strip())


def _train_cmd(
    py: str,
    mat: Path,
    model: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    out: Path,
    suffix: str,
    max_windows_per_record: int | None,
    split_grouping: str,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "src.train",
        "--mat",
        str(mat),
        "--model",
        model,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--out",
        str(out),
        "--run-suffix",
        suffix,
        "--split-grouping",
        split_grouping,
    ]
    if max_windows_per_record is not None:
        cmd.extend(["--max-windows-per-record", str(max_windows_per_record)])
    return cmd


def _analysis_cmd(
    py: str,
    mat: Path,
    run_dir: Path,
    model: str,
    skip_corruption: bool,
    skip_boundary: bool,
    skip_stability: bool,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "src.run_analysis_suite",
        "--mat",
        str(mat),
        "--run-dir",
        str(run_dir),
        "--model",
        model,
    ]
    if skip_corruption:
        cmd.append("--skip-corruption")
    if skip_boundary:
        cmd.append("--skip-boundary")
    if skip_stability:
        cmd.append("--skip-stability")
    return cmd


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "seed",
        "model",
        "run_dir",
        "teacher_run_dir",
        "risk_targets",
        "epochs",
        "analysis_status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 3-seed validation matrix for the two non-prototype core interventions: "
            "regularity feature injection and risk-aligned reliability distillation."
        )
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--waveform-model", choices=["resnet1d", "cnn", "tcn"], default="resnet1d")
    parser.add_argument(
        "--regularity-model",
        choices=["regularity_fusion", "reliability_gated_fusion"],
        default="reliability_gated_fusion",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--risk-head-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument(
        "--split-grouping",
        choices=["record", "duplicate_family"],
        default="record",
    )
    parser.add_argument("--out", type=Path, default=Path("results/core_validation_matrix"))
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--skip-waveform", action="store_true")
    parser.add_argument("--skip-regularity", action="store_true")
    parser.add_argument("--skip-risk-head", action="store_true")
    parser.add_argument("--skip-corruption", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"core_validation_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []

    for seed in args.seeds:
        waveform_run = None
        if not args.skip_waveform:
            suffix = f"regularity_validation_{args.waveform_model}_seed{seed}"
            _run(
                _train_cmd(
                    py,
                    args.mat,
                    args.waveform_model,
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.out,
                    suffix,
                    args.max_windows_per_record,
                    args.split_grouping,
                ),
                root,
                args.dry_run,
            )
            waveform_run = args.out / f"DRY_RUN_{suffix}" if args.dry_run else _latest(args.out)
            if args.run_analysis:
                _run(
                    _analysis_cmd(
                        py,
                        args.mat,
                        waveform_run,
                        args.waveform_model,
                        args.skip_corruption,
                        args.skip_boundary,
                        args.skip_stability,
                    ),
                    root,
                    args.dry_run,
                )
            rows.append(
                {
                    "stage": "waveform_only_baseline",
                    "seed": str(seed),
                    "model": args.waveform_model,
                    "run_dir": str(waveform_run),
                    "teacher_run_dir": "",
                    "risk_targets": "",
                    "epochs": str(args.epochs),
                    "analysis_status": "completed" if args.run_analysis and not args.dry_run else ("dry_run" if args.dry_run else "not_requested"),
                    "notes": "Comparison baseline for regularity feature injection.",
                }
            )
            _write_manifest(manifest, rows)

        regularity_run = None
        if not args.skip_regularity:
            suffix = f"regularity_validation_{args.regularity_model}_seed{seed}"
            _run(
                _train_cmd(
                    py,
                    args.mat,
                    args.regularity_model,
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.out,
                    suffix,
                    args.max_windows_per_record,
                    args.split_grouping,
                ),
                root,
                args.dry_run,
            )
            regularity_run = args.out / f"DRY_RUN_{suffix}" if args.dry_run else _latest(args.out)
            if args.run_analysis:
                _run(
                    _analysis_cmd(
                        py,
                        args.mat,
                        regularity_run,
                        args.regularity_model,
                        args.skip_corruption,
                        args.skip_boundary,
                        args.skip_stability,
                    ),
                    root,
                    args.dry_run,
                )
            risk_targets = regularity_run / "risk_targets.npz"
            _run(
                [
                    py,
                    "-m",
                    "src.generate_risk_targets",
                    "--teacher-run-dir",
                    str(regularity_run),
                    "--out",
                    str(risk_targets),
                ],
                root,
                args.dry_run,
            )
            rows.append(
                {
                    "stage": "regularity_feature_injection",
                    "seed": str(seed),
                    "model": args.regularity_model,
                    "run_dir": str(regularity_run),
                    "teacher_run_dir": str(regularity_run),
                    "risk_targets": str(risk_targets),
                    "epochs": str(args.epochs),
                    "analysis_status": "completed" if args.run_analysis and not args.dry_run else ("dry_run" if args.dry_run else "not_requested"),
                    "notes": "Regularity-injected model used as teacher for risk distillation.",
                }
            )
            _write_manifest(manifest, rows)

            if not args.skip_risk_head:
                suffix = f"risk_distillation_{args.regularity_model}_seed{seed}"
                _run(
                    [
                        py,
                        "-m",
                        "src.train_embedding_risk_head",
                        "--teacher-run-dir",
                        str(regularity_run),
                        "--risk-targets",
                        str(risk_targets),
                        "--epochs",
                        str(args.risk_head_epochs),
                        "--out",
                        str(args.out),
                        "--run-suffix",
                        suffix,
                        "--seed",
                        str(seed),
                    ],
                    root,
                    args.dry_run,
                )
                risk_head_run = args.out / f"DRY_RUN_{suffix}" if args.dry_run else _latest(args.out)
                rows.append(
                    {
                        "stage": "risk_aligned_distillation",
                        "seed": str(seed),
                        "model": "embedding_risk_head",
                        "run_dir": str(risk_head_run),
                        "teacher_run_dir": str(regularity_run),
                        "risk_targets": str(risk_targets),
                        "epochs": str(args.risk_head_epochs),
                        "analysis_status": "not_applicable",
                        "notes": "Risk head trained from regularity-injected teacher embeddings.",
                    }
                )
                _write_manifest(manifest, rows)

    config = vars(args).copy()
    config["mat"] = str(args.mat)
    config["out"] = str(args.out)
    config["manifest"] = str(manifest)
    (args.out / "core_validation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest}")


if __name__ == "__main__":
    main()
