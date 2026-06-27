from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VARIANTS = {
    "baseline": {},
    "boundary_weighted": {"boundary_ce_weight": "1.0"},
    "stability_consistency": {
        "stability_consistency_weight": "0.2",
        "embedding_consistency_weight": "0.05",
    },
    "reliability_guided": {
        "boundary_ce_weight": "1.0",
        "stability_consistency_weight": "0.2",
        "embedding_consistency_weight": "0.05",
    },
    "prototype_separation": {
        "prototype_center_weight": "0.02",
        "prototype_margin_weight": "0.05",
        "prototype_vtvf_margin": "1.0",
    },
    "boundary_contrastive": {
        "contrastive_weight": "0.05",
        "contrastive_temperature": "0.1",
        "contrastive_boundary_anchor_weight": "2.0",
        "contrastive_vtvf_negative_weight": "2.0",
    },
    "regularity_aux": {
        "regularity_aux_weight": "0.05",
    },
    "full_supervisor": {
        "boundary_ce_weight": "1.0",
        "stability_consistency_weight": "0.2",
        "embedding_consistency_weight": "0.05",
        "prototype_center_weight": "0.02",
        "prototype_margin_weight": "0.05",
        "prototype_vtvf_margin": "1.0",
        "contrastive_weight": "0.05",
        "contrastive_temperature": "0.1",
        "contrastive_boundary_anchor_weight": "2.0",
        "contrastive_vtvf_negative_weight": "2.0",
        "regularity_aux_weight": "0.05",
    },
}


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def _read_latest(results_dir: Path) -> Path:
    latest = results_dir / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"Expected latest run pointer at {latest}")
    return Path(latest.read_text(encoding="utf-8").strip())


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "model",
        "seed",
        "run_dir",
        "teacher_run_dir",
        "risk_targets",
        "epochs",
        "batch_size",
        "lr",
        "analysis_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    variant_args: dict[str, str],
    risk_targets: Path | None,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "src.train",
        "--mat",
        str(mat),
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--out",
        str(out),
        "--run-suffix",
        suffix,
        "--split-grouping",
        split_grouping,
    ]
    if max_windows_per_record is not None:
        cmd.extend(["--max-windows-per-record", str(max_windows_per_record)])
    if risk_targets is not None:
        cmd.extend(["--risk-targets", str(risk_targets)])
    for key, value in variant_args.items():
        cmd.extend([f"--{key.replace('_', '-')}", value])
    return cmd


def _analysis_cmd(py: str, mat: Path, run_dir: Path, model: str, skip_corruption: bool, skip_boundary: bool) -> list[str]:
    suite = [
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
        suite.append("--skip-corruption")
    if skip_boundary:
        suite.append("--skip-boundary")
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run reliability-guided mitigation variants. The baseline for each seed is used as the "
            "teacher that generates boundary/risk targets for boundary-weighted training."
        )
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument(
        "--model",
        choices=["cnn", "tcn", "resnet1d", "inception_time", "bigru", "regularity_fusion", "reliability_gated_fusion"],
        default="cnn",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument(
        "--split-grouping",
        choices=["record", "duplicate_family"],
        default="record",
    )
    parser.add_argument("--out", type=Path, default=Path("results/mitigation"))
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--skip-corruption", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"mitigation_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []

    for seed in args.seeds:
        baseline_suffix = f"mitigation_baseline_seed{seed}"
        _run(
            _train_cmd(
                py,
                args.mat,
                args.model,
                seed,
                args.epochs,
                args.batch_size,
                args.lr,
                args.out,
                baseline_suffix,
                args.max_windows_per_record,
                args.split_grouping,
                VARIANTS["baseline"],
                risk_targets=None,
            ),
            cwd=root,
            dry_run=args.dry_run,
        )
        baseline_run = args.out / f"DRY_RUN_{args.model}_{seed}_baseline" if args.dry_run else _read_latest(args.out)
        risk_targets = baseline_run / "risk_targets.npz"
        _run(
            [
                py,
                "-m",
                "src.generate_risk_targets",
                "--teacher-run-dir",
                str(baseline_run),
                "--out",
                str(risk_targets),
            ],
            cwd=root,
            dry_run=args.dry_run,
        )
        if args.run_analysis:
            _run(
                _analysis_cmd(py, args.mat, baseline_run, args.model, args.skip_corruption, args.skip_boundary),
                cwd=root,
                dry_run=args.dry_run,
            )
            analysis_status = "dry_run" if args.dry_run else "completed"
        else:
            analysis_status = "not_requested"
        rows.append(
            {
                "variant": "baseline",
                "model": args.model,
                "seed": str(seed),
                "run_dir": str(baseline_run),
                "teacher_run_dir": str(baseline_run),
                "risk_targets": str(risk_targets),
                "epochs": str(args.epochs),
                "batch_size": str(args.batch_size),
                "lr": str(args.lr),
                "analysis_status": analysis_status,
            }
        )
        _write_manifest(manifest, rows)

        for variant in args.variants:
            if variant == "baseline":
                continue
            if "regularity_aux_weight" in VARIANTS[variant] and args.model not in {"regularity_fusion", "reliability_gated_fusion"}:
                print(f"Skipping {variant}: regularity auxiliary requires a regularity-fusion model.")
                continue
            suffix = f"mitigation_{variant}_seed{seed}"
            _run(
                _train_cmd(
                    py,
                    args.mat,
                    args.model,
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.out,
                    suffix,
                    args.max_windows_per_record,
                    args.split_grouping,
                    VARIANTS[variant],
                    risk_targets=risk_targets if "boundary_ce_weight" in VARIANTS[variant] else None,
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
            run_dir = args.out / f"DRY_RUN_{args.model}_{seed}_{variant}" if args.dry_run else _read_latest(args.out)
            if args.run_analysis:
                _run(
                    _analysis_cmd(py, args.mat, run_dir, args.model, args.skip_corruption, args.skip_boundary),
                    cwd=root,
                    dry_run=args.dry_run,
                )
                analysis_status = "dry_run" if args.dry_run else "completed"
            else:
                analysis_status = "not_requested"
            rows.append(
                {
                    "variant": variant,
                    "model": args.model,
                    "seed": str(seed),
                    "run_dir": str(run_dir),
                    "teacher_run_dir": str(baseline_run),
                    "risk_targets": str(risk_targets) if "boundary_ce_weight" in VARIANTS[variant] else "",
                    "epochs": str(args.epochs),
                    "batch_size": str(args.batch_size),
                    "lr": str(args.lr),
                    "analysis_status": analysis_status,
                }
            )
            _write_manifest(manifest, rows)

    config = {
        "mat": str(args.mat),
        "model": args.model,
        "seeds": args.seeds,
        "variants": args.variants,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_windows_per_record": args.max_windows_per_record,
        "split_grouping": args.split_grouping,
        "run_analysis": args.run_analysis,
        "skip_corruption": args.skip_corruption,
        "skip_boundary": args.skip_boundary,
        "manifest": str(manifest),
    }
    (args.out / "mitigation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest}")


if __name__ == "__main__":
    main()
