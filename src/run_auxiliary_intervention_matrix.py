from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


AUXILIARY_VARIANTS = {
    "baseline": {},
    "boundary_weighted": {"boundary_ce_weight": "1.0"},
    "stability_consistency": {
        "stability_consistency_weight": "0.2",
        "embedding_consistency_weight": "0.05",
    },
    "full_supervisor": {
        "boundary_ce_weight": "1.0",
        "stability_consistency_weight": "0.2",
        "embedding_consistency_weight": "0.05",
        "prototype_center_weight": "0.02",
        "prototype_margin_weight": "0.05",
        "prototype_vtvf_margin": "1.0",
        "regularity_aux_weight": "0.05",
    },
}


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
    ]
    if max_windows_per_record is not None:
        cmd.extend(["--max-windows-per-record", str(max_windows_per_record)])
    if risk_targets is not None:
        cmd.extend(["--risk-targets", str(risk_targets)])
    for key, value in variant_args.items():
        cmd.extend([f"--{key.replace('_', '-')}", value])
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
        "variant",
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
            "Run the auxiliary intervention matrix for analysis/negative findings: "
            "boundary weighting, stability consistency, and full supervisor."
        )
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument(
        "--model",
        choices=["regularity_fusion", "reliability_gated_fusion"],
        default="reliability_gated_fusion",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(AUXILIARY_VARIANTS),
        default=list(AUXILIARY_VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/auxiliary_intervention_matrix"))
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--skip-corruption", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"auxiliary_intervention_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []

    for seed in args.seeds:
        baseline_run: Path | None = None
        risk_targets: Path | None = None

        if "baseline" in args.variants:
            suffix = f"aux_baseline_seed{seed}"
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
                    AUXILIARY_VARIANTS["baseline"],
                    None,
                ),
                root,
                args.dry_run,
            )
            baseline_run = args.out / f"DRY_RUN_{suffix}" if args.dry_run else _latest(args.out)
        else:
            raise ValueError("baseline must be included so risk targets are generated consistently per seed.")

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
            root,
            args.dry_run,
        )
        if args.run_analysis:
            _run(
                _analysis_cmd(
                    py,
                    args.mat,
                    baseline_run,
                    args.model,
                    args.skip_corruption,
                    args.skip_boundary,
                    args.skip_stability,
                ),
                root,
                args.dry_run,
            )
        rows.append(
            {
                "variant": "baseline",
                "seed": str(seed),
                "model": args.model,
                "run_dir": str(baseline_run),
                "teacher_run_dir": str(baseline_run),
                "risk_targets": str(risk_targets),
                "epochs": str(args.epochs),
                "analysis_status": "completed" if args.run_analysis and not args.dry_run else ("dry_run" if args.dry_run else "not_requested"),
                "notes": "Teacher run for per-seed risk-target generation.",
            }
        )
        _write_manifest(manifest, rows)

        for variant in args.variants:
            if variant == "baseline":
                continue
            suffix = f"aux_{variant}_seed{seed}"
            use_risk = "boundary_ce_weight" in AUXILIARY_VARIANTS[variant]
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
                    AUXILIARY_VARIANTS[variant],
                    risk_targets if use_risk else None,
                ),
                root,
                args.dry_run,
            )
            run_dir = args.out / f"DRY_RUN_{suffix}" if args.dry_run else _latest(args.out)
            if args.run_analysis:
                _run(
                    _analysis_cmd(
                        py,
                        args.mat,
                        run_dir,
                        args.model,
                        args.skip_corruption,
                        args.skip_boundary,
                        args.skip_stability,
                    ),
                    root,
                    args.dry_run,
                )
            rows.append(
                {
                    "variant": variant,
                    "seed": str(seed),
                    "model": args.model,
                    "run_dir": str(run_dir),
                    "teacher_run_dir": str(baseline_run),
                    "risk_targets": str(risk_targets) if use_risk else "",
                    "epochs": str(args.epochs),
                    "analysis_status": "completed" if args.run_analysis and not args.dry_run else ("dry_run" if args.dry_run else "not_requested"),
                    "notes": "Auxiliary intervention for robustness/negative-finding analysis.",
                }
            )
            _write_manifest(manifest, rows)

    config = vars(args).copy()
    config["mat"] = str(args.mat)
    config["out"] = str(args.out)
    config["manifest"] = str(manifest)
    (args.out / "auxiliary_intervention_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest}")


if __name__ == "__main__":
    main()
