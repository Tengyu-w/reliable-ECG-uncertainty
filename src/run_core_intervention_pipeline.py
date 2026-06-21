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


def _latest(results_dir: Path) -> Path:
    path = results_dir / "latest"
    if not path.exists():
        raise FileNotFoundError(f"Missing latest pointer: {path}")
    return Path(path.read_text(encoding="utf-8").strip())


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
    extra: dict[str, str],
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
    for key, value in extra.items():
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
        "stage",
        "seed",
        "model",
        "run_dir",
        "teacher_run_dir",
        "risk_targets",
        "epochs",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the narrowed core intervention pipeline: regularity feature injection, "
            "risk-aligned reliability distillation, and VT/VF prototype separation."
        )
    )
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument(
        "--model",
        choices=["regularity_fusion", "reliability_gated_fusion"],
        default="reliability_gated_fusion",
        help="Core pipeline uses a regularity-capable model so ECG feature injection is part of the main line.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--risk-head-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/core_interventions"))
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--skip-corruption", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"core_intervention_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []

    for seed in args.seeds:
        teacher_suffix = f"core_regularity_injection_seed{seed}"
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
                teacher_suffix,
                args.max_windows_per_record,
                extra={},
            ),
            cwd=root,
            dry_run=args.dry_run,
        )
        teacher_run = args.out / f"DRY_RUN_{teacher_suffix}" if args.dry_run else _latest(args.out)
        if args.run_analysis:
            _run(
                _analysis_cmd(
                    py,
                    args.mat,
                    teacher_run,
                    args.model,
                    args.skip_corruption,
                    args.skip_boundary,
                    args.skip_stability,
                ),
                cwd=root,
                dry_run=args.dry_run,
            )

        risk_targets = teacher_run / "risk_targets.npz"
        _run(
            [
                py,
                "-m",
                "src.generate_risk_targets",
                "--teacher-run-dir",
                str(teacher_run),
                "--out",
                str(risk_targets),
            ],
            cwd=root,
            dry_run=args.dry_run,
        )
        rows.append(
            {
                "stage": "regularity_feature_injection_teacher",
                "seed": str(seed),
                "model": args.model,
                "run_dir": str(teacher_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": str(risk_targets),
                "epochs": str(args.epochs),
                "notes": "Regularity-capable teacher used for risk target generation.",
            }
        )
        _write_manifest(manifest, rows)

        risk_head_suffix = f"core_risk_distillation_seed{seed}"
        _run(
            [
                py,
                "-m",
                "src.train_embedding_risk_head",
                "--teacher-run-dir",
                str(teacher_run),
                "--risk-targets",
                str(risk_targets),
                "--epochs",
                str(args.risk_head_epochs),
                "--out",
                str(args.out),
                "--run-suffix",
                risk_head_suffix,
                "--seed",
                str(seed),
            ],
            cwd=root,
            dry_run=args.dry_run,
        )
        risk_head_run = args.out / f"DRY_RUN_{risk_head_suffix}" if args.dry_run else _latest(args.out)
        rows.append(
            {
                "stage": "risk_aligned_reliability_distillation",
                "seed": str(seed),
                "model": "embedding_risk_head",
                "run_dir": str(risk_head_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": str(risk_targets),
                "epochs": str(args.risk_head_epochs),
                "notes": "Distills entropy, KNN, local instability, VT/VF mixing, and softmax ambiguity.",
            }
        )
        _write_manifest(manifest, rows)

        prototype_suffix = f"core_prototype_separation_seed{seed}"
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
                prototype_suffix,
                args.max_windows_per_record,
                extra={
                    "prototype_center_weight": "0.02",
                    "prototype_margin_weight": "0.05",
                    "prototype_vtvf_margin": "1.0",
                },
            ),
            cwd=root,
            dry_run=args.dry_run,
        )
        prototype_run = args.out / f"DRY_RUN_{prototype_suffix}" if args.dry_run else _latest(args.out)
        if args.run_analysis:
            _run(
                _analysis_cmd(
                    py,
                    args.mat,
                    prototype_run,
                    args.model,
                    args.skip_corruption,
                    args.skip_boundary,
                    args.skip_stability,
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
        rows.append(
            {
                "stage": "prototype_separation",
                "seed": str(seed),
                "model": args.model,
                "run_dir": str(prototype_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": "",
                "epochs": str(args.epochs),
                "notes": "Training-time intervention for VT/VF embedding boundary ambiguity.",
            }
        )
        _write_manifest(manifest, rows)

    config = {
        "mat": str(args.mat),
        "model": args.model,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "risk_head_epochs": args.risk_head_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_windows_per_record": args.max_windows_per_record,
        "run_analysis": args.run_analysis,
        "skip_corruption": args.skip_corruption,
        "skip_boundary": args.skip_boundary,
        "skip_stability": args.skip_stability,
        "manifest": str(manifest),
    }
    (args.out / "core_intervention_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest}")


if __name__ == "__main__":
    main()
