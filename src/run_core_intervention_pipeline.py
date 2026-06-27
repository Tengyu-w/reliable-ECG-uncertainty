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


def _find_existing_run(results_dir: Path, suffix: str) -> Path | None:
    matches = [path for path in results_dir.glob(f"*_{suffix}") if path.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


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
        "--split-grouping",
        split_grouping,
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
            "risk-aligned reliability distillation, VT/VF prototype separation, "
            "and boundary-aware contrastive representation learning."
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
    parser.add_argument(
        "--split-grouping",
        choices=["record", "duplicate_family"],
        default="duplicate_family",
        help="Use the stricter duplicate-family split for final reliability claims.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/core_interventions"))
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed stage directories with matching suffixes instead of retraining them.",
    )
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
        teacher_run = None if args.dry_run or not args.reuse_existing else _find_existing_run(args.out, teacher_suffix)
        if teacher_run is None:
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
                    args.split_grouping,
                    extra={},
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
            teacher_run = args.out / f"DRY_RUN_{teacher_suffix}" if args.dry_run else _latest(args.out)
        else:
            print(f"\n$ reuse {teacher_run}", flush=True)
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

        risk_pro_plus_targets = teacher_run / "risk_pro_plus_targets.npz"
        _run(
            [
                py,
                "-m",
                "src.generate_risk_targets",
                "--teacher-run-dir",
                str(teacher_run),
                "--out",
                str(risk_pro_plus_targets),
                "--entropy-weight",
                "0.25",
                "--msp-weight",
                "0.10",
                "--local-instability-weight",
                "0.20",
                "--vtvf-mixing-weight",
                "0.20",
                "--knn-weight",
                "0.15",
                "--prototype-weight",
                "0.05",
                "--boundary-weight",
                "0.05",
            ],
            cwd=root,
            dry_run=args.dry_run,
        )

        risk_head_suffix = f"core_risk_distillation_seed{seed}"
        risk_head_run = None if args.dry_run or not args.reuse_existing else _find_existing_run(args.out, risk_head_suffix)
        if risk_head_run is None:
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
        else:
            print(f"\n$ reuse {risk_head_run}", flush=True)
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
        prototype_run = None if args.dry_run or not args.reuse_existing else _find_existing_run(args.out, prototype_suffix)
        if prototype_run is None:
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
                    args.split_grouping,
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
        else:
            print(f"\n$ reuse {prototype_run}", flush=True)
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

        contrastive_suffix = f"core_boundary_contrastive_seed{seed}"
        contrastive_run = None if args.dry_run or not args.reuse_existing else _find_existing_run(args.out, contrastive_suffix)
        if contrastive_run is None:
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
                    contrastive_suffix,
                    args.max_windows_per_record,
                    args.split_grouping,
                    extra={
                        "contrastive_weight": "0.05",
                        "contrastive_temperature": "0.1",
                        "contrastive_boundary_anchor_weight": "2.0",
                        "contrastive_vtvf_negative_weight": "2.0",
                    },
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
            contrastive_run = args.out / f"DRY_RUN_{contrastive_suffix}" if args.dry_run else _latest(args.out)
        else:
            print(f"\n$ reuse {contrastive_run}", flush=True)
        if args.run_analysis:
            _run(
                _analysis_cmd(
                    py,
                    args.mat,
                    contrastive_run,
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
                "stage": "boundary_contrastive",
                "seed": str(seed),
                "model": args.model,
                "run_dir": str(contrastive_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": "",
                "epochs": str(args.epochs),
                "notes": "Training-time intervention for local VT/VF embedding neighborhood separation.",
            }
        )
        _write_manifest(manifest, rows)

        risk_pro_plus_suffix = f"core_risk_pro_plus_seed{seed}"
        risk_pro_plus_extra = {
            "risk_targets": str(risk_pro_plus_targets),
            "boundary_ce_weight": "0.75",
            "risk_entropy_weight": "0.10",
            "stability_consistency_weight": "0.10",
            "embedding_consistency_weight": "0.02",
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
            "contrastive_weight": "0.03",
            "contrastive_temperature": "0.1",
            "contrastive_boundary_anchor_weight": "2.0",
            "contrastive_vtvf_negative_weight": "2.0",
            "regularity_aux_weight": "0.03",
        }
        if args.model == "reliability_gated_fusion":
            risk_pro_plus_extra.update(
                {
                    "risk_boundary_weight": "0.30",
                    "risk_gate_weight": "0.10",
                }
            )
        risk_pro_plus_run = None if args.dry_run or not args.reuse_existing else _find_existing_run(args.out, risk_pro_plus_suffix)
        if risk_pro_plus_run is None:
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
                    risk_pro_plus_suffix,
                    args.max_windows_per_record,
                    args.split_grouping,
                    extra=risk_pro_plus_extra,
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
            risk_pro_plus_run = args.out / f"DRY_RUN_{risk_pro_plus_suffix}" if args.dry_run else _latest(args.out)
        else:
            print(f"\n$ reuse {risk_pro_plus_run}", flush=True)
        if args.run_analysis:
            _run(
                _analysis_cmd(
                    py,
                    args.mat,
                    risk_pro_plus_run,
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
                "stage": "risk_pro_plus",
                "seed": str(seed),
                "model": args.model,
                "run_dir": str(risk_pro_plus_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": str(risk_pro_plus_targets),
                "epochs": str(args.epochs),
                "notes": "Combines risk-weighted CE, risk entropy alignment, PRO, contrastive, consistency, and regularity auxiliary losses.",
            }
        )
        _write_manifest(manifest, rows)

        risk_pro_readable_suffix = f"core_risk_pro_readable_seed{seed}"
        risk_pro_readable_extra = {
            "risk_targets": str(risk_pro_plus_targets),
            "boundary_ce_weight": "0.55",
            "risk_entropy_weight": "0.15",
            "anti_confident_risk_weight": "0.08",
            "selective_stability_consistency_weight": "0.12",
            "selective_embedding_consistency_weight": "0.02",
            "prototype_center_weight": "0.01",
            "prototype_margin_weight": "0.03",
            "prototype_vtvf_margin": "1.0",
            "contrastive_weight": "0.02",
            "contrastive_temperature": "0.1",
            "contrastive_boundary_anchor_weight": "2.5",
            "contrastive_vtvf_negative_weight": "2.5",
            "regularity_aux_weight": "0.02",
            "vtvf_readability_weight": "0.06",
        }
        if args.model == "reliability_gated_fusion":
            risk_pro_readable_extra.update(
                {
                    "risk_boundary_weight": "0.25",
                    "risk_gate_weight": "0.08",
                }
            )
        risk_pro_readable_run = None if args.dry_run or not args.reuse_existing else _find_existing_run(
            args.out,
            risk_pro_readable_suffix,
        )
        if risk_pro_readable_run is None:
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
                    risk_pro_readable_suffix,
                    args.max_windows_per_record,
                    args.split_grouping,
                    extra=risk_pro_readable_extra,
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
            risk_pro_readable_run = args.out / f"DRY_RUN_{risk_pro_readable_suffix}" if args.dry_run else _latest(args.out)
        else:
            print(f"\n$ reuse {risk_pro_readable_run}", flush=True)
        if args.run_analysis:
            _run(
                _analysis_cmd(
                    py,
                    args.mat,
                    risk_pro_readable_run,
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
                "stage": "risk_pro_readable",
                "seed": str(seed),
                "model": args.model,
                "run_dir": str(risk_pro_readable_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": str(risk_pro_plus_targets),
                "epochs": str(args.epochs),
                "notes": "Risk-Pro with explicit VT/VF readability, selective low-risk stability, and anti-confident high-risk constraints.",
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
        "split_grouping": args.split_grouping,
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
