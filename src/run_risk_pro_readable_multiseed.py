from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def _latest(results_dir: Path) -> Path:
    latest = results_dir / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"Missing latest pointer: {latest}")
    return Path(latest.read_text(encoding="utf-8").strip())


def _find_run(search_dirs: list[Path], suffix: str) -> Path | None:
    matches: list[Path] = []
    for base in search_dirs:
        if base.exists():
            matches.extend(path for path in base.glob(f"*_{suffix}") if path.is_dir())
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
    for key, value in extra.items():
        cmd.extend([f"--{key.replace('_', '-')}", value])
    return cmd


def _risk_target_cmd(py: str, teacher_run: Path, out: Path) -> list[str]:
    return [
        py,
        "-m",
        "src.generate_risk_targets",
        "--teacher-run-dir",
        str(teacher_run),
        "--out",
        str(out),
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
    ]


def _analysis_cmds(py: str, mat: Path, run_dir: Path, model: str, seed: int) -> list[list[str]]:
    return [
        [py, "-m", "src.evaluate_uncertainty", "--run-dir", str(run_dir)],
        [py, "-m", "src.embedding_geometry_analysis", "--run-dir", str(run_dir)],
        [py, "-m", "src.ambiguity_analysis", "--run-dir", str(run_dir)],
        [py, "-m", "src.review_efficiency_analysis", "--run-dir", str(run_dir)],
        [
            py,
            "-m",
            "src.stability_aware_analysis",
            "--mat",
            str(mat),
            "--run-dir",
            str(run_dir),
            "--model",
            model,
            "--seed",
            str(seed),
        ],
        [py, "-m", "src.representation_mechanism_analysis", "--mat", str(mat), "--run-dir", str(run_dir), "--model", model],
    ]


def _is_complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in ["metrics.json", "embeddings_train.npz", "embeddings_val.npz", "embeddings_test.npz"])


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "stage", "model", "run_dir", "teacher_run_dir", "risk_targets", "epochs", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused multi-seed Risk-Pro-readable validation.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--model", choices=["regularity_fusion", "reliability_gated_fusion"], default="reliability_gated_fusion")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-grouping", choices=["record", "duplicate_family"], default="duplicate_family")
    parser.add_argument("--out", type=Path, default=Path("results/risk_pro_readable_10seed"))
    parser.add_argument("--reuse-dir", action="append", type=Path, default=[])
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    py = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    search_dirs = [args.out, *args.reuse_dir]
    manifest = args.out / f"risk_pro_readable_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []

    readable_extra_template = {
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
        "risk_boundary_weight": "0.25",
        "risk_gate_weight": "0.08",
    }

    for seed in args.seeds:
        teacher_suffix = f"core_regularity_injection_seed{seed}"
        teacher_run = _find_run(search_dirs, teacher_suffix)
        if teacher_run is None or not _is_complete(teacher_run):
            _run(
                _train_cmd(
                    py,
                    args.mat,
                    args.model,
                    seed,
                    args.teacher_epochs,
                    args.batch_size,
                    args.lr,
                    args.out,
                    teacher_suffix,
                    args.split_grouping,
                    extra={},
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
            teacher_run = args.out / f"DRY_RUN_{teacher_suffix}" if args.dry_run else _latest(args.out)
        else:
            print(f"\n$ reuse teacher {teacher_run}", flush=True)

        risk_targets = teacher_run / "risk_pro_plus_targets.npz"
        if args.dry_run or not risk_targets.exists():
            _run(_risk_target_cmd(py, teacher_run, risk_targets), cwd=root, dry_run=args.dry_run)

        rows.append(
            {
                "seed": str(seed),
                "stage": "teacher",
                "model": args.model,
                "run_dir": str(teacher_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": str(risk_targets),
                "epochs": str(args.teacher_epochs),
                "notes": "Regularity-injection teacher for seed-matched risk target generation.",
            }
        )

        readable_suffix = f"core_risk_pro_readable_seed{seed}"
        readable_run = _find_run(search_dirs, readable_suffix)
        if readable_run is None or not _is_complete(readable_run):
            readable_extra = {**readable_extra_template, "risk_targets": str(risk_targets)}
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
                    readable_suffix,
                    args.split_grouping,
                    extra=readable_extra,
                ),
                cwd=root,
                dry_run=args.dry_run,
            )
            readable_run = args.out / f"DRY_RUN_{readable_suffix}" if args.dry_run else _latest(args.out)
        else:
            print(f"\n$ reuse readable {readable_run}", flush=True)

        if not args.skip_analysis:
            missing_analysis = not all(
                (readable_run / name).exists()
                for name in [
                    "uncertainty_metrics.csv",
                    "ambiguity_summary.csv",
                    "review_efficiency_curves.csv",
                    "stability_summary.csv",
                    "mechanism_stable_confident_errors.csv",
                ]
            )
            if missing_analysis:
                for cmd in _analysis_cmds(py, args.mat, readable_run, args.model, seed):
                    _run(cmd, cwd=root, dry_run=args.dry_run)

        rows.append(
            {
                "seed": str(seed),
                "stage": "risk_pro_readable",
                "model": args.model,
                "run_dir": str(readable_run),
                "teacher_run_dir": str(teacher_run),
                "risk_targets": str(risk_targets),
                "epochs": str(args.epochs),
                "notes": "Risk-Pro with VT/VF readability, selective stability, and anti-confident-risk constraints.",
            }
        )
        _write_manifest(manifest, rows)

    print(f"\nWrote manifest: {manifest}")


if __name__ == "__main__":
    main()
