from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"}
METRICS = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"]


TARGETED_CANDIDATES: dict[str, dict[str, Any]] = {
    "baseline": {
        "args": {},
        "uses_risk_targets": False,
        "target_mechanism": "control",
        "role": "control",
        "target_variables": [],
        "hypothesis": "No extra mechanism-targeted intervention.",
    },
    "proto_center_only": {
        "args": {"prototype_center_weight": "0.02"},
        "uses_risk_targets": False,
        "target_mechanism": "prototype_compactness",
        "role": "m1_prototype",
        "target_variables": ["within_class_compactness", "silhouette_full", "local_purity_k_mean"],
        "hypothesis": "Test whether within-class prototype compactness alone improves representation reliability.",
    },
    "proto_margin_only": {
        "args": {"prototype_margin_weight": "0.05", "prototype_vtvf_margin": "1.0"},
        "uses_risk_targets": False,
        "target_mechanism": "prototype_vtvf_ambiguity",
        "role": "m1_prototype",
        "target_variables": ["prototype_vtvf_ambiguity", "vt_vf_norm_dist", "vtvf_cross_errors"],
        "hypothesis": "Test whether VT/VF prototype separation alone reduces boundary ambiguity.",
    },
    "proto_center_margin": {
        "args": {
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": False,
        "target_mechanism": "prototype_geometry",
        "role": "m1_prototype",
        "target_variables": ["silhouette_full", "local_purity_k_mean", "prototype_vtvf_ambiguity"],
        "hypothesis": "Re-test the prototype-only guard by separating compactness and VT/VF margin effects.",
    },
    "contrastive_vtvf_light": {
        "args": {
            "contrastive_weight": "0.02",
            "contrastive_boundary_anchor_weight": "2.0",
            "contrastive_vtvf_negative_weight": "2.0",
        },
        "uses_risk_targets": False,
        "target_mechanism": "knn_local_purity",
        "role": "m2_knn",
        "target_variables": ["local_purity_k_mean", "knn_label_entropy_mean", "knn_vtvf_mix_ventricular_mean"],
        "hypothesis": "Test whether a light boundary-aware contrastive objective improves local KNN purity.",
    },
    "embedding_consistency_light": {
        "args": {"embedding_consistency_weight": "0.01", "stability_consistency_weight": "0.0"},
        "uses_risk_targets": False,
        "target_mechanism": "embedding_neighborhood_stability",
        "role": "m2_knn",
        "target_variables": ["neighbor_jaccard_mean", "knn_label_entropy_mean", "local_purity_k_mean"],
        "hypothesis": "Test whether mild embedding consistency stabilizes local neighborhoods.",
    },
    "prototype_plus_contrastive": {
        "args": {
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
            "contrastive_weight": "0.02",
            "contrastive_boundary_anchor_weight": "2.0",
            "contrastive_vtvf_negative_weight": "2.0",
        },
        "uses_risk_targets": False,
        "target_mechanism": "prototype_knn_joint",
        "role": "m2_knn",
        "target_variables": ["prototype_vtvf_ambiguity", "local_purity_k_mean", "knn_vtvf_mix_ventricular_mean"],
        "hypothesis": "Test whether prototype geometry and local KNN purity provide complementary benefits.",
    },
    "boundary050": {
        "args": {"boundary_ce_weight": "0.50"},
        "uses_risk_targets": True,
        "target_mechanism": "softmax_boundary_ambiguity",
        "role": "m3_softmax_boundary",
        "target_variables": ["softmax_vtvf_ambiguity", "prob_margin_mean", "entropy_mean"],
        "hypothesis": "Low-dose boundary weighting for softmax VT/VF ambiguity.",
    },
    "boundary075": {
        "args": {"boundary_ce_weight": "0.75"},
        "uses_risk_targets": True,
        "target_mechanism": "softmax_boundary_ambiguity",
        "role": "m3_softmax_boundary",
        "target_variables": ["softmax_vtvf_ambiguity", "prob_margin_mean", "entropy_mean"],
        "hypothesis": "Test the successful boundary dose without prototype constraints.",
    },
    "boundary100": {
        "args": {"boundary_ce_weight": "1.0"},
        "uses_risk_targets": True,
        "target_mechanism": "softmax_boundary_ambiguity",
        "role": "m3_softmax_boundary",
        "target_variables": ["softmax_vtvf_ambiguity", "prob_margin_mean", "error_migration_penalty"],
        "hypothesis": "High-dose boundary weighting to test for boundary over-emphasis and error migration.",
    },
    "risk_entropy_light": {
        "args": {"risk_entropy_weight": "0.05"},
        "uses_risk_targets": True,
        "target_mechanism": "calibration_entropy_alignment",
        "role": "m3_softmax_boundary",
        "target_variables": ["entropy_mean", "entropy_any_error_auroc", "ece"],
        "hypothesis": "Test whether aligning entropy with risk targets improves calibration without boundary over-weighting.",
    },
    "gate_boundary_joint": {
        "args": {"risk_gate_weight": "0.05", "risk_boundary_weight": "0.05"},
        "uses_risk_targets": True,
        "target_mechanism": "validity_gate_boundary_alignment",
        "role": "m4_gate_validity",
        "target_variables": [
            "validity_gate_any_error_auroc",
            "boundary_score_any_error_auroc",
            "gate_x_boundary_any_error_auroc",
        ],
        "hypothesis": "Test whether gate and boundary heads become useful when aligned to risk targets.",
    },
    "boundary075_gate_joint": {
        "args": {"boundary_ce_weight": "0.75", "risk_gate_weight": "0.05", "risk_boundary_weight": "0.05"},
        "uses_risk_targets": True,
        "target_mechanism": "boundary_gate_interaction",
        "role": "m4_gate_validity",
        "target_variables": ["gate_x_boundary_any_error_auroc", "softmax_vtvf_ambiguity", "vtvf_cross_errors"],
        "hypothesis": "Test whether the successful boundary dose is further explained by gate-boundary alignment.",
    },
    "regularity_aux_medium": {
        "args": {"regularity_aux_weight": "0.02"},
        "uses_risk_targets": False,
        "target_mechanism": "waveform_regularity",
        "role": "m5_regularity",
        "target_variables": ["regularity_feature_alignment", "atypical_signal_error", "macro_f1"],
        "hypothesis": "Test whether regularity reconstruction helps preserve ECG waveform structure in embeddings.",
    },
    "boundary075_prototype": {
        "args": {
            "boundary_ce_weight": "0.75",
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": True,
        "target_mechanism": "boundary_prototype_joint",
        "role": "known_joint_candidate",
        "target_variables": [
            "silhouette_full",
            "local_purity_k_mean",
            "prototype_vtvf_ambiguity",
            "softmax_vtvf_ambiguity",
            "gate_x_boundary_any_error_auroc",
        ],
        "hypothesis": "Reproduce the successful joint intervention as the reference mechanism chain.",
    },
}


DEFAULT_CANDIDATES = [
    "baseline",
    "proto_center_only",
    "proto_margin_only",
    "proto_center_margin",
    "contrastive_vtvf_light",
    "prototype_plus_contrastive",
    "boundary050",
    "boundary075",
    "gate_boundary_joint",
    "regularity_aux_medium",
    "boundary075_prototype",
]


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
    *,
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
    candidate_args: dict[str, str],
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
        "--split-grouping",
        split_grouping,
    ]
    if max_windows_per_record is not None:
        cmd.extend(["--max-windows-per-record", str(max_windows_per_record)])
    if risk_targets is not None:
        cmd.extend(["--risk-targets", str(risk_targets)])
    for key, value in candidate_args.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    return cmd


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "seed",
        "model",
        "role",
        "target_mechanism",
        "target_variables_json",
        "run_dir",
        "teacher_run_dir",
        "risk_targets",
        "epochs",
        "candidate_args_json",
        "hypothesis",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {k: float(raw[k]) for k in ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors"] if k in raw}
    out["error_migration_penalty"] = (
        float(raw.get("vt_as_vf", 0.0))
        + float(raw.get("vf_as_vt", 0.0))
        + 0.5 * float(raw.get("sr_as_vt", 0.0))
        + 0.5 * float(raw.get("sr_as_vf", 0.0))
    )
    return out


def _aggregate(out: Path, manifest: Path) -> dict[str, Any]:
    rows = pd.read_csv(manifest)
    metric_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        if str(row["status"]) != "completed":
            continue
        metrics = _load_metrics(Path(str(row["run_dir"])))
        if metrics:
            metric_rows.append({**row.to_dict(), **metrics})
    run_level = pd.DataFrame(metric_rows)
    if run_level.empty:
        return {"n_run_level_rows": 0}
    run_level.to_csv(out / "mechanism_targeted_ablation_run_level.csv", index=False)

    baseline = run_level[run_level["candidate"].eq("baseline")].set_index("seed")
    delta_rows: list[dict[str, Any]] = []
    for _, row in run_level[~run_level["candidate"].eq("baseline")].iterrows():
        seed = row["seed"]
        if seed not in baseline.index:
            continue
        base = baseline.loc[seed]
        delta = {
            "candidate": row["candidate"],
            "seed": seed,
            "role": row["role"],
            "target_mechanism": row["target_mechanism"],
            "target_variables_json": row["target_variables_json"],
            "candidate_args_json": row["candidate_args_json"],
            "hypothesis": row["hypothesis"],
        }
        for metric in METRICS:
            delta[f"{metric}_baseline"] = base[metric]
            delta[f"{metric}_candidate"] = row[metric]
            delta[f"{metric}_delta"] = row[metric] - base[metric]
        delta_rows.append(delta)
    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        deltas.to_csv(out / "mechanism_targeted_ablation_paired_effects.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for candidate, sub in deltas.groupby("candidate", sort=True):
        item: dict[str, Any] = {
            "candidate": candidate,
            "role": sub["role"].iloc[0],
            "target_mechanism": sub["target_mechanism"].iloc[0],
            "target_variables_json": sub["target_variables_json"].iloc[0],
            "n_paired_seeds": int(sub["seed"].nunique()),
            "candidate_args_json": sub["candidate_args_json"].iloc[0],
            "hypothesis": sub["hypothesis"].iloc[0],
        }
        good_count = 0
        for metric in METRICS:
            col = f"{metric}_delta"
            values = pd.to_numeric(sub[col], errors="coerce")
            mean = float(values.mean())
            item[f"{col}_mean"] = mean
            item[f"{col}_std"] = float(values.std()) if values.notna().sum() > 1 else np.nan
            if metric in LOWER_IS_BETTER:
                item[f"{metric}_good_direction_n"] = int((values <= 0).sum())
                good_count += int(mean <= 0)
            else:
                item[f"{metric}_good_direction_n"] = int((values >= 0).sum())
                good_count += int(mean >= 0)
        item["mean_good_objective_count"] = good_count
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return {"n_run_level_rows": int(len(run_level)), "n_paired_effect_rows": 0}

    objective_vectors = []
    for _, row in summary.iterrows():
        vec = []
        for metric in METRICS:
            value = float(row[f"{metric}_delta_mean"])
            vec.append(-value if metric in LOWER_IS_BETTER else value)
        objective_vectors.append(vec)
    vectors = np.asarray(objective_vectors, dtype=float)
    pareto = np.ones(len(summary), dtype=bool)
    for i in range(len(summary)):
        for j in range(len(summary)):
            if i == j:
                continue
            if np.all(vectors[j] >= vectors[i]) and np.any(vectors[j] > vectors[i]):
                pareto[i] = False
                break
    summary["is_pareto"] = pareto
    summary["passes_basic_guard"] = (
        summary["accuracy_good_direction_n"].ge(np.ceil(summary["n_paired_seeds"] / 2))
        & summary["macro_f1_good_direction_n"].ge(np.ceil(summary["n_paired_seeds"] / 2))
        & summary["ece_good_direction_n"].ge(np.ceil(summary["n_paired_seeds"] / 2))
        & summary["vtvf_cross_errors_good_direction_n"].ge(np.ceil(summary["n_paired_seeds"] / 2))
        & summary["total_errors_good_direction_n"].ge(np.ceil(summary["n_paired_seeds"] / 2))
    )
    summary = summary.sort_values(
        ["is_pareto", "passes_basic_guard", "mean_good_objective_count", "macro_f1_delta_mean"],
        ascending=[False, False, False, False],
    )
    summary.to_csv(out / "mechanism_targeted_ablation_summary.csv", index=False)

    mechanism_summary = (
        summary.groupby("target_mechanism", sort=True)
        .agg(
            n_candidates=("candidate", "nunique"),
            n_pareto=("is_pareto", "sum"),
            n_basic_guard=("passes_basic_guard", "sum"),
            best_good_objective_count=("mean_good_objective_count", "max"),
            best_accuracy_delta=("accuracy_delta_mean", "max"),
            best_macro_f1_delta=("macro_f1_delta_mean", "max"),
            best_ece_delta=("ece_delta_mean", "min"),
            best_vtvf_cross_errors_delta=("vtvf_cross_errors_delta_mean", "min"),
            best_total_errors_delta=("total_errors_delta_mean", "min"),
        )
        .reset_index()
    )
    mechanism_summary.to_csv(out / "mechanism_targeted_ablation_by_mechanism.csv", index=False)
    return {
        "n_run_level_rows": int(len(run_level)),
        "n_paired_effect_rows": int(len(deltas)),
        "n_summary_rows": int(len(summary)),
        "n_pareto_rows": int(summary["is_pareto"].sum()),
        "n_mechanism_rows": int(len(mechanism_summary)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    py = sys.executable
    manifest = out / f"mechanism_targeted_ablation_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []
    candidates = list(args.candidates)
    if "baseline" not in candidates:
        candidates = ["baseline", *candidates]

    for seed in args.seeds:
        baseline_run: Path | None = None
        risk_targets: Path | None = None
        for candidate in candidates:
            spec = TARGETED_CANDIDATES[candidate]
            suffix = f"mechanism_targeted_{candidate}_seed{seed}"
            risk_for_candidate = risk_targets if spec["uses_risk_targets"] else None
            if candidate != "baseline" and spec["uses_risk_targets"] and risk_for_candidate is None:
                raise RuntimeError(f"Risk-target candidate {candidate} reached before baseline risk targets were generated.")
            _run(
                _train_cmd(
                    py,
                    mat=args.mat,
                    model=args.model,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    out=out,
                    suffix=suffix,
                    max_windows_per_record=args.max_windows_per_record,
                    split_grouping=args.split_grouping,
                    candidate_args=spec["args"],
                    risk_targets=risk_for_candidate,
                ),
                root,
                args.dry_run,
            )
            run_dir = out / f"DRY_RUN_{suffix}" if args.dry_run else _latest(out)
            if candidate == "baseline":
                baseline_run = run_dir
                risk_targets = run_dir / "risk_targets.npz"
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
            rows.append(
                {
                    "candidate": candidate,
                    "seed": str(seed),
                    "model": args.model,
                    "role": spec["role"],
                    "target_mechanism": spec["target_mechanism"],
                    "target_variables_json": json.dumps(spec["target_variables"], ensure_ascii=True),
                    "run_dir": str(run_dir),
                    "teacher_run_dir": str(baseline_run or ""),
                    "risk_targets": str(risk_for_candidate or risk_targets or ""),
                    "epochs": str(args.epochs),
                    "candidate_args_json": json.dumps(spec["args"], sort_keys=True),
                    "hypothesis": spec["hypothesis"],
                    "status": "dry_run" if args.dry_run else "completed",
                }
            )
            _write_manifest(manifest, rows)

    config = vars(args).copy()
    config["mat"] = str(args.mat)
    config["out"] = str(out)
    config["manifest"] = str(manifest)
    config["candidate_definitions"] = TARGETED_CANDIDATES
    (out / "mechanism_targeted_ablation_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "out": str(out),
        "manifest": str(manifest),
        "dry_run": bool(args.dry_run),
        "n_manifest_rows": len(rows),
        "candidates": candidates,
        "seeds": args.seeds,
    }
    if not args.dry_run:
        report.update(_aggregate(out, manifest))
    (out / "mechanism_targeted_ablation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mechanism-targeted causal-style ablations for ECG reliability.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--model", choices=["regularity_fusion", "reliability_gated_fusion"], default="reliability_gated_fusion")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--candidates", nargs="+", choices=list(TARGETED_CANDIDATES), default=DEFAULT_CANDIDATES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--split-grouping", choices=["record", "duplicate_family"], default="record")
    parser.add_argument("--out", type=Path, default=Path("results/mechanism_targeted_causal_ablation_20260630"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
