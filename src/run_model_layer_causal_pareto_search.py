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


MODEL_LAYER_CANDIDATES: dict[str, dict[str, Any]] = {
    "baseline": {
        "args": {},
        "uses_risk_targets": False,
        "role": "control",
        "hypothesis": "No extra model-layer constraint.",
    },
    "prototype_guard": {
        "args": {
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": False,
        "role": "old_strong_candidate",
        "hypothesis": "Preserve the strong VT/VF boundary and macro-F1 signal from prototype constraints.",
    },
    "proto_center_only": {
        "args": {"prototype_center_weight": "0.02"},
        "uses_risk_targets": False,
        "role": "mechanism_verified_component",
        "hypothesis": (
            "Single-component mechanism verified by the 33-run ablation: prototype center compactness "
            "tests whether within-class embedding compactness drives the prototype benefit."
        ),
    },
    "proto_margin_only": {
        "args": {
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": False,
        "role": "mechanism_verified_component",
        "hypothesis": (
            "Single-component mechanism verified by the 33-run ablation: VT/VF prototype margin tests "
            "whether explicit class-center separation is sufficient on its own."
        ),
    },
    "proto_center_margin": {
        "args": {
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": False,
        "role": "mechanism_verified_component",
        "hypothesis": "Prototype center plus VT/VF margin component model from the mechanism-targeted ablation.",
    },
    "boundary_risk": {
        "args": {"boundary_ce_weight": "1.0"},
        "uses_risk_targets": True,
        "role": "old_strong_candidate",
        "hypothesis": "Preserve the strong accuracy, total-error, and migration-control signal from boundary risk weighting.",
    },
    "boundary075": {
        "args": {"boundary_ce_weight": "0.75"},
        "uses_risk_targets": True,
        "role": "mechanism_verified_component",
        "hypothesis": (
            "Boundary-risk component verified by the 33-run ablation; tests the lighter 0.75 dose before "
            "combining with representation constraints."
        ),
    },
    "boundary075_prototype": {
        "args": {
            "boundary_ce_weight": "0.75",
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": True,
        "role": "new_causal_pareto_recombination",
        "hypothesis": "Combine boundary weighting with prototype geometry using a slightly lighter boundary dose.",
    },
    "boundary100_prototype": {
        "args": {
            "boundary_ce_weight": "1.0",
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": True,
        "role": "new_causal_pareto_recombination",
        "hypothesis": "Directly combine the two strongest old constraints.",
    },
    "boundary075_prototype_reg": {
        "args": {
            "boundary_ce_weight": "0.75",
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
            "regularity_aux_weight": "0.02",
        },
        "uses_risk_targets": True,
        "role": "new_causal_pareto_recombination",
        "hypothesis": "Add light ECG rhythm/morphology regularity without the heavier full-supervisor burden.",
    },
    "boundary075_prototype_stability": {
        "args": {
            "boundary_ce_weight": "0.75",
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
            "stability_consistency_weight": "0.05",
            "embedding_consistency_weight": "0.01",
        },
        "uses_risk_targets": True,
        "role": "new_causal_pareto_recombination",
        "hypothesis": "Add light perturbation stability while avoiding the old heavy stability penalty.",
    },
    "boundary075_prototype_calibrated": {
        "args": {
            "boundary_ce_weight": "0.75",
            "risk_entropy_weight": "0.05",
            "anti_confident_risk_weight": "0.02",
            "prototype_center_weight": "0.02",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": True,
        "role": "new_causal_pareto_recombination",
        "hypothesis": "Borrow only the calibration part of complex RiskProReadable while keeping prototype/boundary guards.",
    },
    "boundary075_center": {
        "args": {
            "boundary_ce_weight": "0.75",
            "prototype_center_weight": "0.02",
        },
        "uses_risk_targets": True,
        "role": "mechanism_derived_candidate",
        "hypothesis": (
            "Mechanism-derived candidate from the 33-run ablation: combine the stable boundary dose "
            "with prototype compactness only, because proto_center_only was strong while proto_margin_only "
            "was weak as a standalone mechanism."
        ),
    },
    "boundary050_center": {
        "args": {
            "boundary_ce_weight": "0.50",
            "prototype_center_weight": "0.02",
        },
        "uses_risk_targets": True,
        "role": "mechanism_derived_sensitivity",
        "hypothesis": (
            "Lower boundary-dose sensitivity check for the mechanism-derived boundary plus center model."
        ),
    },
    "boundary100_center": {
        "args": {
            "boundary_ce_weight": "1.0",
            "prototype_center_weight": "0.02",
        },
        "uses_risk_targets": True,
        "role": "mechanism_derived_sensitivity",
        "hypothesis": (
            "Higher boundary-dose sensitivity check for boundary plus center; tests whether stronger "
            "boundary weighting reintroduces migration or VT/VF trade-offs."
        ),
    },
    "boundary075_margin": {
        "args": {
            "boundary_ce_weight": "0.75",
            "prototype_margin_weight": "0.05",
            "prototype_vtvf_margin": "1.0",
        },
        "uses_risk_targets": True,
        "role": "mechanism_derived_component",
        "hypothesis": (
            "Component test for whether the VT/VF prototype margin adds value when paired with boundary "
            "weighting, given that proto_margin_only was weak in the 33-run ablation."
        ),
    },
    "boundary075_contrastive": {
        "args": {
            "boundary_ce_weight": "0.75",
            "contrastive_weight": "0.02",
            "contrastive_boundary_anchor_weight": "2.0",
            "contrastive_vtvf_negative_weight": "2.0",
        },
        "uses_risk_targets": True,
        "role": "mechanism_derived_candidate",
        "hypothesis": (
            "Mechanism-derived candidate from the strong contrastive_vtvf_light result; tests whether "
            "boundary weighting combines better with contrastive local-purity control than prototype "
            "plus contrastive did."
        ),
    },
    "boundary075_center_calibrated": {
        "args": {
            "boundary_ce_weight": "0.75",
            "prototype_center_weight": "0.02",
            "risk_entropy_weight": "0.05",
            "anti_confident_risk_weight": "0.02",
        },
        "uses_risk_targets": True,
        "role": "mechanism_derived_candidate",
        "hypothesis": (
            "Calibration add-on for the boundary plus center model; tests whether lightweight entropy "
            "and anti-confident-risk terms improve ECE without sacrificing VT/VF safety outcomes."
        ),
    },
}


LOWER_IS_BETTER = {"ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"}
METRICS = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"]
MECHANISM_DERIVED_CANDIDATES = [
    "baseline",
    "boundary075",
    "proto_center_only",
    "proto_margin_only",
    "proto_center_margin",
    "boundary075_prototype",
    "boundary075_center",
    "boundary050_center",
    "boundary100_center",
    "boundary075_margin",
    "boundary075_contrastive",
    "boundary075_center_calibrated",
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
        if not metrics:
            continue
        metric_rows.append({**row.to_dict(), **metrics})
    run_level = pd.DataFrame(metric_rows)
    if run_level.empty:
        return {"n_run_level_rows": 0}
    run_level.to_csv(out / "model_layer_causal_pareto_search_run_level.csv", index=False)

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
        deltas.to_csv(out / "model_layer_causal_pareto_search_paired_effects.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for candidate, sub in deltas.groupby("candidate", sort=True):
        item: dict[str, Any] = {
            "candidate": candidate,
            "role": sub["role"].iloc[0],
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
    summary["selected_for_full_validation"] = (
        summary["is_pareto"] & summary["passes_basic_guard"] & summary["role"].ne("old_strong_candidate")
    )
    summary = summary.sort_values(
        ["selected_for_full_validation", "is_pareto", "mean_good_objective_count", "macro_f1_delta_mean"],
        ascending=[False, False, False, False],
    )
    summary.to_csv(out / "model_layer_causal_pareto_search_summary.csv", index=False)
    return {
        "n_run_level_rows": int(len(run_level)),
        "n_paired_effect_rows": int(len(deltas)),
        "n_summary_rows": int(len(summary)),
        "n_pareto_rows": int(summary["is_pareto"].sum()),
        "selected_for_full_validation": summary[summary["selected_for_full_validation"]]["candidate"].tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    py = sys.executable
    manifest = out / f"model_layer_causal_pareto_search_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict[str, str]] = []
    candidates = args.candidates
    if candidates is None:
        candidates = MECHANISM_DERIVED_CANDIDATES if args.candidate_set == "mechanism-derived" else list(MODEL_LAYER_CANDIDATES)
    if "baseline" not in candidates:
        candidates = ["baseline", *candidates]

    for seed in args.seeds:
        baseline_run: Path | None = None
        risk_targets: Path | None = None
        for candidate in candidates:
            spec = MODEL_LAYER_CANDIDATES[candidate]
            suffix = f"causal_pareto_search_{candidate}_seed{seed}"
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
    config["candidate_definitions"] = MODEL_LAYER_CANDIDATES
    (out / "model_layer_causal_pareto_search_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "out": str(out),
        "manifest": str(manifest),
        "dry_run": bool(args.dry_run),
        "n_manifest_rows": len(rows),
        "candidates": candidates,
    }
    if not args.dry_run:
        report.update(_aggregate(out, manifest))
    (out / "model_layer_causal_pareto_search_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Search model-layer causal-Pareto recombinations of old ECG constraints.")
    parser.add_argument("--mat", type=Path, default=Path("RHYTHMS.mat"))
    parser.add_argument("--model", choices=["regularity_fusion", "reliability_gated_fusion"], default="reliability_gated_fusion")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--candidate-set",
        choices=["all", "mechanism-derived"],
        default="mechanism-derived",
        help="Use the mechanism-derived candidate family by default, or all registered candidates.",
    )
    parser.add_argument("--candidates", nargs="+", choices=list(MODEL_LAYER_CANDIDATES), default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--split-grouping", choices=["record", "duplicate_family"], default="record")
    parser.add_argument("--out", type=Path, default=Path("results/model_layer_causal_pareto_search_20260630"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
