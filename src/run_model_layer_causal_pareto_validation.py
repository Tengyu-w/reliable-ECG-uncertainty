from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_AUX_DIR = Path("results/auxiliary_intervention_matrix_20260606")
DEFAULT_OUT = Path("results/model_layer_causal_pareto_validation_20260629")

LOWER_IS_BETTER = {
    "ece",
    "vtvf_cross_errors",
    "total_errors",
    "vt_as_vf",
    "vf_as_vt",
    "sr_as_vt",
    "sr_as_vf",
    "error_migration_penalty",
}
METRICS = [
    "accuracy",
    "macro_f1",
    "ece",
    "vtvf_cross_errors",
    "total_errors",
    "vt_as_vf",
    "vf_as_vt",
    "sr_as_vt",
    "sr_as_vf",
]
OBJECTIVES = [
    "accuracy_delta",
    "macro_f1_delta",
    "ece_delta",
    "vtvf_cross_errors_delta",
    "total_errors_delta",
    "error_migration_penalty_delta",
]


VARIANT_SPEC: dict[str, dict[str, Any]] = {
    "baseline": {
        "model_name": "baseline_reliability_gated_fusion",
        "intervention": "do(no_extra_model_layer_intervention)",
        "intervenable_variables": [],
        "training_args": {},
    },
    "boundary_weighted": {
        "model_name": "causal_pareto_boundary_weighted",
        "intervention": "do(boundary_ce_weight=1.0)",
        "intervenable_variables": ["boundary_ce_weight"],
        "training_args": {"boundary_ce_weight": 1.0},
    },
    "stability_consistency": {
        "model_name": "causal_pareto_stability_consistency",
        "intervention": "do(stability_consistency_weight=0.2, embedding_consistency_weight=0.05)",
        "intervenable_variables": ["stability_consistency_weight", "embedding_consistency_weight"],
        "training_args": {"stability_consistency_weight": 0.2, "embedding_consistency_weight": 0.05},
    },
    "full_supervisor": {
        "model_name": "causal_pareto_full_supervisor",
        "intervention": (
            "do(boundary_ce_weight=1.0, stability_consistency_weight=0.2, "
            "embedding_consistency_weight=0.05, prototype_center_weight=0.02, "
            "prototype_margin_weight=0.05, regularity_aux_weight=0.05)"
        ),
        "intervenable_variables": [
            "boundary_ce_weight",
            "stability_consistency_weight",
            "embedding_consistency_weight",
            "prototype_center_weight",
            "prototype_margin_weight",
            "prototype_vtvf_margin",
            "regularity_aux_weight",
        ],
        "training_args": {
            "boundary_ce_weight": 1.0,
            "stability_consistency_weight": 0.2,
            "embedding_consistency_weight": 0.05,
            "prototype_center_weight": 0.02,
            "prototype_margin_weight": 0.05,
            "prototype_vtvf_margin": 1.0,
            "regularity_aux_weight": 0.05,
        },
    },
}


def _latest_manifest(aux_dir: Path) -> Path:
    manifests = sorted(aux_dir.glob("auxiliary_intervention_manifest_*.csv"))
    if not manifests:
        raise FileNotFoundError(f"No auxiliary intervention manifest found under {aux_dir}")
    return manifests[-1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_ci(values: np.ndarray, seed: int = 20260629, n_boot: int = 5000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _load_run_level(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_dir = Path(str(row["run_dir"]))
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _read_json(metrics_path)
        out = {
            "variant": row["variant"],
            "seed": int(row["seed"]),
            "model": row["model"],
            "run_dir": str(run_dir),
            "teacher_run_dir": row.get("teacher_run_dir", ""),
            "risk_targets": row.get("risk_targets", ""),
            "epochs": int(row.get("epochs", 0)),
        }
        for metric in METRICS:
            out[metric] = metrics.get(metric, np.nan)
        out["error_migration_penalty"] = (
            float(out.get("vt_as_vf", 0.0))
            + float(out.get("vf_as_vt", 0.0))
            + 0.5 * float(out.get("sr_as_vt", 0.0))
            + 0.5 * float(out.get("sr_as_vf", 0.0))
        )
        spec = VARIANT_SPEC.get(str(row["variant"]), {})
        out["model_name"] = spec.get("model_name", str(row["variant"]))
        out["intervention"] = spec.get("intervention", "")
        out["intervenable_variables"] = ";".join(spec.get("intervenable_variables", []))
        rows.append(out)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files referenced by {manifest_path}")
    return pd.DataFrame(rows)


def _paired_deltas(run_level: pd.DataFrame, baseline: str) -> pd.DataFrame:
    baseline_rows = run_level[run_level["variant"].eq(baseline)].set_index("seed")
    rows: list[dict[str, Any]] = []
    for _, row in run_level[~run_level["variant"].eq(baseline)].iterrows():
        seed = int(row["seed"])
        if seed not in baseline_rows.index:
            continue
        base = baseline_rows.loc[seed]
        out: dict[str, Any] = {
            "variant": row["variant"],
            "model_name": row["model_name"],
            "seed": seed,
            "baseline": baseline,
            "intervention": row["intervention"],
            "intervenable_variables": row["intervenable_variables"],
            "estimand": f"E[Y | do(model_layer_intervention={row['variant']})] - E[Y | do({baseline})]",
        }
        for metric in METRICS + ["error_migration_penalty"]:
            out[f"{metric}_baseline"] = base[metric]
            out[f"{metric}_intervention"] = row[metric]
            out[f"{metric}_delta"] = row[metric] - base[metric]
        rows.append(out)
    return pd.DataFrame(rows)


def _summarise_effects(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, sub in deltas.groupby("variant", sort=True):
        row: dict[str, Any] = {
            "variant": variant,
            "model_name": sub["model_name"].iloc[0],
            "n_paired_seeds": int(sub["seed"].nunique()),
            "intervention": sub["intervention"].iloc[0],
            "intervenable_variables": sub["intervenable_variables"].iloc[0],
        }
        stable_score = 0
        for objective in OBJECTIVES:
            values = pd.to_numeric(sub[objective], errors="coerce").to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            low, high = _bootstrap_ci(values)
            row[f"{objective}_mean"] = mean
            row[f"{objective}_bootstrap_ci_low"] = low
            row[f"{objective}_bootstrap_ci_high"] = high
            if objective.replace("_delta", "") in LOWER_IS_BETTER:
                good = values <= 0
                row[f"{objective}_good_direction_n"] = int(np.nansum(good))
                if mean <= 0:
                    stable_score += 1
            else:
                good = values >= 0
                row[f"{objective}_good_direction_n"] = int(np.nansum(good))
                if mean >= 0:
                    stable_score += 1
        row["mean_good_objective_count"] = stable_score
        rows.append(row)
    return pd.DataFrame(rows)


def _objective_vector(row: pd.Series) -> np.ndarray:
    values = []
    for objective in OBJECTIVES:
        value = float(row[f"{objective}_mean"])
        if objective.replace("_delta", "") in LOWER_IS_BETTER:
            value = -value
        values.append(value)
    return np.asarray(values, dtype=float)


def _pareto(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    vectors = np.vstack([_objective_vector(row) for _, row in summary.iterrows()])
    pareto_flags = np.ones(len(summary), dtype=bool)
    for i in range(len(summary)):
        for j in range(len(summary)):
            if i == j:
                continue
            dominates = np.all(vectors[j] >= vectors[i]) and np.any(vectors[j] > vectors[i])
            if dominates:
                pareto_flags[i] = False
                break
    out = summary.copy()
    out["is_pareto"] = pareto_flags
    out["weakly_improves_all_mean_objectives_vs_baseline"] = out["mean_good_objective_count"].eq(len(OBJECTIVES))
    out["passes_minimal_stability_guard"] = (
        out["accuracy_delta_good_direction_n"].ge(2)
        & out["macro_f1_delta_good_direction_n"].ge(2)
        & out["ece_delta_good_direction_n"].ge(2)
        & out["vtvf_cross_errors_delta_good_direction_n"].ge(2)
        & out["total_errors_delta_good_direction_n"].ge(2)
    )
    out["selected_model_layer_candidate"] = (
        out["is_pareto"]
        & out["weakly_improves_all_mean_objectives_vs_baseline"]
        & out["passes_minimal_stability_guard"]
    )
    return out.sort_values(
        ["selected_model_layer_candidate", "is_pareto", "mean_good_objective_count", "macro_f1_delta_mean"],
        ascending=[False, False, False, False],
    )


def _model_specs(pareto: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, spec in VARIANT_SPEC.items():
        row = {
            "variant": variant,
            "model_name": spec["model_name"],
            "intervention": spec["intervention"],
            "intervenable_variables": ";".join(spec["intervenable_variables"]),
            "training_args_json": json.dumps(spec["training_args"], ensure_ascii=False, sort_keys=True),
            "status_after_validation": "baseline",
        }
        if not pareto.empty and variant in set(pareto["variant"]):
            p = pareto[pareto["variant"].eq(variant)].iloc[0]
            if bool(p["selected_model_layer_candidate"]):
                row["status_after_validation"] = "selected_internal_model_layer_candidate"
            elif bool(p["is_pareto"]):
                row["status_after_validation"] = "pareto_tradeoff_candidate"
            else:
                row["status_after_validation"] = "not_selected"
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest is not None else _latest_manifest(Path(args.aux_dir))

    run_level = _load_run_level(manifest_path)
    deltas = _paired_deltas(run_level, args.baseline)
    effect_summary = _summarise_effects(deltas)
    pareto = _pareto(effect_summary)
    specs = _model_specs(pareto)

    run_level.to_csv(out / "model_layer_run_level_metrics.csv", index=False)
    deltas.to_csv(out / "model_layer_paired_causal_effects.csv", index=False)
    effect_summary.to_csv(out / "model_layer_effect_summary.csv", index=False)
    pareto.to_csv(out / "model_layer_pareto_selection.csv", index=False)
    specs.to_csv(out / "model_layer_candidate_model_specs.csv", index=False)

    selected = pareto[pareto["selected_model_layer_candidate"]].copy()
    report = {
        "out": str(out),
        "manifest": str(manifest_path),
        "baseline": args.baseline,
        "n_run_level_rows": int(len(run_level)),
        "n_paired_effect_rows": int(len(deltas)),
        "n_effect_summary_rows": int(len(effect_summary)),
        "n_pareto_rows": int(pareto["is_pareto"].sum()) if not pareto.empty else 0,
        "selected_model_layer_candidates": selected["variant"].tolist(),
        "interpretation_boundary": (
            "Model-only validation: classification, calibration, VT/VF errors, total errors, and error migration. "
            "No V5D routing/recover outcome is used for selecting model-layer candidates."
        ),
        "limitations": [
            "Internal paired-seed validation only.",
            "Only 3 paired seeds in the auxiliary intervention matrix.",
            "No external ECG validation set is available.",
            "Clinical or medical-device claims are not supported.",
        ],
    }
    (out / "model_layer_causal_pareto_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-only causal-Pareto validation for ECG classifier interventions.")
    parser.add_argument("--aux-dir", type=Path, default=DEFAULT_AUX_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--baseline", type=str, default="baseline")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
