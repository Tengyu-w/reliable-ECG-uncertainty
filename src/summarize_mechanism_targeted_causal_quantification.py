from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOWER_IS_BETTER_OUTCOMES = {"ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"}
OUTCOMES = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors", "error_migration_penalty"]

HIGHER_IS_BETTER_MECHANISMS = {
    "silhouette_full",
    "sr_vt_norm_dist",
    "sr_vf_norm_dist",
    "vt_vf_norm_dist",
    "local_purity_k_mean",
    "prob_margin_mean",
    "prototype_vtvf_ambiguity_auroc",
    "knn_vtvf_mix_auroc",
    "knn_label_entropy_any_error_auroc",
    "softmax_vtvf_ambiguity_auroc",
    "entropy_any_error_auroc",
    "low_margin_any_error_auroc",
    "boundary_score_any_error_auroc",
    "validity_gate_any_error_auroc",
    "gate_x_boundary_any_error_auroc",
    "boundary_score_vtvf_cross_auroc",
    "gate_x_boundary_vtvf_cross_auroc",
}
LOWER_IS_BETTER_MECHANISMS = {
    "davies_bouldin_full",
    "knn_distance_mean",
    "knn_label_entropy_mean",
    "knn_vtvf_mix_ventricular_mean",
    "prototype_vtvf_ambiguity_ventricular_mean",
    "entropy_mean",
    "softmax_vtvf_ambiguity_ventricular_mean",
    "error_local_purity_mean",
    "vtvf_error_knn_mix_mean",
}


TARGET_ALIASES = {
    "within_class_compactness": ["silhouette_full", "davies_bouldin_full", "local_purity_k_mean"],
    "silhouette_full": ["silhouette_full"],
    "local_purity_k_mean": ["local_purity_k_mean"],
    "prototype_vtvf_ambiguity": [
        "prototype_vtvf_ambiguity_ventricular_mean",
        "prototype_vtvf_ambiguity_auroc",
    ],
    "vt_vf_norm_dist": ["vt_vf_norm_dist"],
    "knn_label_entropy_mean": ["knn_label_entropy_mean"],
    "knn_vtvf_mix_ventricular_mean": ["knn_vtvf_mix_ventricular_mean"],
    "softmax_vtvf_ambiguity": [
        "softmax_vtvf_ambiguity_ventricular_mean",
        "softmax_vtvf_ambiguity_auroc",
    ],
    "prob_margin_mean": ["prob_margin_mean"],
    "entropy_mean": ["entropy_mean"],
    "validity_gate_any_error_auroc": ["validity_gate_any_error_auroc"],
    "boundary_score_any_error_auroc": ["boundary_score_any_error_auroc"],
    "gate_x_boundary_any_error_auroc": ["gate_x_boundary_any_error_auroc"],
    "macro_f1": [],
    "vtvf_cross_errors": [],
    "regularity_feature_alignment": [],
    "atypical_signal_error": [],
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _metric_good_direction(metric: str, delta: float) -> bool | None:
    if not np.isfinite(delta):
        return None
    if metric in HIGHER_IS_BETTER_MECHANISMS:
        return delta > 0
    if metric in LOWER_IS_BETTER_MECHANISMS:
        return delta < 0
    return None


def _outcome_good_direction(outcome: str, delta: float, eps: float = 1e-12) -> bool | None:
    if not np.isfinite(delta) or abs(delta) <= eps:
        return None
    if outcome in LOWER_IS_BETTER_OUTCOMES:
        return delta < 0
    return delta > 0


def _target_metrics(targets: list[str], available: set[str]) -> list[str]:
    metrics: list[str] = []
    for target in targets:
        aliases = TARGET_ALIASES.get(target, [target])
        for alias in aliases:
            if alias in available and alias not in metrics:
                metrics.append(alias)
    return metrics


def _candidate_verdict(
    candidate: str,
    mech_effects: pd.DataFrame,
    outcome_effects: pd.DataFrame,
) -> dict[str, Any]:
    sub = mech_effects[mech_effects["candidate"].eq(candidate)].copy()
    out = outcome_effects[outcome_effects["candidate"].eq(candidate)].copy()
    role = str(sub["role"].iloc[0]) if "role" in sub and not sub.empty else ""
    target_mechanism = str(sub["target_mechanism"].iloc[0]) if "target_mechanism" in sub and not sub.empty else ""
    target_variables = _safe_json_list(sub["target_variables_json"].iloc[0]) if "target_variables_json" in sub and not sub.empty else []

    available = set(sub["mechanism_variable"].astype(str)) if "mechanism_variable" in sub else set()
    target_metrics = _target_metrics(target_variables, available)
    target_rows = sub[sub["mechanism_variable"].isin(target_metrics)].copy()

    judged = []
    target_delta_parts = []
    for _, row in target_rows.iterrows():
        metric = str(row["mechanism_variable"])
        delta = float(row["mechanism_delta_mean"])
        good = _metric_good_direction(metric, delta)
        if good is not None:
            judged.append(bool(good))
        target_delta_parts.append(f"{metric}={delta:.4g}")

    outcome_good = 0
    outcome_judged = 0
    outcome_parts = []
    for outcome in OUTCOMES:
        match = out[out["outcome"].eq(outcome)]
        if match.empty:
            continue
        delta = float(match["outcome_delta_mean"].iloc[0])
        good = _outcome_good_direction(outcome, delta)
        if good is not None:
            outcome_judged += 1
            outcome_good += int(good)
        outcome_parts.append(f"{outcome}={delta:.4g}")

    n_outcomes = len(outcome_parts)
    n_targets = len(judged)
    target_good = int(sum(judged))
    target_changed = "not_measured" if n_targets == 0 else ("yes" if target_good >= max(1, int(np.ceil(n_targets / 2))) else "no")
    if n_outcomes == 0:
        outcome_improved = "not_available"
    elif outcome_judged == 0:
        outcome_improved = "no_effect"
    else:
        outcome_improved = "yes" if outcome_good >= max(1, int(np.ceil(outcome_judged / 2))) else "no"

    if target_changed == "yes" and outcome_improved == "yes":
        verdict = "core_candidate"
    elif target_changed == "yes" and outcome_good > 0:
        verdict = "auxiliary_or_tradeoff"
    elif target_changed == "not_measured" and outcome_improved == "yes":
        verdict = "outcome_only_needs_mechanism_measure"
    elif target_changed == "yes" and outcome_improved == "no":
        verdict = "mechanism_changed_but_outcome_tradeoff"
    else:
        verdict = "negative_or_unstable"

    n_paired_seeds = 0
    if not out.empty and "n_paired_seeds" in out.columns:
        n_paired_seeds = int(pd.to_numeric(out["n_paired_seeds"], errors="coerce").max())

    return {
        "candidate": candidate,
        "role": role,
        "target_mechanism": target_mechanism,
        "target_variables_json": json.dumps(target_variables, ensure_ascii=False),
        "matched_target_metrics": "; ".join(target_metrics),
        "target_mechanism_changed": target_changed,
        "target_good_metric_count": target_good,
        "target_judged_metric_count": n_targets,
        "outcome_improved": outcome_improved,
        "outcome_good_count": outcome_good,
        "outcome_judged_count": outcome_judged,
        "outcome_count": n_outcomes,
        "n_paired_seeds": n_paired_seeds,
        "verdict": verdict,
        "target_delta_summary": "; ".join(target_delta_parts),
        "outcome_delta_summary": "; ".join(outcome_parts),
    }


def _top_associations(associations: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    if associations.empty or "spearman_r" not in associations.columns:
        return pd.DataFrame()
    out = associations.copy()
    out["abs_spearman_r"] = pd.to_numeric(out["spearman_r"], errors="coerce").abs()
    out = out.sort_values("abs_spearman_r", ascending=False)
    return out.head(n)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._\n"
    view = df.loc[:, [col for col in columns if col in df.columns]].copy()
    return view.to_markdown(index=False)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    quant_dir = Path(args.quant_dir)
    out_dir = Path(args.out_dir) if args.out_dir else quant_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = quant_dir / "causal_mechanism_quantification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    mech_effects = _read_csv(quant_dir / "intervention_to_mechanism_effects.csv")
    outcome_effects = _read_csv(quant_dir / "intervention_to_outcome_effects.csv")
    associations = _read_csv(quant_dir / "mechanism_to_outcome_association.csv")

    candidates = sorted(set(mech_effects.get("candidate", pd.Series(dtype=str)).astype(str)))
    verdict_rows = [_candidate_verdict(candidate, mech_effects, outcome_effects) for candidate in candidates]
    verdict = pd.DataFrame(verdict_rows)
    verdict_path = out_dir / "mechanism_targeted_verdict_table.csv"
    verdict.to_csv(verdict_path, index=False)

    top_assoc = _top_associations(associations)
    top_assoc_path = out_dir / "top_mechanism_outcome_associations.csv"
    top_assoc.to_csv(top_assoc_path, index=False)

    n_seeds = report.get("n_seeds", "")
    n_candidates = report.get("n_candidates", "")
    n_associations = report.get("n_association_rows", 0)
    md = [
        "# Mechanism-targeted causal quantification summary",
        "",
        "This report summarizes internal paired do-intervention evidence. It is not external clinical validation.",
        "",
        "## Scope",
        "",
        f"- Quantification directory: `{quant_dir}`",
        f"- Seeds: `{n_seeds}`",
        f"- Candidates: `{n_candidates}`",
        f"- Mechanism-outcome association rows: `{n_associations}`",
        "",
        "## Verdict Table",
        "",
        _markdown_table(
            verdict,
            [
                "candidate",
                "target_mechanism",
                "target_mechanism_changed",
                "outcome_improved",
                "outcome_good_count",
                "outcome_count",
                "n_paired_seeds",
                "verdict",
            ],
        ),
        "",
        "## Target Mechanism Delta Summary",
        "",
        _markdown_table(
            verdict,
            ["candidate", "matched_target_metrics", "target_delta_summary"],
        ),
        "",
        "## Outcome Delta Summary",
        "",
        _markdown_table(
            verdict,
            ["candidate", "outcome_delta_summary"],
        ),
        "",
        "## Strongest Mechanism-Outcome Associations",
        "",
        _markdown_table(
            top_assoc,
            [
                "mechanism_variable",
                "outcome",
                "n_paired_candidate_seed_rows",
                "spearman_r",
                "spearman_p",
                "interpretation",
            ],
        ),
        "",
        "## Limitations",
        "",
        "- Evidence is paired and internal to this ECG dataset; it does not replace external validation.",
        "- Mechanism variables are measured post-training, so the path summary should be described as causal-style evidence rather than definitive mediation proof.",
        "- When fewer than four paired candidate-seed rows exist, mechanism-outcome association is intentionally not estimated.",
        "",
    ]
    md_path = out_dir / "mechanism_targeted_causal_quantification_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    result = {
        "quant_dir": str(quant_dir),
        "verdict_table": str(verdict_path),
        "top_associations": str(top_assoc_path),
        "markdown_summary": str(md_path),
        "n_verdict_rows": int(len(verdict)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mechanism-targeted causal quantification outputs.")
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
