from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _float(row: pd.Series, key: str, default: float = np.nan) -> float:
    if key not in row or pd.isna(row[key]):
        return default
    return float(row[key])


def _int(row: pd.Series, key: str, default: int = 0) -> int:
    if key not in row or pd.isna(row[key]):
        return default
    return int(float(row[key]))


def _add_signal_row(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
    evidence_layer: str,
    mechanism_family: str,
    mechanism_variable: str,
    target: str,
    metric: str,
    mean: float,
    std: float = np.nan,
    n_seeds: int = 0,
    interpretation: str = "",
) -> None:
    rows.append(
        {
            "source_file": str(source_file),
            "evidence_layer": evidence_layer,
            "mechanism_family": mechanism_family,
            "mechanism_variable": mechanism_variable,
            "target": target,
            "metric": metric,
            "mean": mean,
            "std": std,
            "n_seeds": n_seeds,
            "interpretation": interpretation,
        }
    )


def _add_policy_row(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
    evidence_layer: str,
    intervention: str,
    mechanism_family: str,
    mechanism_variable: str,
    budget: float,
    outcome: str,
    mean: float,
    std: float = np.nan,
    n_seeds: int = 0,
    lower_is_better: bool = False,
    interpretation: str = "",
) -> None:
    rows.append(
        {
            "source_file": str(source_file),
            "evidence_layer": evidence_layer,
            "intervention": intervention,
            "mechanism_family": mechanism_family,
            "mechanism_variable": mechanism_variable,
            "budget": budget,
            "outcome": outcome,
            "mean": mean,
            "std": std,
            "n_seeds": n_seeds,
            "lower_is_better": lower_is_better,
            "interpretation": interpretation,
        }
    )


def _wavelet_evidence(root: Path, signal_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> None:
    audit = root / "evidence_informed_mechanism_routing_10seed_v4_20260627" / "wavelet_boundary_routing_audit"
    head_path = audit / "wavelet_risk_head_mean_std.csv"
    if _exists(head_path):
        df = pd.read_csv(head_path)
        for _, row in df.iterrows():
            score = str(row["score"])
            target = str(row["target"])
            _add_signal_row(
                signal_rows,
                source_file=head_path,
                evidence_layer="evidence_head",
                mechanism_family="wavelet_time_frequency",
                mechanism_variable=score,
                target=target,
                metric="test_auroc",
                mean=_float(row, "test_auroc_mean"),
                std=_float(row, "test_auroc_std"),
                n_seeds=_int(row, "n_seeds"),
                interpretation="Wavelet/time-frequency evidence head separates target errors.",
            )
            _add_signal_row(
                signal_rows,
                source_file=head_path,
                evidence_layer="evidence_head",
                mechanism_family="wavelet_time_frequency",
                mechanism_variable=score,
                target=target,
                metric="test_aupr",
                mean=_float(row, "test_aupr_mean"),
                std=_float(row, "test_aupr_std"),
                n_seeds=_int(row, "n_seeds"),
                interpretation="AUPR is sensitive to rare VT/VF boundary positives.",
            )

    policy_path = audit / "wavelet_boundary_policy_mean_std.csv"
    if _exists(policy_path):
        df = pd.read_csv(policy_path)
        for _, row in df.iterrows():
            method = str(row["method"])
            family = "wavelet_time_frequency" if "wavelet" in method else "mechanism_router"
            for outcome, lower in [
                ("all_error_addressed", False),
                ("vtvf_cross_error_addressed", False),
                ("automatic_unresolved_error_rate", True),
                ("automatic_unresolved_vtvf_cross_error_rate", True),
            ]:
                _add_policy_row(
                    policy_rows,
                    source_file=policy_path,
                    evidence_layer="routing_policy",
                    intervention=f"do(policy={method})",
                    mechanism_family=family,
                    mechanism_variable=method,
                    budget=_float(row, "budget"),
                    outcome=outcome,
                    mean=_float(row, f"{outcome}_mean"),
                    std=_float(row, f"{outcome}_std"),
                    n_seeds=_int(row, "n_seeds"),
                    lower_is_better=lower,
                    interpretation="Fixed-budget routing outcome under wavelet/mechanism policy.",
                )


def _mechanism_head_evidence(root: Path, signal_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> None:
    base = root / "evidence_informed_mechanism_routing_10seed_v4_20260627"
    head_path = base / "mechanism_head_mean_std.csv"
    if _exists(head_path):
        df = pd.read_csv(head_path)
        for _, row in df.iterrows():
            mech = str(row["mechanism"])
            for metric in ["test_auroc", "test_aupr"]:
                _add_signal_row(
                    signal_rows,
                    source_file=head_path,
                    evidence_layer="mechanism_risk_head",
                    mechanism_family="mechanism_specific_error_head",
                    mechanism_variable=mech,
                    target=mech,
                    metric=metric,
                    mean=_float(row, f"{metric}_mean"),
                    std=_float(row, f"{metric}_std"),
                    n_seeds=_int(row, "n_seeds"),
                    interpretation="Mechanism-specific risk head separates its target mechanism errors.",
                )

    policy_path = base / "optimized_mechanism_policy_mean_std_by_budget.csv"
    if _exists(policy_path):
        df = pd.read_csv(policy_path)
        for _, row in df.iterrows():
            for outcome, lower in [
                ("all_error_addressed", False),
                ("vtvf_cross_error_addressed", False),
                ("automatic_unresolved_error_rate", True),
                ("automatic_unresolved_vtvf_cross_error_rate", True),
            ]:
                _add_policy_row(
                    policy_rows,
                    source_file=policy_path,
                    evidence_layer="routing_policy",
                    intervention="do(policy=optimized_mechanism_router_v4)",
                    mechanism_family="mechanism_specific_error_head",
                    mechanism_variable="optimized_mechanism_router_v4",
                    budget=_float(row, "budget"),
                    outcome=outcome,
                    mean=_float(row, f"{outcome}_mean"),
                    std=_float(row, f"{outcome}_std"),
                    n_seeds=_int(row, "n_seeds"),
                    lower_is_better=lower,
                    interpretation="Optimized multi-mechanism routing outcome at fixed review budget.",
                )

    paired_path = base / "paired_optimized_vs_fixed_mechanism_policy_mean_std.csv"
    if _exists(paired_path):
        df = pd.read_csv(paired_path)
        for _, row in df.iterrows():
            for col in df.columns:
                if not col.endswith("_mean") or not col.startswith("delta_"):
                    continue
                outcome = col.removeprefix("delta_").removesuffix("_mean")
                _add_policy_row(
                    policy_rows,
                    source_file=paired_path,
                    evidence_layer="paired_policy_effect",
                    intervention="do(policy=optimized) - do(policy=fixed_mechanism)",
                    mechanism_family="mechanism_specific_error_head",
                    mechanism_variable="optimized_vs_fixed_mechanism",
                    budget=_float(row, "budget") if "budget" in row else np.nan,
                    outcome=outcome,
                    mean=_float(row, col),
                    std=_float(row, col.replace("_mean", "_std")),
                    n_seeds=_int(row, "n_seeds"),
                    lower_is_better="unresolved" in outcome or "error_rate" in outcome,
                    interpretation="Paired policy delta; sign depends on outcome direction.",
                )


def _explanation_evidence(root: Path, signal_rows: list[dict[str, Any]]) -> None:
    path = (
        root
        / "evidence_informed_mechanism_routing_10seed_v4_20260627"
        / "explanation_reliability_audit"
        / "explanation_alignment_mean_std.csv"
    )
    if not _exists(path):
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        family = str(row["evidence_family"])
        target = str(row["target_error_type"])
        for metric in ["auroc", "aupr", "top10_capture", "top20_capture", "top10_lift", "top20_lift"]:
            _add_signal_row(
                signal_rows,
                source_file=path,
                evidence_layer="explanation_alignment",
                mechanism_family="explanation_reliability",
                mechanism_variable=family,
                target=target,
                metric=metric,
                mean=_float(row, f"{metric}_mean"),
                std=_float(row, f"{metric}_std"),
                n_seeds=_int(row, "n_seeds"),
                interpretation="Explanation score alignment with the claimed target mechanism/error type.",
            )


def _regularity_evidence(root: Path, signal_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> None:
    summary = root / "core_validation_matrix_20260606" / "core_validation_summary"
    ablation = summary / "regularity_feature_ablation_mean_std.csv"
    if _exists(ablation):
        df = pd.read_csv(ablation)
        for _, row in df.iterrows():
            feature_set = str(row["feature_set"])
            model = str(row["model"])
            for metric, target in [
                ("vtvf_boundary_auroc", "vtvf_cross_error"),
                ("vtvf_boundary_aupr", "vtvf_cross_error"),
                ("error_auroc_entropy", "any_error"),
                ("accuracy", "classification"),
                ("macro_f1", "classification"),
                ("ece", "calibration"),
            ]:
                _add_signal_row(
                    signal_rows,
                    source_file=ablation,
                    evidence_layer="regularity_feature_ablation",
                    mechanism_family="regularity_waveform",
                    mechanism_variable=f"{feature_set}:{model}",
                    target=target,
                    metric=metric,
                    mean=_float(row, f"{metric}_mean"),
                    std=_float(row, f"{metric}_std"),
                    n_seeds=3,
                    interpretation="Regularity feature group ablation quantifies waveform-derived evidence.",
                )

    model_path = summary / "regularity_model_mean_std.csv"
    if _exists(model_path):
        df = pd.read_csv(model_path)
        for _, row in df.iterrows():
            stage = str(row["stage"])
            for metric in [
                "accuracy",
                "macro_f1",
                "ece",
                "vtvf_cross_errors",
                "vt_vf_norm_dist",
                "purity_k15_mean",
                "vtvf_mixing_k15_ventricular",
            ]:
                _add_signal_row(
                    signal_rows,
                    source_file=model_path,
                    evidence_layer="regularity_model_validation",
                    mechanism_family="regularity_waveform",
                    mechanism_variable=stage,
                    target="model_or_representation",
                    metric=metric,
                    mean=_float(row, f"{metric}_mean"),
                    std=_float(row, f"{metric}_std"),
                    n_seeds=_int(row, "n"),
                    interpretation="Model-level regularity validation and representation diagnostics.",
                )

    review_path = summary / "regularity_review_mean_std.csv"
    if _exists(review_path):
        df = pd.read_csv(review_path)
        for _, row in df.iterrows():
            stage = str(row["stage"])
            model = str(row["model"])
            score = str(row["score"])
            for outcome, lower in [
                ("all_error_captured", False),
                ("vtvf_error_captured", False),
                ("auto_error_rate", True),
                ("auto_vtvf_error_rate", True),
            ]:
                _add_policy_row(
                    policy_rows,
                    source_file=review_path,
                    evidence_layer="regularity_review_policy",
                    intervention=f"do(stage={stage}, review_score={score})",
                    mechanism_family="regularity_waveform",
                    mechanism_variable=f"{stage}:{model}:{score}",
                    budget=_float(row, "review_burden"),
                    outcome=outcome,
                    mean=_float(row, f"{outcome}_mean"),
                    std=_float(row, f"{outcome}_std"),
                    n_seeds=_int(row, "n"),
                    lower_is_better=lower,
                    interpretation="Regularity-related review score under fixed review burden.",
                )


def _evidence_inventory(signal_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    signal_sources = signal_df[["source_file", "evidence_layer", "mechanism_family"]].drop_duplicates()
    policy_sources = policy_df[["source_file", "evidence_layer", "mechanism_family"]].drop_duplicates()
    sources = pd.concat([signal_sources, policy_sources], ignore_index=True).drop_duplicates()
    rows = []
    for _, row in sources.iterrows():
        src = row["source_file"]
        rows.append(
            {
                "source_file": src,
                "evidence_layer": row["evidence_layer"],
                "mechanism_family": row["mechanism_family"],
                "n_signal_rows": int(signal_df["source_file"].eq(src).sum()) if not signal_df.empty else 0,
                "n_policy_rows": int(policy_df["source_file"].eq(src).sum()) if not policy_df.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _strongest_signal(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()
    auroc = signal_df[signal_df["metric"].isin(["test_auroc", "auroc", "vtvf_boundary_auroc"])].copy()
    auroc = auroc[pd.to_numeric(auroc["mean"], errors="coerce").notna()]
    return auroc.sort_values("mean", ascending=False).head(30)


def _policy_highlights(policy_df: pd.DataFrame) -> pd.DataFrame:
    if policy_df.empty:
        return pd.DataFrame()
    focus = policy_df[
        policy_df["outcome"].isin(
            [
                "vtvf_cross_error_addressed",
                "automatic_unresolved_vtvf_cross_error_rate",
                "all_error_addressed",
                "vtvf_error_captured",
                "auto_vtvf_error_rate",
            ]
        )
    ].copy()
    focus["sort_value"] = np.where(
        focus["lower_is_better"].astype(bool),
        -pd.to_numeric(focus["mean"], errors="coerce"),
        pd.to_numeric(focus["mean"], errors="coerce"),
    )
    return focus.sort_values(["outcome", "sort_value"], ascending=[True, False]).groupby("outcome").head(10)


def _variable_dictionary() -> pd.DataFrame:
    rows = [
        {
            "variable_type": "intervenable",
            "examples": "do(policy=v5_wavelet_boundary_router), do(review_score=vtvf_mixing), do(feature_set=all)",
            "meaning": "Evidence/policy/feature-family choice that can be changed experimentally.",
        },
        {
            "variable_type": "mechanism",
            "examples": "wavelet_vtvf_boundary_risk, representation_conflict, local_instability, boundary_explanation",
            "meaning": "Mechanism-specific evidence signal or explanation family.",
        },
        {
            "variable_type": "routing_outcome",
            "examples": "vtvf_cross_error_addressed, automatic_unresolved_vtvf_cross_error_rate, all_error_addressed",
            "meaning": "Fixed-budget routing or review outcome.",
        },
        {
            "variable_type": "signal_quality_outcome",
            "examples": "test_auroc, test_aupr, top10_capture, top20_lift",
            "meaning": "How well a mechanism signal aligns with its intended target.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified evidence chain inventory for historical mechanisms.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/mechanism_library_evidence_chain_20260630"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    signal_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    _wavelet_evidence(args.results_root, signal_rows, policy_rows)
    _mechanism_head_evidence(args.results_root, signal_rows, policy_rows)
    _explanation_evidence(args.results_root, signal_rows)
    _regularity_evidence(args.results_root, signal_rows, policy_rows)

    signal_df = pd.DataFrame(signal_rows)
    policy_df = pd.DataFrame(policy_rows)
    signal_df.to_csv(args.out / "mechanism_signal_strength_inventory.csv", index=False)
    policy_df.to_csv(args.out / "mechanism_policy_outcome_inventory.csv", index=False)
    inventory = _evidence_inventory(signal_df, policy_df)
    inventory.to_csv(args.out / "mechanism_evidence_source_inventory.csv", index=False)
    _strongest_signal(signal_df).to_csv(args.out / "strongest_mechanism_signals.csv", index=False)
    _policy_highlights(policy_df).to_csv(args.out / "mechanism_policy_highlights.csv", index=False)
    _variable_dictionary().to_csv(args.out / "mechanism_library_variable_dictionary.csv", index=False)

    family_summary = []
    if not signal_df.empty:
        for family, sub in signal_df.groupby("mechanism_family", sort=True):
            family_summary.append(
                {
                    "mechanism_family": family,
                    "n_signal_rows": int(len(sub)),
                    "n_unique_mechanism_variables": int(sub["mechanism_variable"].nunique()),
                    "n_unique_targets": int(sub["target"].nunique()),
                    "max_signal_mean": float(pd.to_numeric(sub["mean"], errors="coerce").max()),
                }
            )
    family_summary_df = pd.DataFrame(family_summary)
    family_summary_df.to_csv(args.out / "mechanism_family_signal_summary.csv", index=False)

    report = {
        "out": str(args.out),
        "n_signal_rows": int(len(signal_df)),
        "n_policy_rows": int(len(policy_df)),
        "n_source_files": int(inventory["source_file"].nunique()) if not inventory.empty else 0,
        "mechanism_families": sorted(set(signal_df.get("mechanism_family", [])) | set(policy_df.get("mechanism_family", []))),
        "outputs": {
            "signal_inventory": str(args.out / "mechanism_signal_strength_inventory.csv"),
            "policy_inventory": str(args.out / "mechanism_policy_outcome_inventory.csv"),
            "source_inventory": str(args.out / "mechanism_evidence_source_inventory.csv"),
            "strongest_signals": str(args.out / "strongest_mechanism_signals.csv"),
            "policy_highlights": str(args.out / "mechanism_policy_highlights.csv"),
            "variable_dictionary": str(args.out / "mechanism_library_variable_dictionary.csv"),
            "family_summary": str(args.out / "mechanism_family_signal_summary.csv"),
        },
        "limitations": [
            "This is a structured aggregation of existing result tables, not a new randomized experiment.",
            "Model-layer intervention evidence and routing-policy evidence are kept as separate layers.",
            "Regularity evidence partly comes from 3-seed validation, while routing/explanation/mechanism-head evidence is mostly 10-seed.",
        ],
    }
    (args.out / "mechanism_library_evidence_chain_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
