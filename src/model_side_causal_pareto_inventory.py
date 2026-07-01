from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("results/model_side_causal_pareto_inventory_20260629")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(df: pd.DataFrame, metric: str, col: str = "mean") -> float | None:
    if df.empty or "metric" not in df.columns or col not in df.columns:
        return None
    sub = df[df["metric"].astype(str).eq(metric)]
    if sub.empty:
        return None
    value = sub.iloc[0][col]
    return float(value) if pd.notna(value) else None


def _fmt(value: float | None, scale: float = 1.0, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return str(round(value * scale, digits))


def _build_variable_dictionary() -> pd.DataFrame:
    rows = [
        {
            "variable": "do(model_family)",
            "role": "intervention",
            "intervenable": True,
            "examples": "CNN; TCN; ResNet1D; InceptionTime; BiGRU; RegularityFusion; GatedFusion",
            "meaning": "Choose the base ECG classifier architecture.",
        },
        {
            "variable": "do(add_prototype_objective)",
            "role": "intervention",
            "intervenable": True,
            "examples": "prototype center loss; VT/VF prototype margin",
            "meaning": "Change embedding geometry through prototype separation objectives.",
        },
        {
            "variable": "do(add_risk_objective)",
            "role": "intervention",
            "intervenable": True,
            "examples": "risk entropy alignment; risk boundary loss; anti-confident risk penalty",
            "meaning": "Use learned or generated risk targets during training.",
        },
        {
            "variable": "do(add_validity_branch)",
            "role": "intervention",
            "intervenable": True,
            "examples": "validity gate; boundary score; gate x boundary",
            "meaning": "Add a model-side validity/domain branch.",
        },
        {
            "variable": "do(add_wavelet_boundary_branch)",
            "role": "intervention",
            "intervenable": True,
            "examples": "CNN-Wavelet-TCN boundary adapter; wavelet VT/VF risk head",
            "meaning": "Add time-frequency boundary evidence to the model.",
        },
        {
            "variable": "do(add_regularity_auxiliary)",
            "role": "intervention",
            "intervenable": True,
            "examples": "regularity feature fusion; regularity reconstruction head",
            "meaning": "Expose rhythm/morphology regularity evidence to the model.",
        },
        {
            "variable": "ECG morphology/rhythm",
            "role": "observed_context",
            "intervenable": False,
            "examples": "SR/VT/VF waveform structure; rhythm regularity",
            "meaning": "Signal content used for modeling and evaluation, not directly manipulated as a causal treatment.",
        },
        {
            "variable": "train/test split and duplicate family structure",
            "role": "observed_context_or_design",
            "intervenable": False,
            "examples": "record-level split; duplicate-family split",
            "meaning": "Evaluation design constraint; not a model improvement variable.",
        },
        {
            "variable": "Y: classification utility",
            "role": "outcome",
            "intervenable": False,
            "examples": "accuracy; macro-F1; total errors",
            "meaning": "Ordinary classification quality.",
        },
        {
            "variable": "Y: VT/VF reliability",
            "role": "outcome",
            "intervenable": False,
            "examples": "VT/VF cross-errors; VT/VF boundary AUROC",
            "meaning": "Whether the intervention improves the central high-risk boundary.",
        },
        {
            "variable": "Y: calibration and uncertainty",
            "role": "outcome",
            "intervenable": False,
            "examples": "ECE; entropy/MSP error AUROC; risk-head AUROC",
            "meaning": "Whether model confidence remains useful for reliability.",
        },
        {
            "variable": "Y: error migration",
            "role": "outcome",
            "intervenable": False,
            "examples": "SR->VT increase; VT->VF increase; hidden confident errors",
            "meaning": "Whether an intervention fixes one error family while creating another.",
        },
        {
            "variable": "Y_downstream: fixed-router coupling value",
            "role": "downstream_evaluation_outcome",
            "intervenable": False,
            "examples": "V5D(model=A) vs V5D(model=B), same router, same budget",
            "meaning": "Not a model-only outcome. Use only after model-only and evidence-head comparisons are separated.",
        },
    ]
    return pd.DataFrame(rows)


def _build_comparison_layer_dictionary() -> pd.DataFrame:
    rows = [
        {
            "comparison_layer": "model_only",
            "fair_comparison_unit": "trained classifier or classifier intervention",
            "allowed_comparison": "model(A) vs model(B)",
            "allowed_outcomes": "accuracy; macro-F1; ECE; VT/VF cross-errors; total errors; error migration; OOD classification robustness",
            "disallowed_outcomes": "V5D capture or recover metrics as direct proof that the classifier is better",
        },
        {
            "comparison_layer": "evidence_head_only",
            "fair_comparison_unit": "evidence signal/head",
            "allowed_comparison": "evidence_head(A) vs evidence_head(B)",
            "allowed_outcomes": "any-error AUROC; VT/VF boundary AUROC; calibration of risk score; ECG-shift stability",
            "disallowed_outcomes": "classifier accuracy or complete routing utility unless a classifier/router is fixed explicitly",
        },
        {
            "comparison_layer": "fixed_router_downstream",
            "fair_comparison_unit": "same complete router supplied by different model outputs or evidence heads",
            "allowed_comparison": "V5D(model=A) vs V5D(model=B), or router(A) vs router(B) with same base model",
            "allowed_outcomes": "capture; residual risk; review budget; recover action distribution; VT/VF non-regression",
            "disallowed_outcomes": "model(A) vs V5D as if they were the same object",
        },
    ]
    return pd.DataFrame(rows)


def _backbone_rows(perf: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if perf.empty:
        return rows
    best_acc = perf.sort_values("accuracy", ascending=False).iloc[0]
    best_f1 = perf.sort_values("macro_f1", ascending=False).iloc[0]
    best_ece = perf.sort_values("ece", ascending=True).iloc[0]
    for _, row in perf.iterrows():
        flags = []
        if row["version"] == best_acc["version"]:
            flags.append("best_accuracy")
        if row["version"] == best_f1["version"]:
            flags.append("best_macro_f1")
        if row["version"] == best_ece["version"]:
            flags.append("best_ece")
        rows.append(
            {
                "intervention": f"base_model__{row['version']}",
                "intervention_family": "base_architecture",
                "evidence_source": "results_public/tables/model_performance_and_geometry.csv",
                "n_seeds_or_runs": "",
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "ece": row["ece"],
                "vtvf_cross_error_delta": np.nan,
                "total_error_delta": np.nan,
                "boundary_signal_delta": np.nan,
                "error_migration_flag": False,
                "classification_signal": ";".join(flags),
                "routing_value_signal": "baseline model candidate; accuracy alone not sufficient",
                "current_verdict": "baseline_reference",
                "next_action": "Use as comparator, not as the causal-Pareto upgrade itself.",
            }
        )
    return rows


def _public_intervention_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paired = _read_csv(root / "results_public/tables/paired_classification_comparisons.csv")
    migration = _read_csv(root / "results_public/tables/duplicate_family_pro_error_migration_mean_std.csv")

    pro_metrics = {}
    if not paired.empty:
        sub = paired[paired["comparison"].astype(str).eq("prototype_separation_minus_baseline")]
        for _, row in sub.iterrows():
            pro_metrics[str(row["metric"])] = float(row["mean_difference"])
    migration_notes = []
    if not migration.empty and "variant" in migration.columns and "pro_minus_baseline" in migration.columns:
        clean = migration[pd.to_numeric(migration["pro_minus_baseline"], errors="coerce").notna()].copy()
        clean["delta"] = clean["pro_minus_baseline"].astype(float)
        for _, row in clean.sort_values("delta", ascending=False).head(3).iterrows():
            migration_notes.append(f"{row['variant']} {row['delta']:+.1f}")
    rows.append(
        {
            "intervention": "prototype_separation_PRO",
            "intervention_family": "representation_objective",
            "evidence_source": "results_public paired classification and PRO migration tables",
            "n_seeds_or_runs": "3 paired duplicate-family seeds",
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "ece": np.nan,
            "accuracy_delta": pro_metrics.get("accuracy", np.nan),
            "macro_f1_delta": pro_metrics.get("macro_f1", np.nan),
            "ece_delta": pro_metrics.get("ece", np.nan),
            "vtvf_cross_error_delta": pro_metrics.get("vtvf_cross_errors", np.nan),
            "total_error_delta": pro_metrics.get("total_errors", np.nan),
            "boundary_signal_delta": np.nan,
            "error_migration_flag": True,
            "classification_signal": (
                f"accuracy {_fmt(pro_metrics.get('accuracy'), 100)} pp; "
                f"macro-F1 {_fmt(pro_metrics.get('macro_f1'), 100)} pp; "
                f"VT/VF cross-errors {_fmt(pro_metrics.get('vtvf_cross_errors'))}"
            ),
            "routing_value_signal": "Potential downstream evidence source; evaluate separately with a fixed router.",
            "current_verdict": "diagnostic_only_until_guarded",
            "next_action": "Only retry as a model-only guarded ablation with explicit error-migration constraints.",
            "limitation": "Only 3 paired seeds; migration notes: " + "; ".join(migration_notes),
        }
    )
    return rows


def _risk_pro_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delta = _read_csv(root / "results/risk_pro_readable_10seed_20260626/summary/risk_pro_readable_paired_delta_summary.csv")
    stage = _read_csv(root / "results/risk_pro_readable_10seed_20260626/summary/risk_pro_readable_stage_summary.csv")
    row = {
        "intervention": "risk_pro_readable",
        "intervention_family": "risk_and_representation_objective",
        "evidence_source": "results/risk_pro_readable_10seed_20260626/summary",
        "n_seeds_or_runs": "10 paired seeds",
        "accuracy_delta": _metric(delta, "accuracy_delta"),
        "macro_f1_delta": _metric(delta, "macro_f1_delta"),
        "ece_delta": _metric(delta, "ece_delta"),
        "vtvf_cross_error_delta": _metric(delta, "vtvf_cross_errors_delta"),
        "total_error_delta": _metric(delta, "total_errors_delta"),
        "boundary_signal_delta": _metric(delta, "softmax_vtvf_ambiguity_auroc_delta"),
        "knn_vtvf_mix_auroc_delta": _metric(delta, "knn_vtvf_mix_auroc_delta"),
        "error_migration_flag": True,
        "classification_signal": "Ten-seed paired deltas are unstable or negative for core outcomes.",
        "routing_value_signal": "Risk/prototype evidence remains useful diagnostically, but model objective is not stable.",
        "current_verdict": "do_not_scale_as_main_model",
        "next_action": "Use as negative evidence motivating multi-objective constraints.",
        "limitation": "Internal duplicate-family evidence; not external validation.",
    }
    if not stage.empty:
        rp = stage[stage["stage"].astype(str).eq("risk_pro_readable")]
        if not rp.empty:
            for col in ["accuracy_mean", "macro_f1_mean", "ece_mean", "vtvf_cross_errors_mean", "total_errors_mean"]:
                row[col.replace("_mean", "")] = float(rp.iloc[0][col])
    rows.append(row)
    return rows


def _validity_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delta = _read_csv(root / "results/cnn_tcn_validity_20260626/summary/cnn_tcn_validity_paired_delta_summary.csv")
    gate = _read_csv(root / "results/cnn_tcn_validity_20260626/summary/cnn_tcn_validity_gate_summary.csv")
    cnn = delta[(delta["baseline"].astype(str).eq("CNN")) & (delta["comparator"].astype(str).eq("CNN-TCN-Validity"))] if not delta.empty else pd.DataFrame()
    row = {
        "intervention": "cnn_tcn_validity_branch",
        "intervention_family": "validity_domain_branch",
        "evidence_source": "results/cnn_tcn_validity_20260626/summary",
        "n_seeds_or_runs": "10 paired seeds",
        "accuracy_delta": _metric(cnn, "accuracy_delta"),
        "macro_f1_delta": _metric(cnn, "macro_f1_delta"),
        "ece_delta": _metric(cnn, "ece_delta"),
        "vtvf_cross_error_delta": _metric(cnn, "vtvf_cross_errors_delta"),
        "total_error_delta": _metric(cnn, "total_errors_delta"),
        "embedding_silhouette_delta": _metric(cnn, "embedding_silhouette_delta"),
        "softmax_vtvf_boundary_auroc_delta": _metric(cnn, "softmax_vtvf_boundary_auroc_delta"),
        "knn_vtvf_boundary_auroc_delta": _metric(cnn, "knn_vtvf_boundary_auroc_delta"),
        "gate_any_error_auroc": _metric(gate, "gate_any_error_auroc"),
        "gate_vtvf_boundary_auroc": _metric(gate, "gate_vtvf_boundary_auroc"),
        "error_migration_flag": True,
        "classification_signal": "Classifier-level effect mixed; validity gate itself detects any-error well.",
        "routing_value_signal": "Strong candidate as auxiliary evidence/gate, not as standalone classifier replacement.",
        "current_verdict": "retain_as_auxiliary_head",
        "next_action": "Evaluate classifier deltas separately from validity-gate evidence quality.",
        "limitation": "ECE worsened vs CNN in summary; VT/VF boundary gate AUROC is moderate.",
    }
    rows.append(row)
    return rows


def _wavelet_rows(root: Path) -> list[dict[str, Any]]:
    evidence = _read_csv(
        root / "results/mechanism_library_evidence_chain_20260630/mechanism_signal_strength_inventory.csv"
    )
    if evidence.empty:
        evidence = _read_csv(
            root
            / "results/evidence_informed_mechanism_routing_10seed_v4_20260627"
            / "wavelet_boundary_routing_audit"
            / "wavelet_risk_head_mean_std.csv"
        )
    vtvf_auroc = None
    if not evidence.empty:
        rows = evidence[evidence.astype(str).apply(lambda col: col.str.contains("wavelet_vtvf_boundary_risk", na=False)).any(axis=1)]
        for col in rows.columns:
            if "auroc" in col and "mean" in col and pd.to_numeric(rows[col], errors="coerce").notna().any():
                vtvf_auroc = float(pd.to_numeric(rows[col], errors="coerce").dropna().iloc[0])
                break
            if col == "mean" and pd.to_numeric(rows[col], errors="coerce").notna().any():
                vtvf_auroc = float(pd.to_numeric(rows[col], errors="coerce").dropna().iloc[0])
                break
    metrics = _read_json(root / "results/cnn_wavelet_tcn_boundary_20260627/20260627_105654_cnn_wavelet_tcn_boundary_wavelet_boundary_seed42/metrics.json")
    rows = [
        {
            "intervention": "cnn_wavelet_tcn_boundary_branch",
            "intervention_family": "wavelet_time_frequency_branch",
            "evidence_source": "cnn_wavelet_tcn_boundary seed42 metrics; wavelet evidence-head summaries",
            "n_seeds_or_runs": "1 trained wavelet model + 10-seed wavelet evidence-head audit",
            "accuracy": metrics.get("accuracy", np.nan),
            "macro_f1": metrics.get("macro_f1", np.nan),
            "ece": metrics.get("ece", np.nan),
            "vtvf_cross_errors": metrics.get("vtvf_cross_errors", np.nan),
            "total_errors": metrics.get("total_errors", np.nan),
            "wavelet_vtvf_boundary_risk_auroc": vtvf_auroc if vtvf_auroc is not None else np.nan,
            "error_migration_flag": True,
            "classification_signal": "Standalone wavelet-boundary classifier seed42 is not competitive.",
            "routing_value_signal": "Wavelet boundary evidence is strong for VT/VF boundary routing.",
            "current_verdict": "retain_as_evidence_head_not_classifier",
            "next_action": "Keep classifier replacement and wavelet evidence-head evaluation separate.",
            "limitation": "Trained classifier evidence is one-seed; evidence-head value is stronger than classifier value.",
        }
    ]
    return rows


def _regularity_rows(root: Path, perf: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reg = perf[perf["version"].astype(str).eq("RegularityFusion-12")] if not perf.empty else pd.DataFrame()
    gated = perf[perf["version"].astype(str).eq("GatedFusion-12")] if not perf.empty else pd.DataFrame()
    if not reg.empty:
        r = reg.iloc[0]
        rows.append(
            {
                "intervention": "regularity_fusion",
                "intervention_family": "regularity_waveform_branch",
                "evidence_source": "results_public/tables/model_performance_and_geometry.csv",
                "n_seeds_or_runs": "",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "ece": r["ece"],
                "vt_vf_norm_dist": r["vt_vf_norm_dist"],
                "error_migration_flag": False,
                "classification_signal": "Not best classifier, but VT/VF geometry is not poor.",
                "routing_value_signal": "Better suited for atypical-signal/stage2 evidence than primary classifier.",
                "current_verdict": "retain_as_stage2_auxiliary",
                "next_action": "Use regularity auxiliary for atypical signal, not as the main optimization target.",
                "limitation": "Geometry does not prove safer classification.",
            }
        )
    if not gated.empty:
        g = gated.iloc[0]
        rows.append(
            {
                "intervention": "reliability_gated_fusion_backbone",
                "intervention_family": "gated_base_model",
                "evidence_source": "results_public/tables/model_performance_and_geometry.csv",
                "n_seeds_or_runs": "",
                "accuracy": g["accuracy"],
                "macro_f1": g["macro_f1"],
                "ece": g["ece"],
                "vt_vf_norm_dist": g["vt_vf_norm_dist"],
                "error_migration_flag": False,
                "classification_signal": "Best aggregate classifier in public table.",
                "routing_value_signal": "Strong base model, but accuracy does not solve routing alone.",
                "current_verdict": "retain_as_default_backbone",
                "next_action": "Use as teacher/backbone for model-side Pareto interventions.",
                "limitation": "ECE and VT/VF geometry are not the best despite high accuracy.",
            }
        )
    return rows


def _score_shortlist(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    rows = []
    for _, row in inventory.iterrows():
        verdict = str(row.get("current_verdict", ""))
        score = 0
        if verdict in {"retain_as_default_backbone", "retain_as_auxiliary_head", "retain_as_stage2_auxiliary"}:
            score += 3
        if verdict == "retain_as_evidence_head_not_classifier":
            score += 2
        if verdict == "diagnostic_only_until_guarded":
            score += 1
        if verdict == "do_not_scale_as_main_model":
            score -= 2
        if bool(row.get("error_migration_flag", False)):
            score -= 1
        rows.append(
            {
                "intervention": row["intervention"],
                "intervention_family": row["intervention_family"],
                "comparison_layer": row.get("comparison_layer", "model_only"),
                "screening_score": score,
                "current_verdict": verdict,
                "recommended_next_role": row.get("next_action", ""),
                "reason": row.get("classification_signal", ""),
            }
        )
    out = pd.DataFrame(rows).sort_values(["screening_score", "intervention"], ascending=[False, True])
    return out


def _build_model_only_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    model_only = inventory.copy()
    model_only["comparison_layer"] = "model_only"
    model_only["fair_comparison_unit"] = "trained ECG classifier/intervention"
    model_only["fair_comparison_rule"] = "Compare only against other trained classifiers or classifier interventions."
    model_only["model_only_primary_outcomes"] = (
        "accuracy; macro-F1; ECE; VT/VF cross-errors; total errors; error migration"
    )
    model_only["downstream_routing_use"] = (
        "excluded from model-only ranking; use only in fixed_router_downstream_design.csv"
    )
    if "routing_value_signal" in model_only.columns:
        model_only = model_only.drop(columns=["routing_value_signal"])
    for col in ["gate_any_error_auroc", "gate_vtvf_boundary_auroc", "wavelet_vtvf_boundary_risk_auroc"]:
        if col in model_only.columns:
            model_only[col] = np.nan
    return model_only


def _build_evidence_head_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    validity = inventory[inventory["intervention"].astype(str).eq("cnn_tcn_validity_branch")]
    if not validity.empty:
        row = validity.iloc[0]
        rows.append(
            {
                "comparison_layer": "evidence_head_only",
                "evidence_head": "validity_gate",
                "source_intervention": "cnn_tcn_validity_branch",
                "intervenable_variable": "do(add_validity_branch)",
                "evidence_inputs_preserved": "validity gate; boundary score; gate x boundary",
                "fair_comparison_unit": "evidence signal, not classifier",
                "any_error_auroc": row.get("gate_any_error_auroc", np.nan),
                "vtvf_boundary_auroc": row.get("gate_vtvf_boundary_auroc", np.nan),
                "primary_question": "Does the validity head identify errors or boundary cases better than other evidence heads?",
                "not_a_claim": "This row does not claim that CNN-TCN-Validity is a better classifier.",
                "limitation": row.get("limitation", ""),
            }
        )
    wavelet = inventory[inventory["intervention"].astype(str).eq("cnn_wavelet_tcn_boundary_branch")]
    if not wavelet.empty:
        row = wavelet.iloc[0]
        rows.append(
            {
                "comparison_layer": "evidence_head_only",
                "evidence_head": "wavelet_vtvf_boundary_risk",
                "source_intervention": "cnn_wavelet_tcn_boundary_branch",
                "intervenable_variable": "do(add_wavelet_boundary_branch)",
                "evidence_inputs_preserved": "wavelet time-frequency VT/VF boundary features",
                "fair_comparison_unit": "evidence signal, not classifier",
                "any_error_auroc": np.nan,
                "vtvf_boundary_auroc": row.get("wavelet_vtvf_boundary_risk_auroc", np.nan),
                "primary_question": "Does wavelet boundary evidence identify VT/VF boundary risk better than other evidence heads?",
                "not_a_claim": "This row does not claim that the standalone wavelet classifier is a better classifier.",
                "limitation": row.get("limitation", ""),
            }
        )
    rows.extend(
        [
            {
                "comparison_layer": "evidence_head_only",
                "evidence_head": "softmax_vtvf_ambiguity",
                "source_intervention": "existing base classifier output",
                "intervenable_variable": "not directly a training intervention unless used as an auxiliary loss",
                "evidence_inputs_preserved": "softmax SR/VT/VF ambiguity and VT/VF margin",
                "fair_comparison_unit": "evidence signal, not classifier",
                "any_error_auroc": np.nan,
                "vtvf_boundary_auroc": np.nan,
                "primary_question": "Does the classifier's own ambiguity remain useful after model-side changes?",
                "not_a_claim": "Softmax ambiguity is a signal extracted from a classifier, not a separate route policy.",
                "limitation": "Requires paired recomputation for every new trained model.",
            },
            {
                "comparison_layer": "evidence_head_only",
                "evidence_head": "knn_vtvf_mixing_density",
                "source_intervention": "embedding-neighborhood audit",
                "intervenable_variable": "affected by do(model_family) or representation objectives",
                "evidence_inputs_preserved": "KNN neighborhood density; VT/VF mixing; embedding distance",
                "fair_comparison_unit": "evidence signal, not classifier",
                "any_error_auroc": np.nan,
                "vtvf_boundary_auroc": np.nan,
                "primary_question": "Does embedding-neighborhood evidence become cleaner under a model intervention?",
                "not_a_claim": "KNN density is an analysis/evidence variable unless the training objective explicitly uses it.",
                "limitation": "Must be recalculated on held-out records to avoid using the same knife on the same shield.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_fixed_router_downstream_design() -> pd.DataFrame:
    rows = [
        {
            "comparison_layer": "fixed_router_downstream",
            "comparison_id": "downstream_A_validity_aux",
            "allowed_comparison": "V5D(GatedFusion + validity_aux) vs V5D(GatedFusion baseline)",
            "fixed_elements": "same V5D stage1/stage2 policy; same review budget; same split; same recover rules",
            "changed_element": "model outputs and validity evidence supplied to the fixed router",
            "primary_outcomes": "V5D capture; residual VT/VF risk; residual all-error risk; recover action distribution",
            "interpretation": "If improved, the claim is downstream compatibility with V5D, not model-only superiority.",
        },
        {
            "comparison_layer": "fixed_router_downstream",
            "comparison_id": "downstream_B_wavelet_aux",
            "allowed_comparison": "V5D(GatedFusion + wavelet_aux) vs V5D(GatedFusion baseline)",
            "fixed_elements": "same V5D stage1/stage2 policy; same review budget; same split; same recover rules",
            "changed_element": "wavelet boundary evidence supplied by the model-side intervention",
            "primary_outcomes": "VT/VF capture; VT->VF/VF->VT residual errors; OOD-style ECG shift stability",
            "interpretation": "This evaluates whether wavelet evidence helps the same router under a fixed policy.",
        },
        {
            "comparison_layer": "fixed_router_downstream",
            "comparison_id": "downstream_C_regularity_aux",
            "allowed_comparison": "V5D(GatedFusion + regularity_aux) vs V5D(GatedFusion baseline)",
            "fixed_elements": "same V5D stage1/stage2 policy; same review budget; same split; same recover rules",
            "changed_element": "regularity/SR-ventricular evidence supplied to stage2",
            "primary_outcomes": "all-error capture; SR-ventricular residual errors; VT/VF non-regression",
            "interpretation": "This tests downstream recovery support, not whether regularity alone is a better classifier.",
        },
        {
            "comparison_layer": "fixed_router_downstream",
            "comparison_id": "downstream_D_guarded_prototype",
            "allowed_comparison": "V5D(GatedFusion + guarded_PRO) vs V5D(GatedFusion baseline)",
            "fixed_elements": "same V5D stage1/stage2 policy; same review budget; same split; same recover rules",
            "changed_element": "prototype-regularized embedding and derived evidence variables",
            "primary_outcomes": "capture; residual risk; SR->VT/VF migration; VT->VF migration",
            "interpretation": "Only meaningful after model-only error migration is controlled.",
        },
    ]
    return pd.DataFrame(rows)


def _proposed_next_model_only_experiments() -> pd.DataFrame:
    rows = [
        {
            "experiment_id": "model_pareto_A_conservative_validity_gate",
            "base_model": "reliability_gated_fusion",
            "comparison_layer": "model_only",
            "interventions": "validity auxiliary branch as a model-side intervention",
            "why": "Validity gate detects any-error well, but classifier-level effect is mixed.",
            "primary_outcomes": "accuracy; macro-F1; ECE; VT/VF cross-errors; total errors; error migration",
            "guardrails": "Do not accept accuracy-only improvement; reject if VT/VF cross-errors or ECE worsen beyond tolerance.",
        },
        {
            "experiment_id": "model_pareto_B_wavelet_aux_boundary",
            "base_model": "reliability_gated_fusion",
            "comparison_layer": "model_only",
            "interventions": "wavelet boundary auxiliary/risk head, not standalone classifier replacement",
            "why": "Wavelet evidence is valuable for VT/VF boundary routing, while standalone seed42 classifier is weak.",
            "primary_outcomes": "accuracy; macro-F1; ECE; VT/VF cross-errors; total errors; VT->VF/VF->VT migration",
            "guardrails": "Reject if VT->VF or VF->VT error migration increases.",
        },
        {
            "experiment_id": "model_pareto_C_stage2_regularity_sr",
            "base_model": "reliability_gated_fusion",
            "comparison_layer": "model_only",
            "interventions": "regularity auxiliary + SR-ventricular/stage2 residual target",
            "why": "V5D weight upgrade found the safest improvement space in stage2 SR-ventricular weighting.",
            "primary_outcomes": "accuracy; macro-F1; ECE; SR-ventricular errors; total errors; VT/VF non-regression",
            "guardrails": "Must preserve VT/VF classification outcomes within tolerance.",
        },
        {
            "experiment_id": "model_pareto_D_guarded_prototype",
            "base_model": "reliability_gated_fusion",
            "comparison_layer": "model_only",
            "interventions": "prototype objective with explicit error-migration penalty",
            "why": "PRO can improve representation/classification but has clear migration risk.",
            "primary_outcomes": "macro-F1; VT/VF cross-errors; SR->VT/VF migration; total errors; ECE",
            "guardrails": "Only run as guarded ablation; not a main method until migration is controlled.",
        },
    ]
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    perf = _read_csv(root / "results_public/tables/model_performance_and_geometry.csv")
    rows: list[dict[str, Any]] = []
    rows.extend(_backbone_rows(perf))
    rows.extend(_regularity_rows(root, perf))
    rows.extend(_public_intervention_rows(root))
    rows.extend(_risk_pro_rows(root))
    rows.extend(_validity_rows(root))
    rows.extend(_wavelet_rows(root))
    inventory = pd.DataFrame(rows)
    variable_dict = _build_variable_dictionary()
    layer_dict = _build_comparison_layer_dictionary()
    model_only_inventory = _build_model_only_inventory(inventory)
    evidence_head_inventory = _build_evidence_head_inventory(inventory)
    fixed_router_design = _build_fixed_router_downstream_design()
    shortlist = _score_shortlist(model_only_inventory)
    proposed = _proposed_next_model_only_experiments()

    inventory.to_csv(out / "model_side_intervention_inventory.csv", index=False)
    model_only_inventory.to_csv(out / "model_only_intervention_inventory.csv", index=False)
    evidence_head_inventory.to_csv(out / "evidence_head_inventory.csv", index=False)
    fixed_router_design.to_csv(out / "fixed_router_downstream_design.csv", index=False)
    variable_dict.to_csv(out / "model_side_causal_variable_dictionary.csv", index=False)
    layer_dict.to_csv(out / "comparison_layer_dictionary.csv", index=False)
    shortlist.to_csv(out / "model_side_candidate_shortlist.csv", index=False)
    proposed.to_csv(out / "model_side_next_model_only_experiment_plan.csv", index=False)
    proposed.to_csv(out / "model_side_next_experiment_plan.csv", index=False)

    report = {
        "out": str(out),
        "n_mixed_source_inventory_rows": int(len(inventory)),
        "n_model_only_rows": int(len(model_only_inventory)),
        "n_evidence_head_rows": int(len(evidence_head_inventory)),
        "n_fixed_router_design_rows": int(len(fixed_router_design)),
        "n_variable_rows": int(len(variable_dict)),
        "n_comparison_layer_rows": int(len(layer_dict)),
        "n_shortlist_rows": int(len(shortlist)),
        "n_next_experiments": int(len(proposed)),
        "interpretation_boundary": (
            "model-only comparisons, evidence-head comparisons, and fixed-router downstream "
            "comparisons are separated; no new training launched"
        ),
        "top_shortlist": shortlist.head(6).to_dict(orient="records"),
    }
    (out / "model_side_causal_pareto_inventory_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory model-side causal-Pareto interventions for the ECG project.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
