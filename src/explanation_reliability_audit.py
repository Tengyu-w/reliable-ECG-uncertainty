from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SEEDS = list(range(42, 52))


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_aupr(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    lo = np.nanquantile(values, 0.05)
    hi = np.nanquantile(values, 0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(len(values), dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _first_available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _mean_score(df: pd.DataFrame, columns: list[str], invert: list[str] | None = None) -> np.ndarray:
    available = _first_available(df, columns)
    if not available:
        return np.full(len(df), np.nan, dtype=np.float32)
    invert = set(invert or [])
    parts = []
    for col in available:
        values = df[col].to_numpy(float)
        if col in invert:
            values = -values
        parts.append(_robust_scale(values))
    return np.nanmean(np.vstack(parts), axis=0).astype(np.float32)


def _top_metrics(y: np.ndarray, score: np.ndarray, budget: float) -> dict[str, float]:
    if np.all(np.isnan(score)):
        return {
            f"top{int(budget * 100)}_precision": float("nan"),
            f"top{int(budget * 100)}_capture": float("nan"),
            f"top{int(budget * 100)}_lift": float("nan"),
        }
    n = max(1, int(round(len(score) * budget)))
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:n]
    base = float(y.mean())
    precision = float(y[order].mean())
    capture = float(y[order].sum() / max(y.sum(), 1))
    return {
        f"top{int(budget * 100)}_precision": precision,
        f"top{int(budget * 100)}_capture": capture,
        f"top{int(budget * 100)}_lift": float(precision / base) if base > 0 else float("nan"),
    }


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _evidence_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    boundary_cols = [
        "softmax_vtvf_ambiguity",
        "vtvf_boundary_risk",
        "vtvf_boundary_mechanism_risk",
        "runtime_supervisor_boundary_risk",
        "ambiguity_routing_boundary_signal",
        "regularity_analysis_boundary_ambiguity_score",
        "lrii_boundary_lrii",
    ]
    representation_cols = [
        "proto_vtvf_ambiguity",
        "knn_label_entropy",
        "knn_vtvf_mixing",
        "embedding_neighborhood_vtvf_mixing",
        "embedding_neighborhood_error",
        "representation_conflict_mechanism_risk",
        "decision_boundary_mechanism_representation_overlap",
        "decision_boundary_classifier_proto_disagree",
    ]
    regularity_cols = [
        "regularity_analysis_atypicality_score",
        "regularity_analysis_mahalanobis_atypicality",
        "reliability_map_atypicality_score",
        "lrii_atypicality_lrii",
        "atypical_signal_mechanism_risk",
        "regularity_spectral_entropy_z",
        "regularity_dominant_frequency_z",
        "regularity_spectral_bandwidth_z",
        "regularity_line_length_z",
    ]
    disagreement_cols = [
        "model_disagreement",
        "model_disagreement_any_vtvf",
        "second_entropy",
        "second_softmax_vtvf_ambiguity",
    ]
    confidence_cols = [
        "hidden_confident_mechanism_risk",
        "max_prob",
        "temperature_max_prob",
        "runtime_supervisor_hidden_failure_risk",
    ]
    sr_ventricular_cols = [
        "sr_ventricular_mechanism_risk",
        "ventricular_prob",
        "pred_is_vtvf",
        "regularity_analysis_is_vtvf",
        "prior_calibration_ventricular_prob",
    ]
    return {
        "boundary_explanation": _mean_score(df, boundary_cols),
        "representation_explanation": _mean_score(df, representation_cols, invert=["nearest_proto_is_pred"]),
        "regularity_atypicality_explanation": _mean_score(df, regularity_cols),
        "second_opinion_explanation": _mean_score(df, disagreement_cols),
        "hidden_confidence_explanation": _mean_score(df, confidence_cols),
        "sr_ventricular_explanation": _mean_score(df, sr_ventricular_cols),
    }


def _targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    y = df["y_true"].to_numpy(int)
    pred = df["y_pred"].to_numpy(int)
    return {
        "any_error": df["is_error"].to_numpy(int),
        "vtvf_cross_error": df["is_vtvf_cross_error"].to_numpy(int),
        "sr_ventricular_error": df.get("is_sr_ventricular_error", pd.Series(np.zeros(len(df)))).to_numpy(int),
        "representation_conflict_error": df.get(
            "is_representation_conflict_error", pd.Series(np.zeros(len(df)))
        ).to_numpy(int),
        "atypical_signal_error": df.get("is_atypical_signal_error", pd.Series(np.zeros(len(df)))).to_numpy(int),
        "hidden_confident_error": df.get("is_hidden_confident_error", pd.Series(np.zeros(len(df)))).to_numpy(int),
        "vt_to_vf": ((y == 1) & (pred == 2)).astype(int),
        "vf_to_vt": ((y == 2) & (pred == 1)).astype(int),
    }


def _run_seed(seed: int, seed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(seed_dir / "evidence_scores_test.csv")
    scores = _evidence_scores(df)
    targets = _targets(df)
    rows = []
    for evidence_name, score in scores.items():
        for target_name, target in targets.items():
            row: dict[str, Any] = {
                "seed": seed,
                "evidence_family": evidence_name,
                "target_error_type": target_name,
                "target_base_rate": float(target.mean()),
                "auroc": _safe_auc(target, score),
                "aupr": _safe_aupr(target, score),
            }
            row.update(_top_metrics(target, score, 0.10))
            row.update(_top_metrics(target, score, 0.20))
            rows.append(row)

    score_frame = pd.DataFrame({"seed": seed, "sample_id": df["sample_id"]})
    for name, score in scores.items():
        score_frame[name] = score
    for name, target in targets.items():
        score_frame[f"target_{name}"] = target
    return pd.DataFrame(rows), score_frame


def _route_alignment(v5d_dir: Path) -> pd.DataFrame:
    rows = []
    if not v5d_dir.exists():
        return pd.DataFrame()
    route_target = {
        "boundary_first": "is_vtvf_cross_error",
        "sr_ventricular": "is_sr_ventricular_error",
        "representation_conflict": "is_representation_conflict_error",
        "atypical_signal": "is_atypical_signal_error",
        "hidden_confident": "is_hidden_confident_error",
    }
    for seed in SEEDS:
        path = v5d_dir / f"seed{seed}" / "v5d_reserved_routing_assignments_test_compact.csv"
        if not path.exists():
            continue
        routed = pd.read_csv(path)
        for (budget, reserve), sub_budget in routed.groupby(["budget", "reserve_fraction"], sort=True):
            selected = sub_budget[sub_budget["mechanism_route"] != "single_label"]
            for route, sub in selected.groupby("mechanism_route"):
                target_col = route_target.get(route)
                rows.append(
                    {
                        "seed": seed,
                        "budget": float(budget),
                        "reserve_fraction": float(reserve),
                        "mechanism_route": route,
                        "n_selected": int(len(sub)),
                        "all_error_precision": float(sub["is_error"].mean()) if len(sub) else np.nan,
                        "route_target": target_col or "",
                        "route_target_precision": float(sub[target_col].mean())
                        if target_col and target_col in sub.columns and len(sub)
                        else np.nan,
                        "vtvf_cross_error_precision": float(sub["is_vtvf_cross_error"].mean()) if len(sub) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _write_report(out_dir: Path, summary: pd.DataFrame, route_summary: pd.DataFrame) -> None:
    preferred = {
        "boundary_explanation": "vtvf_cross_error",
        "regularity_atypicality_explanation": "atypical_signal_error",
        "representation_explanation": "representation_conflict_error",
        "second_opinion_explanation": "any_error",
        "hidden_confidence_explanation": "hidden_confident_error",
        "sr_ventricular_explanation": "sr_ventricular_error",
    }
    lines = [
        "# Explanation Reliability Audit",
        "",
        "## One-sentence conclusion",
        "",
        "This audit tests whether explanation/evidence families align with the error mechanisms they are supposed to justify, rather than only producing plausible-looking plots.",
        "",
        "## Preferred mechanism alignment",
        "",
        "| evidence family | intended target | AUROC mean | AUPR mean | top 10% capture mean | top 20% capture mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for evidence, target in preferred.items():
        sub = summary[(summary["evidence_family"] == evidence) & (summary["target_error_type"] == target)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        lines.append(
            f"| {evidence} | {target} | {row['auroc_mean']:.4f} | {row['aupr_mean']:.4f} | "
            f"{row['top10_capture_mean']:.4f} | {row['top20_capture_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A strong result means the explanation family is not merely decorative; it identifies the failure mode it claims to explain.",
            "- A weak or off-target result means the evidence can still be useful, but should not be used as the sole justification for that route.",
            "- These are internal explanation-faithfulness checks, not clinical interpretability validation.",
            "",
        ]
    )
    if not route_summary.empty:
        focus = route_summary[
            (route_summary["budget"] == 0.20)
            & (route_summary["reserve_fraction"] == 0.20)
        ]
        if not focus.empty:
            lines.extend(
                [
                    "## v5d route-level alignment at 20% budget / 20% residual reserve",
                    "",
                    "| route | n selected mean | all-error precision | target precision | VT/VF precision |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for _, row in focus.iterrows():
                lines.append(
                    f"| {row['mechanism_route']} | {row['n_selected_mean']:.1f} | "
                    f"{row['all_error_precision_mean']:.4f} | {row['route_target_precision_mean']:.4f} | "
                    f"{row['vtvf_cross_error_precision_mean']:.4f} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Manuscript-safe wording",
            "",
            "> We evaluated explanation reliability by testing whether each evidence family preferentially identifies the failure mechanism it is intended to support. This turns interpretability from a visual plausibility claim into an error-mechanism alignment test.",
            "",
        ]
    )
    (out_dir / "EXPLANATION_RELIABILITY_AUDIT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify whether explanation evidence aligns with intended ECG error mechanisms.")
    parser.add_argument("--routing-dir", type=Path, default=Path("results/evidence_informed_mechanism_routing_10seed_v4_20260627"))
    parser.add_argument("--v5d-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (args.routing_dir / "explanation_reliability_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_scores = []
    manifest = []
    for seed in SEEDS:
        seed_dir = args.routing_dir / f"seed{seed}"
        if not (seed_dir / "evidence_scores_test.csv").exists():
            manifest.append({"seed": seed, "status": "missing"})
            continue
        rows, scores = _run_seed(seed, seed_dir)
        all_rows.append(rows)
        all_scores.append(scores)
        rows.to_csv(out_dir / f"seed{seed}_explanation_alignment.csv", index=False)
        scores.to_csv(out_dir / f"seed{seed}_explanation_scores.csv", index=False)
        manifest.append({"seed": seed, "status": "completed", "n_test": int(len(scores))})

    alignment = pd.concat(all_rows, ignore_index=True)
    alignment.to_csv(out_dir / "all_seed_explanation_alignment.csv", index=False)
    metrics = ["target_base_rate", "auroc", "aupr", "top10_precision", "top10_capture", "top10_lift", "top20_precision", "top20_capture", "top20_lift"]
    summary = _mean_std(alignment, ["evidence_family", "target_error_type"], metrics)
    summary.to_csv(out_dir / "explanation_alignment_mean_std.csv", index=False)

    v5d_dir = args.v5d_dir or (args.routing_dir / "hierarchical_router_v5d_reserved")
    route = _route_alignment(v5d_dir)
    if not route.empty:
        route.to_csv(out_dir / "v5d_route_alignment.csv", index=False)
        route_summary = _mean_std(
            route,
            ["budget", "reserve_fraction", "mechanism_route", "route_target"],
            ["n_selected", "all_error_precision", "route_target_precision", "vtvf_cross_error_precision"],
        )
        route_summary.to_csv(out_dir / "v5d_route_alignment_mean_std.csv", index=False)
    else:
        route_summary = pd.DataFrame()

    (out_dir / "explanation_reliability_manifest.json").write_text(
        json.dumps({"manifest": manifest, "routing_dir": str(args.routing_dir), "v5d_dir": str(v5d_dir)}, indent=2),
        encoding="utf-8",
    )
    _write_report(out_dir, summary, route_summary)
    print(f"Wrote explanation reliability audit to {out_dir}")


if __name__ == "__main__":
    main()
