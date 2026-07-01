from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("results/model_layer_all_model_benchmark_20260629")
INTERVENTION_ARG_KEYS = [
    "aux_boundary_weight",
    "gate_target_weight",
    "gate_sparsity_weight",
    "risk_boundary_weight",
    "risk_gate_weight",
    "risk_entropy_weight",
    "boundary_ce_weight",
    "boundary_weighted_ce_weight",
    "stability_consistency_weight",
    "embedding_consistency_weight",
    "selective_stability_consistency_weight",
    "selective_embedding_consistency_weight",
    "anti_confident_risk_weight",
    "prototype_center_weight",
    "prototype_margin_weight",
    "prototype_vtvf_margin",
    "regularity_aux_weight",
    "vtvf_specialist_weight",
]
METRIC_COLS = ["accuracy", "macro_f1", "ece", "vtvf_cross_errors", "total_errors"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint_args(run_dir: Path) -> dict[str, Any]:
    ckpt = run_dir / "best_model.pt"
    if not ckpt.exists():
        return {}
    try:
        import torch

        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        if isinstance(args, dict):
            return args
        return vars(args)
    except Exception:
        return {}


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        f = float(value)
        if abs(f) < 1e-12:
            return None
        return round(f, 8)
    except (TypeError, ValueError):
        s = str(value)
        return s if s and s.lower() not in {"none", "false"} else None


def _intervention_signature(args: dict[str, Any], run_name: str) -> tuple[str, str]:
    model = str(args.get("model") or _infer_model_from_name(run_name))
    active: list[str] = []
    for key in INTERVENTION_ARG_KEYS:
        value = _normalise_value(args.get(key))
        if value is not None:
            if key == "prototype_vtvf_margin" and not (
                _normalise_value(args.get("prototype_margin_weight"))
                or _normalise_value(args.get("prototype_center_weight"))
            ):
                continue
            active.append(f"{key}={value}")
    risk_targets = str(args.get("risk_targets", "") or "")
    if risk_targets:
        active.append("risk_targets=used")
    if not active:
        active.append("no_extra_constraint")
    signature = f"{model}__" + "__".join(active)
    return model, signature


def _infer_model_from_name(run_name: str) -> str:
    known = [
        "cnn_wavelet_tcn_boundary",
        "cnn_tcn_validity_v2",
        "cnn_tcn_validity",
        "reliability_gated_fusion",
        "regularity_fusion",
        "inception_time",
        "cnn_lstm",
        "resnet1d",
        "bigru",
        "tcn",
        "cnn",
    ]
    for model in known:
        if model in run_name:
            return model
    return "unknown"


def _seed_from_args_or_name(args: dict[str, Any], run_name: str) -> int | None:
    if "seed" in args:
        try:
            return int(args["seed"])
        except (TypeError, ValueError):
            pass
    marker = "seed"
    if marker in run_name:
        tail = run_name.split(marker, 1)[1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return int(digits)
    return None


def _discover_all_metric_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((root / "results").rglob("metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        args = _load_checkpoint_args(run_dir)
        model, signature = _intervention_signature(args, run_dir.name)
        rel = run_dir.relative_to(root)
        experiment = rel.parts[1] if len(rel.parts) > 2 else "top_level_results"
        is_smoke = any(token in str(rel).lower() for token in ["smoke", "dry_run", "dry-run"])
        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "experiment": experiment,
            "run_name": run_dir.name,
            "model": model,
            "model_signature": signature,
            "seed": _seed_from_args_or_name(args, run_dir.name),
            "epochs": args.get("epochs"),
            "batch_size": args.get("batch_size"),
            "lr": args.get("lr"),
            "max_windows_per_record": args.get("max_windows_per_record"),
            "split_grouping": args.get("split_grouping", ""),
            "risk_targets_used": bool(args.get("risk_targets")),
            "is_smoke_or_dry_run": is_smoke,
            "source": str(metrics_path),
        }
        for key in INTERVENTION_ARG_KEYS:
            row[key] = args.get(key)
        for metric in METRIC_COLS + ["vt_as_vf", "vf_as_vt", "sr_as_vt", "sr_as_vf", "nll"]:
            row[metric] = metrics.get(metric, np.nan)
        row["error_migration_penalty"] = (
            float(row.get("vt_as_vf") or 0.0)
            + float(row.get("vf_as_vt") or 0.0)
            + 0.5 * float(row.get("sr_as_vt") or 0.0)
            + 0.5 * float(row.get("sr_as_vf") or 0.0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_discovered_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for signature, sub in runs.groupby("model_signature", sort=True):
        if sub["seed"].notna().any():
            metric_cols = [m for m in METRIC_COLS + ["error_migration_penalty"] if m in sub.columns]
            seeded = sub[sub["seed"].notna()].copy()
            metric_base = seeded.groupby("seed", as_index=False)[metric_cols].mean()
        else:
            metric_base = sub.copy()
        row: dict[str, Any] = {
            "model_signature": signature,
            "model": sub["model"].mode().iloc[0] if not sub["model"].mode().empty else sub["model"].iloc[0],
            "n_runs": int(len(sub)),
            "n_seeds": int(sub["seed"].dropna().nunique()),
            "seeds": ";".join(str(int(s)) for s in sorted(sub["seed"].dropna().unique())),
            "experiments": ";".join(sorted(sub["experiment"].astype(str).unique())),
            "has_smoke_or_dry_run": bool(sub["is_smoke_or_dry_run"].any()),
            "risk_targets_used_any": bool(sub["risk_targets_used"].any()),
        }
        for metric in METRIC_COLS + ["error_migration_penalty"]:
            values = pd.to_numeric(metric_base[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric}_std"] = float(values.std()) if values.notna().sum() > 1 else np.nan
            row[f"{metric}_min"] = float(values.min()) if values.notna().any() else np.nan
            row[f"{metric}_max"] = float(values.max()) if values.notna().any() else np.nan
        active_args = []
        for key in INTERVENTION_ARG_KEYS:
            if key == "prototype_vtvf_margin":
                has_proto = any(
                    _normalise_value(v) is not None
                    for proto_key in ["prototype_center_weight", "prototype_margin_weight"]
                    for v in sub[proto_key].dropna().tolist()
                )
                if not has_proto:
                    continue
            vals = sorted({_normalise_value(v) for v in sub[key].dropna().tolist() if _normalise_value(v) is not None})
            if vals:
                active_args.append(f"{key}={','.join(map(str, vals))}")
        row["active_intervention_args"] = ";".join(active_args) if active_args else "no_extra_constraint"
        rows.append(row)
    out = pd.DataFrame(rows)
    for metric, ascending in [
        ("accuracy_mean", False),
        ("macro_f1_mean", False),
        ("ece_mean", True),
        ("vtvf_cross_errors_mean", True),
        ("total_errors_mean", True),
        ("error_migration_penalty_mean", True),
    ]:
        out[f"rank_{metric}"] = pd.to_numeric(out[metric], errors="coerce").rank(
            ascending=ascending, method="min", na_option="bottom"
        )
    out["mean_rank_all_model_metrics"] = out[
        [
            "rank_accuracy_mean",
            "rank_macro_f1_mean",
            "rank_ece_mean",
            "rank_vtvf_cross_errors_mean",
            "rank_total_errors_mean",
            "rank_error_migration_penalty_mean",
        ]
    ].mean(axis=1)
    return out.sort_values(["mean_rank_all_model_metrics", "model_signature"])


def _append_model(
    rows: list[dict[str, Any]],
    *,
    model: str,
    family: str,
    evidence_type: str,
    n: int | str,
    source: str,
    accuracy: float | None = None,
    macro_f1: float | None = None,
    ece: float | None = None,
    vtvf_cross_errors: float | None = None,
    total_errors: float | None = None,
    comparison_baseline: str = "",
    accuracy_delta: float | None = None,
    macro_f1_delta: float | None = None,
    ece_delta: float | None = None,
    vtvf_cross_errors_delta: float | None = None,
    total_errors_delta: float | None = None,
    interpretation: str = "",
) -> None:
    rows.append(
        {
            "model": model,
            "model_family": family,
            "evidence_type": evidence_type,
            "n_seeds_or_runs": n,
            "comparison_baseline": comparison_baseline,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "ece": ece,
            "vtvf_cross_errors": vtvf_cross_errors,
            "total_errors": total_errors,
            "accuracy_delta": accuracy_delta,
            "macro_f1_delta": macro_f1_delta,
            "ece_delta": ece_delta,
            "vtvf_cross_errors_delta": vtvf_cross_errors_delta,
            "total_errors_delta": total_errors_delta,
            "source": source,
            "interpretation": interpretation,
        }
    )


def _public_snapshot_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    perf = _read_csv(root / "results_public/tables/model_performance_and_geometry.csv")
    if perf.empty:
        return
    for _, r in perf.iterrows():
        version = str(r["version"])
        _append_model(
            rows,
            model=version,
            family="architecture_snapshot",
            evidence_type="public aggregate snapshot",
            n="reported aggregate",
            source="results_public/tables/model_performance_and_geometry.csv",
            accuracy=float(r["accuracy"]),
            macro_f1=float(r["macro_f1"]),
            ece=float(r["ece"]),
            interpretation="Architecture-level model snapshot; useful for broad model-stage comparison.",
        )


def _cnn_lstm_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    per_seed = _read_csv(
        root / "results/cnn_lstm_baseline_20260626/multiseed_summary_10seed/per_seed_model_comparison.csv"
    )
    delta = _read_csv(
        root / "results/cnn_lstm_baseline_20260626/multiseed_summary_10seed/aggregate_delta_summary.csv"
    )
    if per_seed.empty:
        return
    for prefix, label in [("cnn", "CNN-10seed"), ("cnn_lstm", "CNN-LSTM-10seed")]:
        _append_model(
            rows,
            model=label,
            family="architecture_paired",
            evidence_type="10-seed paired model architecture comparison",
            n=int(per_seed["seed"].nunique()),
            source="results/cnn_lstm_baseline_20260626/multiseed_summary_10seed/per_seed_model_comparison.csv",
            accuracy=float(per_seed[f"{prefix}_accuracy"].mean()),
            macro_f1=float(per_seed[f"{prefix}_macro_f1"].mean()),
            ece=float(per_seed[f"{prefix}_ece"].mean()),
            vtvf_cross_errors=float(per_seed[f"{prefix}_vtvf_cross_errors"].mean()),
            total_errors=float(per_seed[f"{prefix}_total_errors"].mean()),
            comparison_baseline="CNN-10seed" if label == "CNN-LSTM-10seed" else "",
            interpretation=(
                "CNN-LSTM is a model-stage comparator; it improves VT/VF cross-errors on average "
                "but worsens accuracy, ECE, and total errors vs paired CNN."
                if label == "CNN-LSTM-10seed"
                else "Paired CNN baseline for CNN-LSTM."
            ),
        )
    if not delta.empty:
        delta_map = dict(zip(delta["metric"], delta["mean"]))
        for r in rows:
            if r["model"] == "CNN-LSTM-10seed":
                r["accuracy_delta"] = delta_map.get("accuracy_delta")
                r["macro_f1_delta"] = delta_map.get("macro_f1_delta")
                r["ece_delta"] = delta_map.get("ece_delta")
                r["vtvf_cross_errors_delta"] = delta_map.get("vtvf_cross_errors_delta")
                r["total_errors_delta"] = delta_map.get("total_errors_delta")


def _paired_public_interventions(root: Path, rows: list[dict[str, Any]]) -> None:
    paired = _read_csv(root / "results_public/tables/paired_classification_comparisons.csv")
    if paired.empty:
        return
    names = {
        "prototype_separation_minus_baseline": ("PRO / prototype_separation", "training_objective_paired"),
        "full_supervisor_minus_baseline": ("FullSupervisor-public", "training_objective_paired"),
    }
    for comparison, (model, family) in names.items():
        sub = paired[paired["comparison"].astype(str).eq(comparison)]
        if sub.empty:
            continue
        values = {str(r["metric"]): r for _, r in sub.iterrows()}
        _append_model(
            rows,
            model=model,
            family=family,
            evidence_type="3-seed paired training-intervention comparison",
            n=int(sub["n_paired_seeds"].max()),
            comparison_baseline="paired internal baseline",
            source="results_public/tables/paired_classification_comparisons.csv",
            accuracy=float(values["accuracy"]["comparator_mean"]) if "accuracy" in values else None,
            macro_f1=float(values["macro_f1"]["comparator_mean"]) if "macro_f1" in values else None,
            ece=float(values["ece"]["comparator_mean"]) if "ece" in values else None,
            vtvf_cross_errors=float(values["vtvf_cross_errors"]["comparator_mean"])
            if "vtvf_cross_errors" in values
            else None,
            total_errors=float(values["total_errors"]["comparator_mean"]) if "total_errors" in values else None,
            accuracy_delta=float(values["accuracy"]["mean_difference"]) if "accuracy" in values else None,
            macro_f1_delta=float(values["macro_f1"]["mean_difference"]) if "macro_f1" in values else None,
            ece_delta=float(values["ece"]["mean_difference"]) if "ece" in values else None,
            vtvf_cross_errors_delta=float(values["vtvf_cross_errors"]["mean_difference"])
            if "vtvf_cross_errors" in values
            else None,
            total_errors_delta=float(values["total_errors"]["mean_difference"]) if "total_errors" in values else None,
            interpretation=(
                "Positive model-stage intervention but kept cautious because error migration was observed."
                if "prototype" in comparison
                else "Same family as causal-Pareto full-supervisor candidate in public paired table."
            ),
        )


def _risk_pro_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    stage = _read_csv(root / "results/risk_pro_readable_10seed_20260626/summary/risk_pro_readable_stage_summary.csv")
    delta = _read_csv(root / "results/risk_pro_readable_10seed_20260626/summary/risk_pro_readable_paired_delta_summary.csv")
    if stage.empty:
        return
    delta_map = dict(zip(delta["metric"], delta["mean"])) if not delta.empty else {}
    for _, r in stage.iterrows():
        name = str(r["stage"])
        _append_model(
            rows,
            model=f"RiskProReadable-{name}",
            family="risk_pro_training_objective",
            evidence_type="10-seed paired risk/prototype objective comparison",
            n=10,
            comparison_baseline="RiskProReadable-teacher" if name == "risk_pro_readable" else "",
            source="results/risk_pro_readable_10seed_20260626/summary",
            accuracy=float(r["accuracy_mean"]),
            macro_f1=float(r["macro_f1_mean"]),
            ece=float(r["ece_mean"]),
            vtvf_cross_errors=float(r["vtvf_cross_errors_mean"]),
            total_errors=float(r["total_errors_mean"]),
            accuracy_delta=delta_map.get("accuracy_delta") if name == "risk_pro_readable" else None,
            macro_f1_delta=delta_map.get("macro_f1_delta") if name == "risk_pro_readable" else None,
            ece_delta=delta_map.get("ece_delta") if name == "risk_pro_readable" else None,
            vtvf_cross_errors_delta=delta_map.get("vtvf_cross_errors_delta") if name == "risk_pro_readable" else None,
            total_errors_delta=delta_map.get("total_errors_delta") if name == "risk_pro_readable" else None,
            interpretation=(
                "Negative/unstable risk-pro objective; useful as evidence that model optimization needs guards."
                if name == "risk_pro_readable"
                else "Teacher comparator for risk-pro-readable."
            ),
        )


def _causal_pareto_candidate_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    run_level = _read_csv(root / "results/model_layer_causal_pareto_validation_20260629/model_layer_run_level_metrics.csv")
    pareto = _read_csv(root / "results/model_layer_causal_pareto_validation_20260629/model_layer_pareto_selection.csv")
    if run_level.empty:
        return
    for variant in ["baseline", "full_supervisor", "boundary_weighted", "stability_consistency"]:
        sub = run_level[run_level["variant"].astype(str).eq(variant)]
        if sub.empty:
            continue
        p = pareto[pareto["variant"].astype(str).eq(variant)] if not pareto.empty else pd.DataFrame()
        selected = bool(p.iloc[0]["selected_model_layer_candidate"]) if not p.empty else False
        _append_model(
            rows,
            model="causal_pareto_full_supervisor" if variant == "full_supervisor" else f"aux_{variant}",
            family="causal_pareto_training_objective",
            evidence_type="3-seed paired causal-Pareto model-layer validation",
            n=int(sub["seed"].nunique()),
            comparison_baseline="aux_baseline" if variant != "baseline" else "",
            source="results/model_layer_causal_pareto_validation_20260629",
            accuracy=float(sub["accuracy"].mean()),
            macro_f1=float(sub["macro_f1"].mean()),
            ece=float(sub["ece"].mean()),
            vtvf_cross_errors=float(sub["vtvf_cross_errors"].mean()),
            total_errors=float(sub["total_errors"].mean()),
            accuracy_delta=float(p.iloc[0]["accuracy_delta_mean"]) if not p.empty else None,
            macro_f1_delta=float(p.iloc[0]["macro_f1_delta_mean"]) if not p.empty else None,
            ece_delta=float(p.iloc[0]["ece_delta_mean"]) if not p.empty else None,
            vtvf_cross_errors_delta=float(p.iloc[0]["vtvf_cross_errors_delta_mean"]) if not p.empty else None,
            total_errors_delta=float(p.iloc[0]["total_errors_delta_mean"]) if not p.empty else None,
            interpretation=(
                "Selected model-layer causal-Pareto candidate; compare as a model, not as a router."
                if selected
                else "Auxiliary model-layer comparator."
            ),
        )


def _wavelet_single_seed(root: Path, rows: list[dict[str, Any]]) -> None:
    metrics = _read_json(
        root
        / "results/cnn_wavelet_tcn_boundary_20260627/20260627_105654_cnn_wavelet_tcn_boundary_wavelet_boundary_seed42/metrics.json"
    )
    if not metrics:
        return
    _append_model(
        rows,
        model="CNN-Wavelet-TCN-boundary-seed42",
        family="wavelet_architecture_single_seed",
        evidence_type="single-seed model classifier result",
        n=1,
        source="results/cnn_wavelet_tcn_boundary_20260627/.../metrics.json",
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        ece=float(metrics["ece"]),
        vtvf_cross_errors=float(metrics["vtvf_cross_errors"]),
        total_errors=float(metrics["total_errors"]),
        interpretation="Weak standalone classifier; wavelet remains stronger as evidence head than as classifier.",
    )


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric, ascending in [
        ("accuracy", False),
        ("macro_f1", False),
        ("ece", True),
        ("vtvf_cross_errors", True),
        ("total_errors", True),
    ]:
        out[f"rank_{metric}"] = pd.to_numeric(out[metric], errors="coerce").rank(
            ascending=ascending, method="min", na_option="bottom"
        )
    out["mean_rank_core_model_metrics"] = out[
        ["rank_accuracy", "rank_macro_f1", "rank_ece", "rank_vtvf_cross_errors", "rank_total_errors"]
    ].mean(axis=1)
    return out.sort_values(["mean_rank_core_model_metrics", "model"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    discovered_runs = _discover_all_metric_runs(root)
    discovered_summary = _aggregate_discovered_runs(discovered_runs)

    rows: list[dict[str, Any]] = []
    _public_snapshot_rows(root, rows)
    _cnn_lstm_rows(root, rows)
    _paired_public_interventions(root, rows)
    _risk_pro_rows(root, rows)
    _causal_pareto_candidate_rows(root, rows)
    _wavelet_single_seed(root, rows)
    benchmark = pd.DataFrame(rows)
    ranked = _rank(benchmark)
    discovered_runs.to_csv(out / "all_discovered_model_runs.csv", index=False)
    discovered_summary.to_csv(out / "all_discovered_model_signature_summary.csv", index=False)
    benchmark.to_csv(out / "all_model_layer_benchmark.csv", index=False)
    ranked.to_csv(out / "all_model_layer_ranked_summary.csv", index=False)
    nonsmoke = discovered_summary[~discovered_summary["has_smoke_or_dry_run"]].copy() if not discovered_summary.empty else pd.DataFrame()
    report = {
        "out": str(out),
        "n_discovered_metric_runs": int(len(discovered_runs)),
        "n_discovered_model_signatures": int(len(discovered_summary)),
        "n_non_smoke_model_signatures": int(len(nonsmoke)),
        "n_model_rows": int(len(benchmark)),
        "n_evidence_types": int(benchmark["evidence_type"].nunique()) if not benchmark.empty else 0,
        "top_discovered_non_smoke_signatures": nonsmoke.head(12).to_dict(orient="records"),
        "top_ranked_models": ranked.head(8).to_dict(orient="records"),
        "interpretation_boundary": (
            "All rows are model-stage evidence, including constrained and complex training-objective models. "
            "Evidence types are not equally strong or fully paired; use n_runs, n_seeds, experiments, and smoke flags."
        ),
    }
    (out / "all_model_layer_benchmark_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all model-stage ECG results into a benchmark table.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
