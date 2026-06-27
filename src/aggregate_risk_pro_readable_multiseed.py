from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_row(seed: int, stage: str, run_dir: Path) -> dict:
    metrics = _load_json(run_dir / "metrics.json")
    row = {
        "seed": seed,
        "stage": stage,
        "run_dir": str(run_dir),
        "accuracy": _safe_float(metrics.get("accuracy")),
        "macro_f1": _safe_float(metrics.get("macro_f1")),
        "ece": _safe_float(metrics.get("ece")),
        "vtvf_cross_errors": _safe_float(metrics.get("vtvf_cross_errors")),
        "total_errors": _safe_float(metrics.get("total_errors")),
        "vt_as_vf": _safe_float(metrics.get("vt_as_vf")),
        "vf_as_vt": _safe_float(metrics.get("vf_as_vt")),
    }

    uncertainty_path = run_dir / "uncertainty_metrics.csv"
    if uncertainty_path.exists():
        uncertainty = pd.read_csv(uncertainty_path)
        for score in ["msp", "entropy", "prototype", "knn"]:
            sub = uncertainty[uncertainty["score"].eq(score)]
            if not sub.empty:
                row[f"{score}_error_auroc"] = _safe_float(sub["error_auroc"].iloc[0])
                row[f"{score}_error_aupr"] = _safe_float(sub["error_aupr"].iloc[0])

    ambiguity_path = run_dir / "ambiguity_summary.csv"
    if ambiguity_path.exists():
        ambiguity = pd.read_csv(ambiguity_path)
        for score in ["softmax_vtvf_ambiguity", "knn_vtvf_mix", "ventricular_ambiguity_index"]:
            sub = ambiguity[ambiguity["score"].eq(score)]
            if not sub.empty:
                row[f"{score}_auroc"] = _safe_float(sub["vtvf_boundary_auroc"].iloc[0])
                row[f"{score}_aupr"] = _safe_float(sub["vtvf_boundary_aupr"].iloc[0])

    stability_path = run_dir / "stability_summary.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        for group in ["any_error", "vtvf_cross_error", "confident_stable_error"]:
            sub = stability[stability["group"].eq(group)]
            if not sub.empty:
                prefix = f"stability_{group}"
                for col in ["n", "confidence_mean", "flip_rate_mean", "stability_aware_risk_mean"]:
                    if col in sub.columns:
                        row[f"{prefix}_{col}"] = _safe_float(sub[col].iloc[0])

    stable_error_path = run_dir / "mechanism_stable_confident_errors.csv"
    if stable_error_path.exists():
        stable = pd.read_csv(stable_error_path)
        for group in ["confident_stable_error", "confident_stable_vtvf_cross_error"]:
            sub = stable[stable["group"].eq(group)]
            if not sub.empty:
                prefix = f"mechanism_{group}"
                for col in ["n", "fraction", "mean_confidence", "mean_flip_rate", "mean_final_embedding_shift"]:
                    if col in sub.columns:
                        row[f"{prefix}_{col}"] = _safe_float(sub[col].iloc[0])

    return row


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [col for col in df.columns if col not in {*group_cols, "run_dir"} and pd.api.types.is_numeric_dtype(df[col])]
    return (
        df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
        .pipe(lambda out: out.set_axis(["_".join(c).strip("_") for c in out.columns], axis=1))
    )


def _paired_delta(df: pd.DataFrame) -> pd.DataFrame:
    teacher = df[df["stage"].eq("teacher")]
    readable = df[df["stage"].eq("risk_pro_readable")]
    merged = teacher.merge(readable, on="seed", suffixes=("_teacher", "_risk_pro_readable"))
    rows = []
    metrics = [
        "accuracy",
        "macro_f1",
        "ece",
        "vtvf_cross_errors",
        "total_errors",
        "softmax_vtvf_ambiguity_auroc",
        "knn_vtvf_mix_auroc",
        "mechanism_confident_stable_vtvf_cross_error_n",
        "mechanism_confident_stable_vtvf_cross_error_mean_confidence",
    ]
    for _, row in merged.iterrows():
        out = {"seed": int(row["seed"])}
        for metric in metrics:
            left = f"{metric}_teacher"
            right = f"{metric}_risk_pro_readable"
            if left in row and right in row:
                out[f"{metric}_delta"] = row[right] - row[left]
        rows.append(out)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _delta_summary(deltas: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for col in deltas.columns:
        if col == "seed":
            continue
        values = deltas[col].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        lo, hi = _bootstrap_ci(values, n_boot=n_boot, seed=20260626)
        rows.append(
            {
                "metric": col,
                "n": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "n_positive": int((values > 0).sum()),
                "n_negative": int((values < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate focused Risk-Pro-readable multi-seed validation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, item in manifest.iterrows():
        stage = str(item["stage"])
        if stage not in {"teacher", "risk_pro_readable"}:
            continue
        run_dir = Path(str(item["run_dir"]))
        if not (run_dir / "metrics.json").exists():
            continue
        rows.append(_metric_row(int(item["seed"]), stage, run_dir))

    args.out.mkdir(parents=True, exist_ok=True)
    run_level = pd.DataFrame(rows).sort_values(["seed", "stage"])
    run_level.to_csv(args.out / "risk_pro_readable_run_level.csv", index=False)
    _summary(run_level, ["stage"]).to_csv(args.out / "risk_pro_readable_stage_summary.csv", index=False)

    deltas = _paired_delta(run_level)
    deltas.to_csv(args.out / "risk_pro_readable_paired_deltas.csv", index=False)
    delta_summary = _delta_summary(deltas, n_boot=args.bootstrap)
    delta_summary.to_csv(args.out / "risk_pro_readable_paired_delta_summary.csv", index=False)

    report = {
        "manifest": str(args.manifest),
        "n_runs": int(len(run_level)),
        "n_seeds": int(run_level["seed"].nunique()) if not run_level.empty else 0,
        "outputs": {
            "run_level": str(args.out / "risk_pro_readable_run_level.csv"),
            "stage_summary": str(args.out / "risk_pro_readable_stage_summary.csv"),
            "paired_deltas": str(args.out / "risk_pro_readable_paired_deltas.csv"),
            "paired_delta_summary": str(args.out / "risk_pro_readable_paired_delta_summary.csv"),
        },
    }
    (args.out / "risk_pro_readable_aggregate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
