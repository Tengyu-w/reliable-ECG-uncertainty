from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(sub))
        for col in value_cols:
            values = sub[col].dropna().to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_row(manifest_row: pd.Series) -> dict[str, float | str | int]:
    run_dir = Path(str(manifest_row["run_dir"]))
    metrics = _read_json(run_dir / "metrics.json")
    row: dict[str, float | str | int] = {
        "stage": manifest_row["stage"],
        "model": manifest_row["model"],
        "seed": int(manifest_row["seed"]),
        "run_dir": str(run_dir),
        "accuracy": float(metrics.get("accuracy", np.nan)),
        "macro_f1": float(metrics.get("macro_f1", np.nan)),
        "nll": float(metrics.get("nll", np.nan)),
        "ece": float(metrics.get("ece", np.nan)),
    }
    cm = np.asarray(metrics.get("confusion_matrix", []), dtype=float)
    if cm.shape == (3, 3):
        row["vtvf_cross_errors"] = float(cm[1, 2] + cm[2, 1])
        row["vt_as_vf"] = float(cm[1, 2])
        row["vf_as_vt"] = float(cm[2, 1])
    geom = run_dir / "embedding_geometry_summary.csv"
    if geom.exists():
        g = pd.read_csv(geom).iloc[0]
        for col in [
            "sr_vt_norm_dist",
            "sr_vf_norm_dist",
            "vt_vf_norm_dist",
            "purity_k15_mean",
            "purity_k15_vt",
            "purity_k15_vf",
            "vtvf_mixing_k15_ventricular",
            "error_mean_local_purity",
            "correct_mean_local_purity",
        ]:
            if col in g:
                row[col] = float(g[col])
    return row


def _review_rows(manifest_row: pd.Series) -> list[dict[str, float | str | int]]:
    run_dir = Path(str(manifest_row["run_dir"]))
    path = run_dir / "review_efficiency_curves.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for score in ["entropy", "local_instability", "vtvf_mixing", "lrii", "boundary_lrii", "atypicality_lrii"]:
        for burden in [0.10, 0.20, 0.30]:
            sub = df[(df["score"] == score) & np.isclose(df["review_burden"].astype(float), burden)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "stage": manifest_row["stage"],
                    "model": manifest_row["model"],
                    "seed": int(manifest_row["seed"]),
                    "score": score,
                    "review_burden": burden,
                    "all_error_captured": float(r["all_error_captured"]),
                    "vtvf_error_captured": float(r["vtvf_error_captured"]),
                    "auto_error_rate": float(r["auto_error_rate"]),
                    "auto_vtvf_error_rate": float(r["auto_vtvf_error_rate"]),
                }
            )
    return rows


def _risk_row(manifest_row: pd.Series) -> dict[str, float | str | int]:
    run_dir = Path(str(manifest_row["run_dir"]))
    summary = _read_json(run_dir / "summary.json")["test"]
    row: dict[str, float | str | int] = {
        "stage": manifest_row["stage"],
        "model": manifest_row["model"],
        "seed": int(manifest_row["seed"]),
        "run_dir": str(run_dir),
    }
    for key, value in summary.items():
        row[key] = float(value)
    curves = run_dir / "review_curves.csv"
    if curves.exists():
        df = pd.read_csv(curves)
        for burden in [0.10, 0.20, 0.30]:
            sub = df[np.isclose(df["review_burden"].astype(float), burden)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            prefix = f"review_{int(burden * 100)}"
            row[f"{prefix}_vtvf_error_captured"] = float(r["vtvf_error_captured"])
            row[f"{prefix}_all_error_captured"] = float(r["all_error_captured"])
            row[f"{prefix}_auto_error_rate"] = float(r["auto_error_rate"])
            row[f"{prefix}_auto_vtvf_error_rate"] = float(r["auto_vtvf_error_rate"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    out_dir = args.out_dir or args.manifest.parent / "core_validation_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []
    review_rows = []
    for _, row in manifest[manifest["stage"].isin(["waveform_only_baseline", "regularity_feature_injection"])].iterrows():
        run_dir = Path(str(row["run_dir"]))
        if (run_dir / "metrics.json").exists():
            model_rows.append(_metrics_row(row))
            review_rows.extend(_review_rows(row))
    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df.to_csv(out_dir / "regularity_model_run_level.csv", index=False)
        value_cols = [c for c in model_df.columns if c not in {"stage", "model", "seed", "run_dir"}]
        _mean_std(model_df, ["stage", "model"], value_cols).to_csv(
            out_dir / "regularity_model_mean_std.csv", index=False
        )

    review_df = pd.DataFrame(review_rows)
    if not review_df.empty:
        review_df.to_csv(out_dir / "regularity_review_run_level.csv", index=False)
        value_cols = [
            "all_error_captured",
            "vtvf_error_captured",
            "auto_error_rate",
            "auto_vtvf_error_rate",
        ]
        _mean_std(review_df, ["stage", "model", "score", "review_burden"], value_cols).to_csv(
            out_dir / "regularity_review_mean_std.csv", index=False
        )

    risk_rows = []
    for _, row in manifest[manifest["stage"] == "risk_aligned_distillation"].iterrows():
        run_dir = Path(str(row["run_dir"]))
        if (run_dir / "summary.json").exists():
            risk_rows.append(_risk_row(row))
    risk_df = pd.DataFrame(risk_rows)
    if not risk_df.empty:
        risk_df.to_csv(out_dir / "risk_distillation_run_level.csv", index=False)
        value_cols = [c for c in risk_df.columns if c not in {"stage", "model", "seed", "run_dir"}]
        _mean_std(risk_df, ["stage", "model"], value_cols).to_csv(
            out_dir / "risk_distillation_mean_std.csv", index=False
        )

    print(out_dir)


if __name__ == "__main__":
    main()
