from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


MAIN_METRICS = [
    "accuracy",
    "macro_f1",
    "nll",
    "ece",
    "macro_sensitivity",
    "macro_specificity",
    "vtvf_cross_errors",
    "total_errors",
    "vt_as_vf",
    "vf_as_vt",
]


def _safe_metric(metrics: dict, key: str) -> float:
    value = metrics.get(key, np.nan)
    if isinstance(value, (int, float)):
        return float(value)
    return np.nan


def _boundary_from_predictions(run_dir: Path) -> dict[str, int]:
    pred_path = run_dir / "test_predictions.csv"
    if not pred_path.exists():
        return {}
    pred = pd.read_csv(pred_path)
    y_true = pred["y_true"].to_numpy(int)
    y_pred = pred["y_pred"].to_numpy(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "sr_as_vt": int(cm[0, 1]),
        "sr_as_vf": int(cm[0, 2]),
        "vt_as_sr": int(cm[1, 0]),
        "vt_as_vf": int(cm[1, 2]),
        "vf_as_sr": int(cm[2, 0]),
        "vf_as_vt": int(cm[2, 1]),
        "vtvf_cross_errors": int(cm[1, 2] + cm[2, 1]),
        "total_errors": int((y_true != y_pred).sum()),
    }


def _review_rows(run_dir: Path, variant: str, seed: int) -> list[dict[str, float | str | int]]:
    rows = []
    candidates = [
        ("selective", run_dir / "review_efficiency_curves.csv", "boundary_lrii"),
        ("stability", run_dir / "stability_review_curves.csv", "stability_aware_risk"),
    ]
    for source, path, score in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "score" in df.columns:
            df = df[df["score"] == score]
        for burden in [0.10, 0.20, 0.30]:
            sub = df[np.isclose(df["review_burden"], burden)]
            if sub.empty:
                continue
            item = sub.iloc[0]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "source": source,
                    "score": score,
                    "review_burden": burden,
                    "all_error_captured": float(item.get("all_error_captured", np.nan)),
                    "vtvf_error_captured": float(item.get("vtvf_error_captured", np.nan)),
                    "auto_error_rate": float(item.get("auto_error_rate", np.nan)),
                    "auto_vtvf_error_rate": float(item.get("auto_vtvf_error_rate", np.nan)),
                }
            )
    return rows


def _stability_rows(run_dir: Path, variant: str, seed: int) -> list[dict[str, float | str | int]]:
    path = run_dir / "stability_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for item in df.to_dict(orient="records"):
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "group": item["group"],
                "n": int(item["n"]),
                "pred_flip_rate_mean": float(item["pred_flip_rate_mean"]),
                "embedding_drift_mean": float(item["embedding_drift_mean"]),
                "stability_aware_risk_mean": float(item["stability_aware_risk_mean"]),
            }
        )
    return rows


def _routing_rows(run_dir: Path, variant: str, seed: int) -> list[dict[str, float | str | int]]:
    path = run_dir / "ambiguity_routing_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for item in df.to_dict(orient="records"):
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "decision": item["decision"],
                "n": int(item["n"]),
                "rate": float(item["rate"]),
                "error_rate": float(item["error_rate"]),
                "vtvf_cross_error_rate": float(item["vtvf_cross_error_rate"]),
                "vtvf_cross_errors": int(item["vtvf_cross_errors"]),
            }
        )
    return rows


def _mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(group))
        for col in value_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/mitigation_summary"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    review_rows = []
    stability_rows = []
    routing_rows = []

    for row in manifest.to_dict(orient="records"):
        run_dir = Path(row["run_dir"])
        variant = str(row["variant"])
        seed = int(row["seed"])
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Skipping missing metrics: {metrics_path}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        out = {"variant": variant, "model": row["model"], "seed": seed, "run_dir": str(run_dir)}
        out.update({key: _safe_metric(metrics, key) for key in MAIN_METRICS})
        if np.isnan(out["vtvf_cross_errors"]):
            out.update(_boundary_from_predictions(run_dir))
        run_rows.append(out)
        review_rows.extend(_review_rows(run_dir, variant, seed))
        stability_rows.extend(_stability_rows(run_dir, variant, seed))
        routing_rows.extend(_routing_rows(run_dir, variant, seed))

    run_df = pd.DataFrame(run_rows)
    review_df = pd.DataFrame(review_rows)
    stability_df = pd.DataFrame(stability_rows)
    routing_df = pd.DataFrame(routing_rows)
    run_df.to_csv(args.out_dir / "mitigation_run_level_metrics.csv", index=False)
    if not review_df.empty:
        review_df.to_csv(args.out_dir / "mitigation_review_run_level.csv", index=False)
    if not stability_df.empty:
        stability_df.to_csv(args.out_dir / "mitigation_stability_run_level.csv", index=False)
    if not routing_df.empty:
        routing_df.to_csv(args.out_dir / "mitigation_routing_run_level.csv", index=False)

    _mean_std(run_df, ["variant"], [c for c in MAIN_METRICS if c in run_df.columns]).to_csv(
        args.out_dir / "mitigation_model_summary_mean_std.csv", index=False
    )
    if not review_df.empty:
        _mean_std(
            review_df,
            ["variant", "source", "review_burden"],
            ["all_error_captured", "vtvf_error_captured", "auto_error_rate", "auto_vtvf_error_rate"],
        ).to_csv(args.out_dir / "mitigation_review_summary_mean_std.csv", index=False)
    if not stability_df.empty:
        _mean_std(
            stability_df,
            ["variant", "group"],
            ["n", "pred_flip_rate_mean", "embedding_drift_mean", "stability_aware_risk_mean"],
        ).to_csv(args.out_dir / "mitigation_stability_summary_mean_std.csv", index=False)
    if not routing_df.empty:
        _mean_std(
            routing_df,
            ["variant", "decision"],
            ["n", "rate", "error_rate", "vtvf_cross_error_rate", "vtvf_cross_errors"],
        ).to_csv(args.out_dir / "mitigation_routing_summary_mean_std.csv", index=False)

    print(f"Wrote mitigation summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
