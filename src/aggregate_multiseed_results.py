from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ["SR", "VT", "VF"]
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


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "seed", "run_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return df


def _safe_metric(metrics: dict, key: str) -> float:
    value = metrics.get(key, np.nan)
    if isinstance(value, (int, float)):
        return float(value)
    return np.nan


def _per_class_from_predictions(run_dir: Path) -> list[dict[str, float | str | int]]:
    pred_path = run_dir / "test_predictions.csv"
    if not pred_path.exists():
        return []
    pred = pd.read_csv(pred_path)
    y_true = pred["y_true"].to_numpy(int)
    y_pred = pred["y_pred"].to_numpy(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return [
        {
            "class": CLASS_NAMES[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(3)
    ]


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


def _review_at_burdens(run_dir: Path) -> list[dict[str, float | str]]:
    review_path = run_dir / "review_curves.csv"
    if not review_path.exists():
        review_path = run_dir / "risk_head_review_curves.csv"
    if not review_path.exists():
        return []
    df = pd.read_csv(review_path)
    rows = []
    for burden in [0.10, 0.20, 0.30]:
        sub = df[np.isclose(df["review_burden"], burden)]
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        keep = {
            "review_burden": burden,
            "all_error_captured": row.get("all_error_captured", np.nan),
            "vtvf_error_captured": row.get("vtvf_error_captured", np.nan),
            "auto_error_rate": row.get("auto_error_rate", np.nan),
            "auto_vtvf_error_rate": row.get("auto_vtvf_error_rate", np.nan),
        }
        rows.append(keep)
    return rows


def _mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(group))
        for col in value_cols:
            if col not in group.columns:
                continue
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/multiseed_summary"))
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    per_class_rows = []
    boundary_rows = []
    review_rows = []
    split_rows = []

    for item in manifest.to_dict(orient="records"):
        run_dir = Path(item["run_dir"])
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Skipping missing metrics: {metrics_path}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "model": item["model"],
            "seed": int(item["seed"]),
            "run_dir": str(run_dir),
        }
        row.update({key: _safe_metric(metrics, key) for key in MAIN_METRICS})
        if np.isnan(row["vtvf_cross_errors"]):
            row.update(_boundary_from_predictions(run_dir))
        run_rows.append(row)

        for pc in _per_class_from_predictions(run_dir):
            per_class_rows.append({"model": item["model"], "seed": int(item["seed"]), **pc})
        boundary = _boundary_from_predictions(run_dir)
        if boundary:
            boundary_rows.append({"model": item["model"], "seed": int(item["seed"]), **boundary})
        for review in _review_at_burdens(run_dir):
            review_rows.append({"model": item["model"], "seed": int(item["seed"]), **review})

        split_path = run_dir / "split_summary.json"
        if split_path.exists():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            for split_name in ["train", "val", "test"]:
                counts = split.get(split_name)
                if counts:
                    split_rows.append(
                        {
                            "model": item["model"],
                            "seed": int(item["seed"]),
                            "split": split_name,
                            "sr_windows": counts[0],
                            "vt_windows": counts[1],
                            "vf_windows": counts[2],
                        }
                    )

    run_df = pd.DataFrame(run_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    review_df = pd.DataFrame(review_rows)
    split_df = pd.DataFrame(split_rows)

    run_df.to_csv(args.out_dir / "multiseed_run_level_metrics.csv", index=False)
    if not per_class_df.empty:
        per_class_df.to_csv(args.out_dir / "multiseed_per_class_run_level.csv", index=False)
    if not boundary_df.empty:
        boundary_df.to_csv(args.out_dir / "multiseed_boundary_run_level.csv", index=False)
    if not review_df.empty:
        review_df.to_csv(args.out_dir / "multiseed_review_run_level.csv", index=False)
    if not split_df.empty:
        split_df.to_csv(args.out_dir / "multiseed_split_run_level.csv", index=False)

    model_summary = _mean_std(run_df, ["model"], [c for c in MAIN_METRICS if c in run_df.columns])
    model_summary.to_csv(args.out_dir / "multiseed_model_summary_mean_std.csv", index=False)

    if not per_class_df.empty:
        per_class_summary = _mean_std(per_class_df, ["model", "class"], ["precision", "recall", "f1", "support"])
        per_class_summary.to_csv(args.out_dir / "multiseed_per_class_summary_mean_std.csv", index=False)

    if not boundary_df.empty:
        boundary_summary = _mean_std(
            boundary_df,
            ["model"],
            ["sr_as_vt", "sr_as_vf", "vt_as_sr", "vt_as_vf", "vf_as_sr", "vf_as_vt", "vtvf_cross_errors", "total_errors"],
        )
        boundary_summary.to_csv(args.out_dir / "multiseed_boundary_summary_mean_std.csv", index=False)

    if not review_df.empty:
        review_summary = _mean_std(
            review_df,
            ["model", "review_burden"],
            ["all_error_captured", "vtvf_error_captured", "auto_error_rate", "auto_vtvf_error_rate"],
        )
        review_summary.to_csv(args.out_dir / "multiseed_review_summary_mean_std.csv", index=False)

    print(f"Wrote multiseed summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
