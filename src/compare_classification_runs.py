from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import classification_metrics, expected_calibration_error, softmax


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "latest" and run_dir.is_file():
        return Path(run_dir.read_text(encoding="utf-8").strip())
    return run_dir


def _load_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = run_dir / "test_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        col_lookup = {c.lower(): c for c in df.columns}
        prob_cols = [col_lookup[c] for c in ["prob_sr", "prob_vt", "prob_vf"] if c in col_lookup]
        if {"y_true", "y_pred"}.issubset(df.columns) and prob_cols:
            probs = df[prob_cols].to_numpy(float)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            return df["y_true"].to_numpy(int), probs, df["y_pred"].to_numpy(int)

    emb_path = run_dir / "embeddings_test.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Cannot find test_predictions.csv or embeddings_test.npz in {run_dir}")
    data = np.load(emb_path)
    probs = softmax(data["logits"])
    return data["y"].astype(int), probs, probs.argmax(axis=1)


def _probability_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def _summarise_run(name: str, run_dir: Path) -> dict[str, float | int | str]:
    y_true, probs, y_pred = _load_predictions(run_dir)
    metrics = classification_metrics(y_true, probs)
    margin = _probability_margin(probs)
    vtvf_mask = np.isin(y_true, [1, 2])
    error_mask = y_true != y_pred
    row: dict[str, float | int | str] = {
        "model": name,
        "run_dir": str(run_dir),
        "n_test": int(len(y_true)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "ece": expected_calibration_error(y_true, probs),
        "nll": float(metrics["nll"]),
        "total_errors": int(metrics["total_errors"]),
        "vtvf_cross_errors": int(metrics["vtvf_cross_errors"]),
        "vt_as_vf": int(metrics["vt_as_vf"]),
        "vf_as_vt": int(metrics["vf_as_vt"]),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "low_margin_rate_0p10": float((margin < 0.10).mean()),
        "low_margin_rate_0p20": float((margin < 0.20).mean()),
        "mean_margin_correct": float(margin[~error_mask].mean()) if (~error_mask).any() else np.nan,
        "mean_margin_error": float(margin[error_mask].mean()) if error_mask.any() else np.nan,
        "mean_margin_vtvf": float(margin[vtvf_mask].mean()) if vtvf_mask.any() else np.nan,
        "vtvf_error_rate_within_vtvf_truth": float(
            (
                ((y_true == 1) & (y_pred == 2))
                | ((y_true == 2) & (y_pred == 1))
            )[vtvf_mask].mean()
        )
        if vtvf_mask.any()
        else np.nan,
    }
    for item in metrics["per_class"]:
        label = ["sr", "vt", "vf"][int(item["class_index"])]
        row[f"{label}_precision"] = float(item["precision"])
        row[f"{label}_recall"] = float(item["recall"])
        row[f"{label}_f1"] = float(item["f1"])
        row[f"{label}_support"] = int(item["support"])
    return row


def _paired_delta(summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if baseline not in set(summary["model"]):
        return pd.DataFrame()
    base = summary[summary["model"].eq(baseline)].iloc[0]
    rows = []
    for _, row in summary.iterrows():
        if row["model"] == baseline:
            continue
        rows.append(
            {
                "baseline": baseline,
                "comparator": row["model"],
                "accuracy_delta": float(row["accuracy"] - base["accuracy"]),
                "macro_f1_delta": float(row["macro_f1"] - base["macro_f1"]),
                "ece_delta": float(row["ece"] - base["ece"]),
                "mean_margin_delta": float(row["mean_margin"] - base["mean_margin"]),
                "vtvf_cross_errors_delta": int(row["vtvf_cross_errors"] - base["vtvf_cross_errors"]),
                "total_errors_delta": int(row["total_errors"] - base["total_errors"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare classification runs with accuracy, macro-F1, ECE, probability margin, and VT/VF errors."
    )
    parser.add_argument("--run", nargs=2, action="append", metavar=("MODEL_NAME", "RUN_DIR"), required=True)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/model_run_comparison"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, run_dir_text in args.run:
        rows.append(_summarise_run(name, _resolve_run_dir(Path(run_dir_text))))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "classification_run_comparison.csv", index=False)

    baseline = args.baseline or str(summary.iloc[0]["model"])
    delta = _paired_delta(summary, baseline)
    if not delta.empty:
        delta.to_csv(args.out / "classification_run_deltas.csv", index=False)

    report = {
        "baseline": baseline,
        "runs": summary.to_dict(orient="records"),
        "deltas": delta.to_dict(orient="records"),
        "metric_note": {
            "margin": "top-1 probability minus top-2 probability; lower means the classifier is less decisive.",
            "low_margin_rate_0p10": "fraction of samples whose top-1/top-2 probability gap is below 0.10.",
        },
    }
    (args.out / "classification_run_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary)
    if not delta.empty:
        print(delta)


if __name__ == "__main__":
    main()
